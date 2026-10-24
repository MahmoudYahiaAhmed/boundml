"""End-to-end tests of the Python package on the repository's breast-cancer example."""

import csv
import os
import pathlib
import shutil
import subprocess
import threading

import numpy as np
import pytest

import boundml

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFIG = REPO / "examples" / "breast_cancer" / "linear.toml"
TEST_CSV = REPO / "datasets" / "breast_cancer" / "test.csv"
CLI = REPO / "target" / "release" / ("boundml.exe" if os.name == "nt" else "boundml")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("bc with spaces")
    result = boundml.train(CONFIG, d / "ckpt")
    model = boundml.pack(d / "ckpt", d / "model.boundml")
    return d, result, model


def rows():
    with TEST_CSV.open() as f:
        return list(csv.DictReader(f))


def test_train_evaluate_pack(trained):
    d, result, model = trained
    assert result["completed"] and result["global_step"] > 0 and result["history"]
    metrics = boundml.evaluate(d / "ckpt", TEST_CSV)
    assert metrics["model"]["accuracy"] == pytest.approx(0.9597, abs=1e-4)
    assert metrics["baseline"]["accuracy"] < metrics["model"]["accuracy"]
    assert boundml.evaluate(d / "model.boundml", TEST_CSV)["model"] == metrics["model"]
    assert model.verify()["passed"] is True
    assert model.outputs and len(model.features) == 30


def test_all_input_forms_agree(trained):
    _, _, model = trained
    recs = [{k: float(r[k]) for k in model.features} for r in rows()]
    x = np.array([[r[k] for k in model.features] for r in recs])
    by_array = model.predict(x)
    assert by_array.shape == (len(recs), len(model.outputs)) and by_array.dtype == np.float32
    np.testing.assert_array_equal(model.predict(recs), by_array)
    np.testing.assert_array_equal(model.predict(recs[0]), by_array[0])
    np.testing.assert_array_equal(model.predict(x[0]), by_array[0])
    np.testing.assert_array_equal(model.predict(x.tolist()), by_array)
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(recs)[list(reversed(model.features))]  # column order must not matter
    np.testing.assert_array_equal(model.predict(df), by_array)
    assert model.predict(x[:0]).shape == (0, len(model.outputs))


@pytest.mark.skipif(not CLI.exists(), reason="needs target/release/boundml (cargo build --release)")
def test_predictions_match_the_cli_bitwise(trained, tmp_path):
    d, _, model = trained
    out = tmp_path / "pred.csv"
    subprocess.run([CLI, "predict", d / "model.boundml", "--data", TEST_CSV, "--out", out], check=True,
                   capture_output=True)
    with out.open() as f:
        cli = np.array([[float(r[k]) for k in model.outputs] for r in csv.DictReader(f)], dtype=np.float32)
    recs = [{k: float(r[k]) for k in model.features} for r in rows()]
    np.testing.assert_array_equal(model.predict(recs), cli)


def test_errors_carry_stable_codes(trained, tmp_path):
    d, _, model = trained
    with pytest.raises(boundml.BoundMLError) as e:
        model.predict(np.zeros((2, 3)))
    assert e.value.code == "BML-D002"
    with pytest.raises(boundml.BoundMLError) as e:
        model.predict({k: float("inf") for k in model.features})
    assert e.value.code.startswith("BML-")
    with pytest.raises(boundml.BoundMLError) as e:
        boundml.load(tmp_path / "missing")
    assert e.value.code == "BML-A001"
    broken = tmp_path / "broken.boundml"
    shutil.copytree(d / "model.boundml", broken)
    w = broken / "weights.safetensors"
    w.write_bytes(w.read_bytes()[:-4] + b"\0\0\0\0")
    with pytest.raises(boundml.BoundMLError) as e:
        boundml.load(broken)
    assert e.value.code == "BML-A004"  # integrity (hash) mismatch


def test_missing_values_follow_the_schema_policy(trained):
    _, _, model = trained
    one = {k: 1.0 for k in model.features}
    one[model.features[0]] = None
    policy = model.manifest["schema"]["missing"]
    if policy == "reject":
        with pytest.raises(boundml.BoundMLError):
            model.predict(one)
    else:
        assert np.isfinite(model.predict(one)).all()


def test_memory_report_and_threads(trained):
    _, _, model = trained
    assert model.memory_report(1)["total_planned_bytes"] > 0
    x = np.array([[float(r[k]) for k in model.features] for r in rows()])
    expected = model.predict(x)
    results = []
    threads = [threading.Thread(target=lambda: results.append(model.predict(x))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 4 and all(np.array_equal(r, expected) for r in results)


def test_stopped_training_resumes_to_the_same_result(trained, tmp_path):
    d, full, _ = trained
    part = boundml.train(CONFIG, tmp_path / "ckpt", max_steps=5)
    assert not part["completed"] and part["global_step"] == 5
    done = boundml.train(CONFIG, tmp_path / "ckpt", resume=True)
    assert done["completed"] and done["global_step"] == full["global_step"]
    assert boundml.evaluate(tmp_path / "ckpt", TEST_CSV) == boundml.evaluate(d / "ckpt", TEST_CSV)


def test_dataframe_with_non_numeric_extra_columns(trained):
    # The real test CSV, as read by pandas: 'id' and the string label 'diagnosis'
    # ("M"/"B") are not features; the schema ignores extra columns, as the CLI does.
    pd = pytest.importorskip("pandas")
    _, _, model = trained
    df = pd.read_csv(TEST_CSV)
    assert not pd.api.types.is_numeric_dtype(df["diagnosis"])  # object or StringDtype
    recs = [{k: float(r[k]) for k in model.features} for r in rows()]
    expected = model.predict(recs)
    np.testing.assert_array_equal(model.predict(df), expected)
    with_labels = [{**r, "diagnosis": "M", "id": "x"} for r in recs]
    np.testing.assert_array_equal(model.predict(with_labels), expected)


def test_non_numeric_feature_values_raise_bml_d005(trained):
    pd = pytest.importorskip("pandas")
    _, _, model = trained
    row = {k: 1.0 for k in model.features}
    bad = {**row, model.features[0]: "abc"}
    for data in (bad, [bad], pd.DataFrame([bad]), [[*(["abc"]), *([1.0] * (len(model.features) - 1))]]):
        with pytest.raises(boundml.BoundMLError) as e:
            model.predict(data)
        assert e.value.code == "BML-D005", type(data)
    # Numeric strings are accepted like CSV cells; missing values follow the policy.
    assert np.isfinite(model.predict({**row, model.features[0]: "1.5"})).all()


def test_unknown_columns_under_a_reject_schema_are_errors(tmp_path):
    pd = pytest.importorskip("pandas")
    # Copies of the training data without the 'id' column, which a strict schema refuses.
    src = (CONFIG.parent / "../../datasets/breast_cancer").resolve()
    for name in ("train.csv", "valid.csv"):
        with (src / name).open(newline="") as f:
            data = [{k: v for k, v in r.items() if k != "id"} for r in csv.DictReader(f)]
        with (tmp_path / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    text = CONFIG.read_text().replace('extra_columns = "ignore"', 'extra_columns = "reject"')
    text = text.replace('"../../datasets/breast_cancer/', f'"{tmp_path.as_posix()}/')
    assert 'extra_columns = "reject"' in text
    cfg = tmp_path / "strict.toml"
    cfg.write_text(text)
    boundml.train(cfg, tmp_path / "ckpt")
    model = boundml.pack(tmp_path / "ckpt", tmp_path / "strict.boundml")
    row = {k: 1.0 for k in model.features}
    assert model.predict(row).shape == (len(model.outputs),)
    for data in ({**row, "diagnosis": "M"}, pd.DataFrame([{**row, "diagnosis": "M"}])):
        with pytest.raises(boundml.BoundMLError) as e:
            model.predict(data)
        assert "diagnosis" in str(e.value) and e.value.code.startswith("BML-")
