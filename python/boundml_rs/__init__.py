"""BoundML for Python: train and run small numeric models whose schema, fitted
preprocessing, self-tests and memory plan travel inside one package.

    import boundml_rs
    boundml_rs.train("model.toml", "ckpt")
    model = boundml_rs.pack("ckpt", "model.boundml")
    model.predict({"temperature": 20, "humidity": None, "load": 0.5})

The computation is the same native Rust code as the ``boundml`` CLI, so a package
gives identical predictions from Python, the CLI, the HTTP server and Rust.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, Union

import numpy as np

from . import _native
from ._errors import BoundMLError

__all__ = ["BoundMLError", "Model", "load", "pack", "train", "evaluate", "__version__"]
__version__: str = _native.__version__

PathLike = Union[str, "os.PathLike[str]"]


class Model:
    """A loaded, validated BoundML package. Thread-safe; predictions release the GIL."""

    def __init__(self, native: Any):
        self._native = native
        self._manifest: dict = json.loads(native.manifest_json())

    @classmethod
    def load(cls, path: PathLike) -> "Model":
        """Load a package directory (integrity, version, graph and plan are checked)."""
        return cls(_native.NativeModel.load(os.fspath(path)))

    @property
    def name(self) -> str:
        return self._manifest["name"]

    @property
    def features(self) -> list[str]:
        """Input feature names, in schema order."""
        return list(self._manifest["schema"]["features"])

    @property
    def outputs(self) -> list[str]:
        """Names of the output columns returned by :meth:`predict`."""
        return list(self._native.output_names())

    @property
    def task(self) -> Any:
        return self._manifest["task"]

    @property
    def manifest(self) -> dict:
        """The package manifest (schema, preprocessing, bounds, provenance, ...)."""
        return json.loads(json.dumps(self._manifest))

    def memory_report(self, sessions: int = 1) -> dict:
        """Planned tensor memory for ``sessions`` concurrent sessions."""
        return json.loads(self._native.memory_report_json(sessions))

    def verify(self) -> dict:
        """Re-run the bundled self-test; raises nothing, check ``result["passed"]``."""
        return json.loads(self._native.verify_json())

    def predict(self, data: Any) -> np.ndarray:
        """Predict from raw (unpreprocessed) values.

        ``data`` may be:

        * a pandas DataFrame: columns are matched by name;
        * a mapping (one row) or a sequence of mappings: keys are feature names;
        * a NumPy array or nested list of shape ``(rows, n_features)`` (or one row of
          ``n_features``), in :attr:`features` order.

        Missing values are ``None`` or NaN and follow the schema's missing policy.
        Returns float32 ``(rows, len(outputs))``; a single mapping or 1-D row gives
        ``(len(outputs),)``.
        """
        if _is_dataframe(data):
            names = [str(c) for c in data.columns]
            return self._predict_columns(names, len(data), lambda j: _column_floats(data.iloc[:, j], names[j]))
        if isinstance(data, Mapping):
            return self._predict_records([data])[0]
        if isinstance(data, Sequence) and data and all(isinstance(r, Mapping) for r in data):
            return self._predict_records(data)
        try:
            x = np.asarray(_none_to_nan(data), dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise BoundMLError(f"BML-D005 input values must be numbers: {e}", "BML-D005") from None
        if x.ndim == 1:
            return self._native.predict_array(_matrix(x.reshape(1, -1)))[0]
        return self._native.predict_array(_matrix(x))

    def _predict_records(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        names = [str(n) for n in dict.fromkeys(k for r in rows for k in r)]

        def column(j: int) -> np.ndarray:
            return np.array([_cell(r.get(names[j]), names[j]) for r in rows], dtype=np.float64)

        return self._predict_columns(names, len(rows), column)

    def _predict_columns(self, names: list[str], rows: int, column) -> np.ndarray:
        """Predict from named columns. Only feature columns are converted to numbers.

        Unknown columns are dropped when the schema ignores extra columns. When it
        rejects them they are passed on as empty placeholders, so the native layer
        raises the same error as the CLI and the HTTP server.
        """
        features = set(self.features)
        reject = self._manifest["schema"].get("extra_columns") == "reject"
        keep = [j for j, n in enumerate(names) if n in features or reject]
        x = np.full((rows, len(keep)), np.nan)
        for c, j in enumerate(keep):
            if names[j] in features:
                x[:, c] = column(j)
        return self._native.predict_named([names[j] for j in keep], x)

    def __repr__(self) -> str:
        return f"<boundml_rs.Model {self.name!r}: {len(self.features)} features -> {self.outputs}>"


def _is_dataframe(data: Any) -> bool:
    return hasattr(data, "columns") and hasattr(data, "to_numpy") and getattr(data, "ndim", 0) == 2


def _cell(v: Any, name: str) -> float:
    """One raw feature value: None/NaN = missing; numbers (and numeric strings) as float."""
    if v is None:
        return np.nan
    if isinstance(v, bool):
        raise BoundMLError(f"BML-D005 feature '{name}': boolean value {v!r} is not a number", "BML-D005")
    try:
        return float(v)
    except (TypeError, ValueError):
        raise BoundMLError(f"BML-D005 feature '{name}': value {v!r} is not a number", "BML-D005") from None


def _column_floats(col: Any, name: str) -> np.ndarray:
    """A DataFrame column as float64 (missing = NaN), or BoundMLError BML-D005."""
    try:
        return col.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError):
        return np.array([_cell(None if _is_missing(v) else v, name) for v in col.tolist()], dtype=np.float64)


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and v != v) or str(v) in ("<NA>", "NaT")


def _none_to_nan(data: Any) -> Any:
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, (list, tuple)):
        return [_none_to_nan(v) for v in data]
    return np.nan if data is None else data


def _matrix(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise BoundMLError(f"BML-D002 expected a 2-D table of rows, got {x.ndim} dimension(s)", "BML-D002")
    return np.ascontiguousarray(x, dtype=np.float64)


def load(path: PathLike) -> Model:
    """Load a package directory. Same as :meth:`Model.load`."""
    return Model.load(path)


def train(config: PathLike, checkpoint: PathLike, *, resume: bool = False,
          max_steps: int | None = None, verbose: bool = False) -> dict:
    """Train from a TOML config (see docs/quickstart.md) into ``checkpoint``.

    Returns ``{"completed", "global_step", "history", "memory"}``. The memory plan is
    checked against the config's budget before training starts. Ctrl-C saves a
    checkpoint and raises KeyboardInterrupt; resume with ``resume=True``.
    """
    return json.loads(_native.train_json(os.fspath(config), os.fspath(checkpoint), resume, max_steps, verbose))


def evaluate(path: PathLike, data: PathLike) -> dict:
    """Metrics of a checkpoint or package on a labelled CSV: ``{"model", "baseline"}``."""
    return json.loads(_native.evaluate_json(os.fspath(path), os.fspath(data)))


def pack(checkpoint: PathLike, out: PathLike, name: str | None = None) -> Model:
    """Pack a checkpoint into a self-verifying package directory and return it loaded."""
    return Model(_native.pack(os.fspath(checkpoint), os.fspath(out), name))
