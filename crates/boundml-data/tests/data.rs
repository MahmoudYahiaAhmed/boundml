use boundml_core::Rng;
use boundml_data::*;
use std::io::Write;
use std::path::PathBuf;

fn write(dir: &tempfile::TempDir, name: &str, content: &str) -> PathBuf {
    let p = dir.path().join(name);
    std::fs::File::create(&p)
        .unwrap()
        .write_all(content.as_bytes())
        .unwrap();
    p
}

fn schema(missing: MissingPolicy, extra: ExtraColumns) -> Schema {
    Schema {
        features: vec!["a".into(), "b".into()],
        label: "y".into(),
        label_kind: LabelKind::Classes {
            classes: vec!["no".into(), "yes".into()],
        },
        missing,
        extra_columns: extra,
    }
}

#[test]
fn maps_by_name_and_fits_on_train_only() {
    let d = tempfile::tempdir().unwrap();
    // Columns deliberately out of order, plus an extra id column.
    let p = write(&d, "t.csv", "id,y,b,a\n1,no,10,1\n2,yes,20,3\n3,yes,30,5\n");
    let s = schema(MissingPolicy::Reject, ExtraColumns::Ignore);
    let pre = Preprocessor::fit(&p, &s, Normalize::Standardize).unwrap();
    assert_eq!(pre.features[0].name, "a");
    assert!((pre.features[0].mean - 3.0).abs() < 1e-12);
    assert!((pre.features[1].mean - 20.0).abs() < 1e-12);
    let std_a = (8.0f64 / 3.0).sqrt();
    assert!((pre.features[0].std - std_a).abs() < 1e-12);

    let mut r = BatchReader::open(&p, &s, &pre, 1, Rng::seed(0), true).unwrap();
    let (mut x, mut y, mut l) = (vec![0.0; 6], vec![0.0; 3], vec![0; 3]);
    assert_eq!(r.next_batch(3, &mut x, &mut y, &mut l).unwrap(), 3);
    assert_eq!(y, vec![0.0, 1.0, 1.0]);
    assert!((x[0] as f64 - (1.0 - 3.0) / std_a).abs() < 1e-6);
    assert_eq!(l, vec![2, 3, 4]);
    assert_eq!(r.next_batch(3, &mut x, &mut y, &mut l).unwrap(), 0);
}

#[test]
fn header_policies() {
    let d = tempfile::tempdir().unwrap();
    let s = schema(MissingPolicy::Reject, ExtraColumns::Reject);
    let missing_col = write(&d, "m.csv", "a,y\n1,no\n");
    let e = CsvSource::open(&missing_col, &s, true)
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("missing feature column(s): b"), "{e}");
    let dup = write(&d, "d.csv", "a,b,a,y\n1,2,3,no\n");
    assert!(
        CsvSource::open(&dup, &s, true)
            .err()
            .unwrap()
            .to_string()
            .contains("duplicate")
    );
    let extra = write(&d, "e.csv", "a,b,y,z\n1,2,no,3\n");
    assert!(
        CsvSource::open(&extra, &s, true)
            .err()
            .unwrap()
            .to_string()
            .contains("unexpected")
    );
    let nolabel = write(&d, "n.csv", "a,b\n1,2\n");
    assert!(CsvSource::open(&nolabel, &s, true).is_err());
    assert!(CsvSource::open(&nolabel, &s, false).is_ok());
}

#[test]
fn values_missing_and_nonfinite() {
    let d = tempfile::tempdir().unwrap();
    let s = schema(MissingPolicy::Reject, ExtraColumns::Ignore);
    for (content, needle) in [
        ("a,b,y\n1,,no\n", "missing value"),
        ("a,b,y\n1,NaN,no\n", "non-finite"),
        ("a,b,y\n1,inf,no\n", "non-finite"),
        ("a,b,y\n1,abc,no\n", "not a number"),
        ("a,b,y\n1,2,maybe\n", "unknown class"),
        ("a,b,y\n1,2,\n", "label is missing"),
        ("a,b,y\n1,2\n", "CSV format"),
    ] {
        let p = write(&d, "v.csv", content);
        let e = Preprocessor::fit(&p, &s, Normalize::Standardize)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains(needle), "{content:?}: {e}");
    }
    let empty = write(&d, "empty.csv", "a,b,y\n");
    assert!(Preprocessor::fit(&empty, &s, Normalize::Standardize).is_err());
}

#[test]
fn impute_mean_and_constant_columns() {
    let d = tempfile::tempdir().unwrap();
    let s = schema(MissingPolicy::ImputeMean, ExtraColumns::Ignore);
    let p = write(&d, "t.csv", "a,b,y\n1,7,no\n,7,yes\n3,7,no\n");
    let pre = Preprocessor::fit(&p, &s, Normalize::Standardize).unwrap();
    assert_eq!(pre.features[0].missing, 1);
    assert!((pre.features[0].mean - 2.0).abs() < 1e-12);
    assert!(pre.features[1].constant);
    let mut out = [0.0f32; 2];
    pre.apply(&[None, Some(7.0)], &mut out).unwrap();
    assert_eq!(out, [0.0, 0.0]);
    let json = serde_json::to_string(&pre).unwrap();
    let back: Preprocessor = serde_json::from_str(&json).unwrap();
    assert_eq!(back, pre);
    back.validate(&s).unwrap();
    let mut other = s.clone();
    other.features.swap(0, 1);
    assert!(back.validate(&other).is_err());
}

#[test]
fn shuffle_is_bounded_deterministic_and_resumable() {
    let d = tempfile::tempdir().unwrap();
    let mut csv = String::from("a,b,y\n");
    for i in 0..103 {
        csv.push_str(&format!(
            "{i},{},{}\n",
            i * 2,
            if i % 2 == 0 { "no" } else { "yes" }
        ));
    }
    let p = write(&d, "t.csv", &csv);
    let s = schema(MissingPolicy::Reject, ExtraColumns::Ignore);
    let pre = Preprocessor::fit(&p, &s, Normalize::None).unwrap();
    let collect = |r: &mut BatchReader, batches: Option<usize>| {
        let (mut x, mut y, mut l) = (vec![0.0; 20], vec![0.0; 10], vec![0; 10]);
        let mut seen = Vec::new();
        let mut k = 0;
        loop {
            if batches == Some(k) {
                break;
            }
            let n = r.next_batch(10, &mut x, &mut y, &mut l).unwrap();
            if n == 0 {
                break;
            }
            seen.extend(x[..2 * n].chunks(2).map(|c| c[0] as i64));
            k += 1;
        }
        seen
    };
    let mut full = BatchReader::open(&p, &s, &pre, 16, Rng::seed(5), true).unwrap();
    let all = collect(&mut full, None);
    let mut sorted = all.clone();
    sorted.sort();
    assert_eq!(
        sorted,
        (0..103).collect::<Vec<_>>(),
        "every row exactly once"
    );
    assert_ne!(all, (0..103).collect::<Vec<_>>(), "order is shuffled");

    let mut part = BatchReader::open(&p, &s, &pre, 16, Rng::seed(5), true).unwrap();
    let head = collect(&mut part, Some(4));
    let pos = part.position();
    assert_eq!(pos.rows_emitted, 40);
    let mut resumed = BatchReader::resume(&p, &s, &pre, 16, &pos).unwrap();
    let tail = collect(&mut resumed, None);
    assert_eq!(
        [head, tail].concat(),
        all,
        "resume reproduces the exact order"
    );

    let mut ordered = BatchReader::open(&p, &s, &pre, 1, Rng::seed(9), true).unwrap();
    assert_eq!(collect(&mut ordered, None), (0..103).collect::<Vec<_>>());
}

#[test]
fn split_is_deterministic() {
    let d = tempfile::tempdir().unwrap();
    let mut csv = String::from("a,b,y\n");
    for i in 0..1000 {
        csv.push_str(&format!("{i},{i},no\n"));
    }
    let p = write(&d, "all.csv", &csv);
    let c1 = split_csv(&p, &d.path().join("s1"), 0.2, 0.2, 3).unwrap();
    let c2 = split_csv(&p, &d.path().join("s2"), 0.2, 0.2, 3).unwrap();
    assert_eq!(c1, c2);
    assert_eq!(c1.train + c1.valid + c1.test, 1000);
    assert!((150..250).contains(&c1.valid) && (150..250).contains(&c1.test));
    let a = std::fs::read(d.path().join("s1/test.csv")).unwrap();
    let b = std::fs::read(d.path().join("s2/test.csv")).unwrap();
    assert_eq!(a, b);
    assert!(split_csv(&p, &d.path().join("s3"), 0.6, 0.5, 3).is_err());
}

#[test]
fn split_refuses_to_overwrite_its_input() {
    let d = tempfile::tempdir().unwrap();
    let content = "a,b,y\n1,2,no\n3,4,yes\n";
    let p = write(&d, "train.csv", content);
    let e = split_csv(&p, d.path(), 0.2, 0.2, 1)
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("overwritten"), "{e}");
    assert_eq!(
        std::fs::read_to_string(&p).unwrap(),
        content,
        "input untouched"
    );
}

#[test]
fn oversized_or_unterminated_records_fail_while_reading() {
    let d = tempfile::tempdir().unwrap();
    let s = schema(MissingPolicy::Reject, ExtraColumns::Ignore);
    // An unterminated quote would otherwise make the parser buffer the whole file.
    let mut big = String::from("a,b,y\n1,\"");
    big.push_str(&"x".repeat(3 << 20));
    let p = write(&d, "big.csv", &big);
    let e = Preprocessor::fit(&p, &s, Normalize::Standardize)
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("exceeds"), "{e}");
    let e = split_csv(&p, &d.path().join("out"), 0.2, 0.2, 1)
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("exceeds"), "{e}");
    assert!(
        !d.path().join("out/train.csv").exists(),
        "no partial outputs left"
    );
}

/// Review regression: an input named like the split's fixed temporary file was
/// deleted while `split` reported success.
#[test]
fn split_input_named_like_a_temporary_file_survives() {
    let d = tempfile::tempdir().unwrap();
    let content = "a,b,y\n1,2,no\n3,4,yes\n5,6,no\n";
    let p = write(&d, ".train.csv.partial", content);
    let r = split_csv(&p, d.path(), 0.2, 0.2, 1);
    assert_eq!(
        std::fs::read_to_string(&p).ok().as_deref(),
        Some(content),
        "input lost; result {r:?}"
    );
}

#[test]
fn split_replaces_existing_outputs_and_leaves_no_temporary_files() {
    let d = tempfile::tempdir().unwrap();
    let input = write(&d, "all.csv", "a,b,y\n1,2,no\n3,4,yes\n");
    let out = d.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join("train.csv"), "stale").unwrap();
    std::fs::write(out.join(".train.csv.partial"), "unrelated").unwrap();
    split_csv(&input, &out, 0.0, 0.0, 1).unwrap();
    assert!(
        std::fs::read_to_string(out.join("train.csv"))
            .unwrap()
            .starts_with("a,b,y")
    );
    assert_eq!(
        std::fs::read_to_string(out.join(".train.csv.partial")).unwrap(),
        "unrelated"
    );
    let tmp: Vec<_> = std::fs::read_dir(&out)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .filter(|n| n.ends_with(".tmp"))
        .collect();
    assert!(tmp.is_empty(), "{tmp:?}");
}

#[test]
fn preprocessor_apply_checks_lengths_in_every_build() {
    let s = schema(MissingPolicy::Reject, ExtraColumns::Ignore);
    let pre = Preprocessor {
        normalize: Normalize::None,
        missing: MissingPolicy::Reject,
        features: s
            .features
            .iter()
            .map(|n| FeatureStats {
                name: n.clone(),
                mean: 0.0,
                std: 1.0,
                count: 1,
                missing: 0,
                constant: false,
            })
            .collect(),
        rows_fitted: 1,
    };
    let mut out2 = [0.0f32; 2];
    let mut out1 = [0.0f32; 1];
    let mut out3 = [0.0f32; 3];
    assert!(matches!(
        pre.apply(&[Some(1.0)], &mut out2),
        Err(ApplyError::Length { .. })
    ));
    assert!(matches!(
        pre.apply(&[Some(1.0); 3], &mut out2),
        Err(ApplyError::Length { .. })
    ));
    assert!(matches!(
        pre.apply(&[Some(1.0); 2], &mut out1),
        Err(ApplyError::Length { .. })
    ));
    assert!(matches!(
        pre.apply(&[Some(1.0); 2], &mut out3),
        Err(ApplyError::Length { .. })
    ));
    pre.apply(&[Some(1.0), Some(2.0)], &mut out2).unwrap();
    assert_eq!(out2, [1.0, 2.0]);
    assert_eq!(
        pre.apply(&[None, Some(2.0)], &mut out2),
        Err(ApplyError::Feature(0))
    );
}
