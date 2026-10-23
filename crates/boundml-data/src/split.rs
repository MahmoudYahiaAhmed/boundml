use crate::error::{DataError, Result};
use boundml_core::Rng;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitCounts {
    pub train: u64,
    pub valid: u64,
    pub test: u64,
}

/// Deterministically split a CSV (with header) into `train.csv`, `valid.csv` and
/// `test.csv` in `out_dir`. Each row is assigned independently by a seeded draw,
/// so proportions are approximate. Streams; memory is bounded by one record.
pub fn split_csv(
    input: &Path,
    out_dir: &Path,
    valid: f64,
    test: f64,
    seed: u64,
) -> Result<SplitCounts> {
    if !(0.0..1.0).contains(&valid) || !(0.0..1.0).contains(&test) || valid + test >= 1.0 {
        return Err(DataError::Schema(
            "valid and test fractions must be >= 0 and sum to < 1".into(),
        ));
    }
    let ip = input.display().to_string();
    let csv_err = |path: &str, e: csv::Error| DataError::Csv {
        path: path.to_string(),
        message: e.to_string(),
    };
    let mut reader = csv::ReaderBuilder::new()
        .flexible(false)
        .from_path(input)
        .map_err(|e| csv_err(&ip, e))?;
    let headers = reader.byte_headers().map_err(|e| csv_err(&ip, e))?.clone();
    std::fs::create_dir_all(out_dir).map_err(|source| DataError::Io {
        path: out_dir.display().to_string(),
        source,
    })?;
    let mut writers = Vec::new();
    for name in ["train.csv", "valid.csv", "test.csv"] {
        let p = out_dir.join(name);
        let ps = p.display().to_string();
        let mut w = csv::Writer::from_path(&p).map_err(|e| csv_err(&ps, e))?;
        w.write_byte_record(&headers).map_err(|e| csv_err(&ps, e))?;
        writers.push((w, ps));
    }
    let mut rng = Rng::seed(seed);
    let mut counts = [0u64; 3];
    let mut rec = csv::ByteRecord::new();
    while reader
        .read_byte_record(&mut rec)
        .map_err(|e| csv_err(&ip, e))?
    {
        let u = rng.next_f32() as f64;
        let k = if u < valid {
            1
        } else if u < valid + test {
            2
        } else {
            0
        };
        let (w, ps) = &mut writers[k];
        w.write_byte_record(&rec).map_err(|e| csv_err(ps, e))?;
        counts[k] += 1;
    }
    for (mut w, ps) in writers {
        w.flush()
            .map_err(|source| DataError::Io { path: ps, source })?;
    }
    Ok(SplitCounts {
        train: counts[0],
        valid: counts[1],
        test: counts[2],
    })
}
