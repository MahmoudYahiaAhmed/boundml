use crate::csvsrc::MAX_RECORD_BYTES;
use crate::error::{DataError, Result};
use crate::limit::{PARSER_BUFFER, RecordLimit, reset};
use boundml_core::Rng;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitCounts {
    pub train: u64,
    pub valid: u64,
    pub test: u64,
}

const OUTPUTS: [&str; 3] = ["train.csv", "valid.csv", "test.csv"];

/// Deterministically split a CSV (with header) into `train.csv`, `valid.csv` and
/// `test.csv` in `out_dir`. Each row is assigned independently by a seeded draw, so
/// proportions are approximate. Streams; memory is bounded by one record.
///
/// Outputs are written to temporary files and renamed only after the whole input
/// was read, and an output path that is the input file itself is refused.
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
    let io = |path: &Path| {
        let path = path.display().to_string();
        move |source| DataError::Io { path, source }
    };
    std::fs::create_dir_all(out_dir).map_err(io(out_dir))?;
    let canon_in = std::fs::canonicalize(input).map_err(io(input))?;
    let canon_out = std::fs::canonicalize(out_dir).map_err(io(out_dir))?;
    if OUTPUTS.iter().any(|n| canon_out.join(n) == canon_in) {
        return Err(DataError::Schema(format!(
            "{ip} would be overwritten by the split outputs; choose a different --out directory"
        )));
    }
    let csv_err = |path: &str, e: csv::Error| DataError::Csv {
        path: path.to_string(),
        message: e.to_string(),
    };
    let file = std::fs::File::open(input).map_err(io(input))?;
    let (limited, consumed) = RecordLimit::new(file, MAX_RECORD_BYTES);
    let mut reader = csv::ReaderBuilder::new()
        .flexible(false)
        .buffer_capacity(PARSER_BUFFER)
        .from_reader(limited);
    let headers = reader.byte_headers().map_err(|e| csv_err(&ip, e))?.clone();
    let tmp_paths: Vec<PathBuf> = OUTPUTS
        .iter()
        .map(|n| out_dir.join(format!(".{n}.partial")))
        .collect();
    let result = (|| {
        let mut writers = Vec::new();
        for p in &tmp_paths {
            let ps = p.display().to_string();
            let mut w = csv::Writer::from_path(p).map_err(|e| csv_err(&ps, e))?;
            w.write_byte_record(&headers).map_err(|e| csv_err(&ps, e))?;
            writers.push((w, ps));
        }
        let mut rng = Rng::seed(seed);
        let mut counts = [0u64; 3];
        let mut rec = csv::ByteRecord::new();
        reset(&consumed);
        while reader
            .read_byte_record(&mut rec)
            .map_err(|e| csv_err(&ip, e))?
        {
            reset(&consumed);
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
        Ok(counts)
    })();
    let counts = match result {
        Ok(c) => c,
        Err(e) => {
            for p in &tmp_paths {
                let _ = std::fs::remove_file(p);
            }
            return Err(e);
        }
    };
    for (tmp, name) in tmp_paths.iter().zip(OUTPUTS) {
        let dest = out_dir.join(name);
        std::fs::rename(tmp, &dest).map_err(io(&dest))?;
    }
    Ok(SplitCounts {
        train: counts[0],
        valid: counts[1],
        test: counts[2],
    })
}
