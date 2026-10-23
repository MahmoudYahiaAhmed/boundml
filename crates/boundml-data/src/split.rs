use crate::csvsrc::MAX_RECORD_BYTES;
use crate::error::{DataError, Result};
use crate::limit::{PARSER_BUFFER, RecordLimit, reset};
use crate::reserve::{ReservedFile, same_file};
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
/// Outputs are written to exclusively reserved, uniquely named temporary files and
/// renamed only after the whole input was read; an output path that is the input
/// file itself is refused. Existing `train.csv`/`valid.csv`/`test.csv` are replaced.
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
    std::fs::canonicalize(input).map_err(io(input))?;
    if OUTPUTS.iter().any(|n| same_file(&out_dir.join(n), input)) {
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
    // Uniquely named, exclusively created temporary files in `out_dir` (same
    // filesystem as the final names). Dropping them on any error removes only the
    // files this call created; the input and unrelated files are never touched.
    let dests: Vec<PathBuf> = OUTPUTS.iter().map(|n| out_dir.join(n)).collect();
    let mut reserved = Vec::with_capacity(3);
    for d in &dests {
        reserved.push(ReservedFile::next_to(d).map_err(io(d))?);
    }
    let mut writers = Vec::with_capacity(3);
    for r in reserved.iter_mut() {
        let ps = r.path().display().to_string();
        let mut w = csv::Writer::from_writer(r.take_file());
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
    for (w, ps) in writers {
        let f = w.into_inner().map_err(|e| DataError::Io {
            path: ps.clone(),
            source: e.into_error(),
        })?;
        f.sync_all()
            .map_err(|source| DataError::Io { path: ps, source })?;
    }
    // Commit only after the whole input was read successfully.
    for (r, d) in reserved.into_iter().zip(&dests) {
        r.commit(d).map_err(io(d))?;
    }
    Ok(SplitCounts {
        train: counts[0],
        valid: counts[1],
        test: counts[2],
    })
}
