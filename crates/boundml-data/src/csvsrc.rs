use crate::error::{DataError, Result};
use crate::limit::{PARSER_BUFFER, RecordLimit, reset};
use crate::schema::{ExtraColumns, LabelKind, Schema};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

/// Maximum number of header columns accepted.
pub const MAX_COLUMNS: usize = 100_000;
/// Maximum bytes in a single CSV record; longer records are rejected.
pub const MAX_RECORD_BYTES: usize = 1 << 20;

/// One parsed row, reused between reads to avoid per-row allocation.
#[derive(Debug, Clone, Default)]
pub struct RawRow {
    /// Feature values in schema order; `None` = missing (empty cell).
    pub features: Vec<Option<f64>>,
    /// Regression value or class index; `None` when the file has no label column.
    pub label: Option<f32>,
    /// 1-based line number in the file.
    pub line: u64,
}

/// Streams rows of a CSV file through a [`Schema`]. Memory is bounded by one record
/// of at most [`MAX_RECORD_BYTES`] plus the parser's 64 KiB buffer; longer records
/// (including an unterminated quote) fail while reading.
pub struct CsvSource {
    path: String,
    reader: csv::Reader<RecordLimit<File>>,
    consumed: Arc<AtomicU64>,
    record: csv::StringRecord,
    feat_idx: Vec<usize>,
    label_idx: Option<usize>,
    schema: Schema,
    /// Added to the csv crate's record line: 1 for CRLF files, whose records it
    /// reports one line too low (the `\n` of the previous `\r\n` is counted
    /// only after the next record has started).
    line_offset: u64,
}

/// Whether the file's first line ends with `\r\n`. Reads at most one record's worth.
fn first_line_is_crlf(path: &Path) -> bool {
    use std::io::{BufRead, BufReader, Read};
    let Ok(file) = File::open(path) else {
        return false;
    };
    let mut line = Vec::new();
    let limit = MAX_RECORD_BYTES as u64 + 2;
    match BufReader::new(file.take(limit)).read_until(b'\n', &mut line) {
        Ok(_) => line.ends_with(b"\r\n"),
        Err(_) => false,
    }
}

impl CsvSource {
    /// Open `path`, validate its header against `schema`. If `require_label`, the
    /// label column must exist.
    pub fn open(path: &Path, schema: &Schema, require_label: bool) -> Result<Self> {
        schema.validate()?;
        let p = path.display().to_string();
        let file = File::open(path).map_err(|source| DataError::Io {
            path: p.clone(),
            source,
        })?;
        let (limited, consumed) = RecordLimit::new(file, MAX_RECORD_BYTES);
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(false)
            .buffer_capacity(PARSER_BUFFER)
            .from_reader(limited);
        let headers = reader
            .headers()
            .map_err(|e| DataError::Csv {
                path: p.clone(),
                message: e.to_string(),
            })?
            .clone();
        let hdr = |m: String| DataError::Header {
            path: p.clone(),
            message: m,
        };
        if headers.len() > MAX_COLUMNS {
            return Err(hdr(format!(
                "{} columns exceeds limit {MAX_COLUMNS}",
                headers.len()
            )));
        }
        let names: Vec<&str> = headers.iter().map(str::trim).collect();
        let mut seen = std::collections::HashSet::new();
        for n in &names {
            if !seen.insert(*n) {
                return Err(hdr(format!("duplicate column '{n}'")));
            }
        }
        let find = |name: &str| names.iter().position(|n| *n == name);
        let mut feat_idx = Vec::with_capacity(schema.features.len());
        let mut missing = Vec::new();
        for f in &schema.features {
            match find(f) {
                Some(i) => feat_idx.push(i),
                None => missing.push(f.as_str()),
            }
        }
        if !missing.is_empty() {
            return Err(hdr(format!(
                "missing feature column(s): {}",
                missing.join(", ")
            )));
        }
        let label_idx = find(&schema.label);
        if require_label && label_idx.is_none() {
            return Err(hdr(format!("missing label column '{}'", schema.label)));
        }
        if schema.extra_columns == ExtraColumns::Reject {
            let extra: Vec<&str> = names
                .iter()
                .enumerate()
                .filter(|(i, _)| !feat_idx.contains(i) && Some(*i) != label_idx)
                .map(|(_, n)| *n)
                .collect();
            if !extra.is_empty() {
                return Err(hdr(format!(
                    "unexpected column(s) {} and extra_columns policy is 'reject'",
                    extra.join(", ")
                )));
            }
        }
        reset(&consumed);
        Ok(CsvSource {
            path: p,
            reader,
            consumed,
            record: csv::StringRecord::new(),
            feat_idx,
            label_idx,
            schema: schema.clone(),
            line_offset: u64::from(first_line_is_crlf(path)),
        })
    }

    pub fn has_label(&self) -> bool {
        self.label_idx.is_some()
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    /// Read the next row into `row`. Returns `false` at end of file.
    pub fn read_row(&mut self, row: &mut RawRow) -> Result<bool> {
        let more = self
            .reader
            .read_record(&mut self.record)
            .map_err(|e| DataError::Csv {
                path: self.path.clone(),
                message: e.to_string(),
            })?;
        if !more {
            return Ok(false);
        }
        reset(&self.consumed);
        let line = self.record.position().map_or(0, |p| p.line()) + self.line_offset;
        if self.record.as_byte_record().as_slice().len() > MAX_RECORD_BYTES {
            return Err(DataError::Csv {
                path: self.path.clone(),
                message: format!("line {line} exceeds {MAX_RECORD_BYTES} bytes"),
            });
        }
        row.line = line;
        row.features.clear();
        for (k, &i) in self.feat_idx.iter().enumerate() {
            let cell = self.record.get(i).unwrap_or("").trim();
            let v = parse_feature(cell).map_err(|message| DataError::Value {
                path: self.path.clone(),
                line,
                column: self.schema.features[k].clone(),
                value: cell.chars().take(64).collect(),
                message,
            })?;
            row.features.push(v);
        }
        row.label = match self.label_idx {
            None => None,
            Some(i) => {
                let cell = self.record.get(i).unwrap_or("").trim();
                Some(
                    parse_label(cell, &self.schema.label_kind).map_err(|message| {
                        DataError::Value {
                            path: self.path.clone(),
                            line,
                            column: self.schema.label.clone(),
                            value: cell.chars().take(64).collect(),
                            message,
                        }
                    })?,
                )
            }
        };
        Ok(true)
    }
}

/// Parse a feature cell: empty = missing; otherwise a finite number.
pub fn parse_feature(cell: &str) -> std::result::Result<Option<f64>, String> {
    if cell.is_empty() {
        return Ok(None);
    }
    match cell.parse::<f64>() {
        Ok(v) if v.is_finite() => Ok(Some(v)),
        Ok(_) => Err("non-finite numbers (NaN/inf) are not accepted; leave the cell empty to mark it missing".into()),
        Err(_) => Err("not a number".into()),
    }
}

/// Parse a label cell into a regression value or class index.
pub fn parse_label(cell: &str, kind: &LabelKind) -> std::result::Result<f32, String> {
    if cell.is_empty() {
        return Err("label is missing".into());
    }
    match kind {
        LabelKind::Regression => match cell.parse::<f64>() {
            Ok(v) if v.is_finite() && v.abs() < f32::MAX as f64 => Ok(v as f32),
            _ => Err("label must be a finite number".into()),
        },
        LabelKind::Classes { classes } => classes
            .iter()
            .position(|c| c == cell)
            .map(|i| i as f32)
            .ok_or_else(|| format!("unknown class; expected one of {}", classes.join(", "))),
    }
}
