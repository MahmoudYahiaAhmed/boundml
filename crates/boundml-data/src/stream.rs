//! Deterministic minibatch streaming with a bounded shuffle buffer.
//!
//! Strategy: rows are read in file order into a buffer of `shuffle_buffer` rows.
//! Each emitted row is drawn uniformly from the buffer and its slot is refilled
//! with the next file row. Memory is `O(shuffle_buffer * features)` regardless of
//! file size. This is a *local* shuffle: rows far apart in the file are not mixed
//! unless the buffer is at least as large as the file. With a buffer of 1 rows are
//! emitted in file order.
//!
//! Resume: the RNG state at the start of the epoch plus the number of rows already
//! emitted fully determine the rest of the epoch. [`BatchReader::resume`] replays the
//! draws (reading but discarding rows) to reach the exact same position.

use crate::csvsrc::{CsvSource, RawRow};
use crate::error::{DataError, Result};
use crate::preprocess::Preprocessor;
use crate::schema::Schema;
use boundml_core::Rng;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const MAX_SHUFFLE_BUFFER: usize = 1 << 24;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReaderPosition {
    /// RNG state when the epoch started.
    pub rng_state: [u64; 4],
    pub rows_emitted: u64,
}

pub struct BatchReader {
    src: CsvSource,
    pre: Preprocessor,
    rng: Rng,
    start_state: [u64; 4],
    cap: usize,
    nf: usize,
    buf_x: Vec<f32>,
    buf_y: Vec<f32>,
    buf_line: Vec<u64>,
    len: usize,
    filled: bool,
    exhausted: bool,
    row: RawRow,
    emitted: u64,
}

impl BatchReader {
    /// `shuffle_buffer = 1` gives file order.
    pub fn open(
        path: &Path,
        schema: &Schema,
        pre: &Preprocessor,
        shuffle_buffer: usize,
        rng: Rng,
        require_label: bool,
    ) -> Result<Self> {
        pre.validate(schema)?;
        if shuffle_buffer == 0 || shuffle_buffer > MAX_SHUFFLE_BUFFER {
            return Err(DataError::Schema(format!(
                "shuffle_buffer must be 1..={MAX_SHUFFLE_BUFFER}"
            )));
        }
        let nf = schema.features.len();
        let src = CsvSource::open(path, schema, require_label)?;
        let elems = shuffle_buffer
            .checked_mul(nf)
            .ok_or_else(|| DataError::Schema("shuffle buffer size overflows".into()))?;
        Ok(BatchReader {
            src,
            pre: pre.clone(),
            start_state: rng.state(),
            rng,
            cap: shuffle_buffer,
            nf,
            buf_x: vec![0.0; elems],
            buf_y: vec![0.0; shuffle_buffer],
            buf_line: vec![0; shuffle_buffer],
            len: 0,
            filled: false,
            exhausted: false,
            row: RawRow::default(),
            emitted: 0,
        })
    }

    /// Reopen at a saved position (same file, schema, preprocessing, buffer size).
    pub fn resume(
        path: &Path,
        schema: &Schema,
        pre: &Preprocessor,
        shuffle_buffer: usize,
        pos: &ReaderPosition,
    ) -> Result<Self> {
        let rng = Rng::from_state(pos.rng_state)
            .ok_or_else(|| DataError::Resume("all-zero RNG state".into()))?;
        let mut r = Self::open(path, schema, pre, shuffle_buffer, rng, true)?;
        for i in 0..pos.rows_emitted {
            if r.emit_one(None)?.is_none() {
                return Err(DataError::Resume(format!(
                    "file has fewer rows ({i}) than the checkpoint already consumed ({}); was it modified?",
                    pos.rows_emitted
                )));
            }
        }
        Ok(r)
    }

    pub fn position(&self) -> ReaderPosition {
        ReaderPosition {
            rng_state: self.start_state,
            rows_emitted: self.emitted,
        }
    }

    /// RNG state after the epoch's draws so far (use to seed the next epoch).
    pub fn rng(&self) -> &Rng {
        &self.rng
    }

    pub fn n_features(&self) -> usize {
        self.nf
    }

    /// Fill up to `max` rows. `x` must hold `max * features`, `y` and `lines` `max`.
    /// Returns the number of rows written (0 at end of data).
    pub fn next_batch(
        &mut self,
        max: usize,
        x: &mut [f32],
        y: &mut [f32],
        lines: &mut [u64],
    ) -> Result<usize> {
        let mut n = 0;
        while n < max {
            let dst = (
                &mut x[n * self.nf..(n + 1) * self.nf],
                &mut y[n],
                &mut lines[n],
            );
            match self.emit_one(Some(dst))? {
                Some(()) => n += 1,
                None => break,
            }
        }
        Ok(n)
    }

    /// Read one file row into buffer slot `slot`. Returns false at EOF.
    fn load(&mut self, slot: usize) -> Result<bool> {
        if !self.src.read_row(&mut self.row)? {
            self.exhausted = true;
            return Ok(false);
        }
        let dst = &mut self.buf_x[slot * self.nf..(slot + 1) * self.nf];
        if let Err(e) = self.pre.apply(&self.row.features, dst) {
            let i = match e {
                crate::preprocess::ApplyError::Feature(i) => i,
                // The reader sizes rows from the same schema; this is an internal bug.
                e @ crate::preprocess::ApplyError::Length { .. } => {
                    return Err(DataError::Preprocess(e.to_string()));
                }
            };
            let column = self.pre.features[i].name.clone();
            return Err(match self.row.features[i] {
                None => DataError::Missing {
                    path: self.src.path().to_string(),
                    line: self.row.line,
                    column,
                },
                Some(v) => DataError::Value {
                    path: self.src.path().to_string(),
                    line: self.row.line,
                    column,
                    value: v.to_string(),
                    message: "value overflows f32 after preprocessing".into(),
                },
            });
        }
        self.buf_y[slot] = self.row.label.unwrap_or(0.0);
        self.buf_line[slot] = self.row.line;
        Ok(true)
    }

    fn emit_one(&mut self, dst: Option<(&mut [f32], &mut f32, &mut u64)>) -> Result<Option<()>> {
        if !self.filled {
            self.filled = true;
            while self.len < self.cap && self.load(self.len)? {
                self.len += 1;
            }
        }
        if self.len == 0 {
            return Ok(None);
        }
        let j = self.rng.below(self.len as u64) as usize;
        if let Some((x, y, line)) = dst {
            x.copy_from_slice(&self.buf_x[j * self.nf..(j + 1) * self.nf]);
            *y = self.buf_y[j];
            *line = self.buf_line[j];
        }
        self.emitted += 1;
        if self.exhausted || !self.load(j)? {
            let last = self.len - 1;
            if j != last {
                self.buf_x
                    .copy_within(last * self.nf..(last + 1) * self.nf, j * self.nf);
                self.buf_y[j] = self.buf_y[last];
                self.buf_line[j] = self.buf_line[last];
            }
            self.len -= 1;
        }
        Ok(Some(()))
    }
}
