//! A reader that bounds how many bytes the CSV parser may pull in for one record.
//!
//! The `csv` crate buffers a whole record before returning it, so checking the
//! record length afterwards would not bound memory. This wrapper counts bytes read
//! since the last completed record (the caller resets the counter after each one)
//! and fails the read once the count passes the limit plus one parser buffer.

use std::io::Read;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// The parser's internal buffer size; bytes buffered ahead are at most this much.
pub(crate) const PARSER_BUFFER: usize = 64 * 1024;

pub(crate) struct RecordLimit<R> {
    inner: R,
    since_reset: Arc<AtomicU64>,
    limit: u64,
}

impl<R: Read> RecordLimit<R> {
    pub(crate) fn new(inner: R, record_limit: usize) -> (Self, Arc<AtomicU64>) {
        let counter = Arc::new(AtomicU64::new(0));
        let me = RecordLimit {
            inner,
            since_reset: Arc::clone(&counter),
            limit: (record_limit + PARSER_BUFFER) as u64,
        };
        (me, counter)
    }
}

impl<R: Read> Read for RecordLimit<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.since_reset.load(Ordering::Relaxed) > self.limit {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "a CSV record (or an unterminated quoted field) exceeds {} bytes",
                    self.limit - PARSER_BUFFER as u64
                ),
            ));
        }
        let n = self.inner.read(buf)?;
        self.since_reset.fetch_add(n as u64, Ordering::Relaxed);
        Ok(n)
    }
}

/// Reset after each completed record.
pub(crate) fn reset(counter: &AtomicU64) {
    counter.store(0, Ordering::Relaxed);
}
