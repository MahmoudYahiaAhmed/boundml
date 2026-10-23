//! Exclusively reserved temporary files for write-then-rename commits.
//!
//! A [`ReservedFile`] is created with `create_new` (fails if the name exists), with
//! a unique name in the destination's own directory, so the final rename stays on
//! the same filesystem. It never touches a pre-existing file: if anything goes
//! wrong before [`ReservedFile::commit`], dropping it removes only the file this
//! invocation created.

use std::fs::{File, OpenOptions};
use std::hash::{BuildHasher, Hasher};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub struct ReservedFile {
    path: PathBuf,
    file: Option<File>,
    committed: bool,
}

impl ReservedFile {
    /// Reserve a new, uniquely named file next to `dest` (same directory).
    pub fn next_to(dest: &Path) -> io::Result<ReservedFile> {
        let dir = match dest.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => PathBuf::from("."),
        };
        let stem = dest
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "output".into());
        for _ in 0..64 {
            let mut h = std::collections::hash_map::RandomState::new().build_hasher();
            h.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
            h.write_u32(std::process::id());
            let name = format!(".{stem}.{:x}-{:016x}.tmp", std::process::id(), h.finish());
            let path = dir.join(name);
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => {
                    return Ok(ReservedFile {
                        path,
                        file: Some(file),
                        committed: false,
                    });
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not reserve a unique temporary file",
        ))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The open file handle (for writing); `None` after [`ReservedFile::take_file`].
    pub fn take_file(&mut self) -> File {
        self.file.take().expect("file taken once")
    }

    /// Flush to disk and atomically move into place, replacing `dest` if it exists.
    pub fn commit(mut self, dest: &Path) -> io::Result<()> {
        if let Some(f) = self.file.take() {
            f.sync_all()?;
        } else {
            // Windows needs write access to flush a file's buffers.
            OpenOptions::new()
                .write(true)
                .open(&self.path)?
                .sync_all()?;
        }
        std::fs::rename(&self.path, dest)?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for ReservedFile {
    fn drop(&mut self) {
        if !self.committed {
            drop(self.file.take());
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// True if `a` and `b` name the same existing file (after resolving links and `..`).
pub fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservations_are_unique_exclusive_and_cleaned_up() {
        let d = tempfile::tempdir().unwrap();
        let dest = d.path().join("out.csv");
        let a = ReservedFile::next_to(&dest).unwrap();
        let b = ReservedFile::next_to(&dest).unwrap();
        assert_ne!(a.path(), b.path());
        let (pa, pb) = (a.path().to_path_buf(), b.path().to_path_buf());
        drop(a);
        assert!(!pa.exists(), "uncommitted reservation removed");
        let mut b = b;
        use std::io::Write;
        b.take_file().write_all(b"x").unwrap();
        b.commit(&dest).unwrap();
        assert!(!pb.exists() && std::fs::read(&dest).unwrap() == b"x");
    }
}
