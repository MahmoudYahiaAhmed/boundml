//! Bounded file reading, SHA-256, safetensors encode/decode and atomic directory writes.

use crate::error::{ArtifactError, Result};
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// SHA-256 of a file, streamed (memory bounded by an 64 KiB buffer).
pub fn sha256_file(path: &Path) -> Result<(String, u64)> {
    let mut f = std::fs::File::open(path).map_err(|e| ArtifactError::io(path, e))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let n = f.read(&mut buf).map_err(|e| ArtifactError::io(path, e))?;
        if n == 0 {
            break;
        }
        total += n as u64;
        h.update(&buf[..n]);
    }
    Ok((hex::encode(h.finalize()), total))
}

/// Read a regular (non-symlink) file whose size must not exceed `max` bytes. The size
/// is checked from metadata *before* allocating, and again while reading.
pub fn read_limited(path: &Path, max: u64) -> Result<Vec<u8>> {
    let meta = std::fs::symlink_metadata(path).map_err(|e| ArtifactError::io(path, e))?;
    if meta.file_type().is_symlink() || !meta.is_file() {
        return Err(ArtifactError::Format(format!(
            "{} must be a regular file (symlinks are rejected)",
            path.display()
        )));
    }
    if meta.len() > max {
        return Err(ArtifactError::TooLarge {
            what: path.display().to_string(),
            bytes: meta.len(),
            limit: max,
        });
    }
    let f = std::fs::File::open(path).map_err(|e| ArtifactError::io(path, e))?;
    let mut out = Vec::with_capacity(meta.len() as usize);
    f.take(max + 1)
        .read_to_end(&mut out)
        .map_err(|e| ArtifactError::io(path, e))?;
    if out.len() as u64 > max {
        return Err(ArtifactError::TooLarge {
            what: path.display().to_string(),
            bytes: out.len() as u64,
            limit: max,
        });
    }
    Ok(out)
}

/// Encode named f32 matrices as safetensors (row-major, little-endian).
pub fn encode_tensors(tensors: &[(String, [usize; 2], &[f32])]) -> Result<Vec<u8>> {
    let bytes: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_, _, d)| d.iter().flat_map(|v| v.to_le_bytes()).collect())
        .collect();
    let mut views = Vec::with_capacity(tensors.len());
    for ((name, shape, _), b) in tensors.iter().zip(&bytes) {
        let v = TensorView::new(Dtype::F32, shape.to_vec(), b)
            .map_err(|e| ArtifactError::Format(format!("tensor '{name}': {e}")))?;
        views.push((name.clone(), v));
    }
    safetensors::serialize(views, None).map_err(|e| ArtifactError::Format(e.to_string()))
}

/// A decoded rank-2 f32 tensor: `(name, [rows, cols], values)`.
pub type NamedTensor = (String, [usize; 2], Vec<f32>);

/// Decode a safetensors buffer into `(name, shape, values)` with strict checks:
/// f32 only, rank 2 (rank 1 and 0 are rejected), finite values, at most `max_tensors`.
pub fn decode_tensors(bytes: &[u8], max_tensors: usize) -> Result<Vec<NamedTensor>> {
    let st = SafeTensors::deserialize(bytes)
        .map_err(|e| ArtifactError::Format(format!("invalid safetensors: {e}")))?;
    let tensors = st.tensors();
    if tensors.len() > max_tensors {
        return Err(ArtifactError::Format(format!(
            "{} tensors exceeds limit {max_tensors}",
            tensors.len()
        )));
    }
    let mut out = Vec::with_capacity(tensors.len());
    for (name, view) in tensors {
        if view.dtype() != Dtype::F32 {
            return Err(ArtifactError::Format(format!(
                "tensor '{name}' has dtype {:?}; only F32 is supported",
                view.dtype()
            )));
        }
        let shape: [usize; 2] = view.shape().try_into().map_err(|_| {
            ArtifactError::Format(format!(
                "tensor '{name}' must be rank 2, has shape {:?}",
                view.shape()
            ))
        })?;
        let values: Vec<f32> = view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(ArtifactError::Format(format!(
                "tensor '{name}' contains non-finite values"
            )));
        }
        out.push((name, shape, values));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Write `files` into `dest` atomically with respect to readers of `dest`:
/// files are written and fsynced in a sibling temp directory, an existing `dest` is
/// renamed to `<dest>.prev` (replacing any older `.prev`), then the temp directory is
/// renamed to `dest`. If the process dies between the two renames, `<dest>.prev`
/// still holds the previous complete version.
pub fn write_dir_atomic(dest: &Path, files: &[(&str, &[u8])]) -> Result<()> {
    let parent = dest
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(|e| ArtifactError::io(parent, e))?;
    let name = dest
        .file_name()
        .ok_or_else(|| ArtifactError::Format(format!("invalid destination {}", dest.display())))?
        .to_string_lossy()
        .to_string();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = parent.join(format!(".{name}.tmp-{}-{nanos}", std::process::id()));
    std::fs::create_dir(&tmp).map_err(|e| ArtifactError::io(&tmp, e))?;
    let result = (|| {
        for (fname, bytes) in files {
            if fname.contains(['/', '\\']) || fname.starts_with('.') {
                return Err(ArtifactError::Format(format!("invalid file name {fname}")));
            }
            let p = tmp.join(fname);
            let mut f = std::fs::File::create(&p).map_err(|e| ArtifactError::io(&p, e))?;
            f.write_all(bytes).map_err(|e| ArtifactError::io(&p, e))?;
            f.sync_all().map_err(|e| ArtifactError::io(&p, e))?;
        }
        let prev = prev_path(dest);
        if dest.exists() {
            if prev.exists() {
                std::fs::remove_dir_all(&prev).map_err(|e| ArtifactError::io(&prev, e))?;
            }
            std::fs::rename(dest, &prev).map_err(|e| ArtifactError::io(dest, e))?;
        }
        std::fs::rename(&tmp, dest).map_err(|e| ArtifactError::io(dest, e))
    })();
    if result.is_err() && tmp.exists() {
        let _ = std::fs::remove_dir_all(&tmp);
    }
    result
}

/// `<dest>.prev`: the previous version kept by [`write_dir_atomic`].
pub fn prev_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(".prev");
    PathBuf::from(s)
}

/// Ensure `dir` is a real directory (not a symlink).
pub fn check_dir(dir: &Path) -> Result<()> {
    let meta = std::fs::symlink_metadata(dir).map_err(|e| ArtifactError::io(dir, e))?;
    if meta.file_type().is_symlink() || !meta.is_dir() {
        return Err(ArtifactError::Format(format!(
            "{} must be a directory (symlinks are rejected)",
            dir.display()
        )));
    }
    Ok(())
}

pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
