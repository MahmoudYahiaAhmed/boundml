use boundml_core::CoreError;
use boundml_data::DataError;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ArtifactError {
    #[error("BML-A001 I/O error on {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("BML-A002 malformed artifact: {0}")]
    Format(String),
    #[error("BML-A003 unsupported artifact version: {0}")]
    Version(String),
    #[error(
        "BML-A004 integrity check failed for {file}: expected sha256 {expected}, found {found}"
    )]
    Integrity {
        file: String,
        expected: String,
        found: String,
    },
    #[error("BML-A005 {what} is {bytes} bytes, exceeding the limit of {limit} bytes")]
    TooLarge {
        what: String,
        bytes: u64,
        limit: u64,
    },
    #[error("BML-A006 artifact is inconsistent: {0}")]
    Inconsistent(String),
    #[error("BML-A007 self-test failed: {0}")]
    SelfTest(String),
    #[error("BML-A008 invalid input row {row}: {message}")]
    Input { row: usize, message: String },
    #[error("BML-A009 invalid option: {0}")]
    InvalidOption(String),
    #[error(transparent)]
    Core(#[from] CoreError),
    #[error(transparent)]
    Data(#[from] DataError),
}

impl ArtifactError {
    pub fn io(path: &Path, source: std::io::Error) -> Self {
        ArtifactError::Io {
            path: path.display().to_string(),
            source,
        }
    }
}

pub type Result<T> = std::result::Result<T, ArtifactError>;
