use thiserror::Error;

#[derive(Debug, Error)]
pub enum DataError {
    #[error("BML-D001 cannot read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("BML-D002 CSV format error in {path}: {message}")]
    Csv { path: String, message: String },
    #[error("BML-D003 schema error: {0}")]
    Schema(String),
    #[error("BML-D004 header of {path}: {message}")]
    Header { path: String, message: String },
    #[error("BML-D005 {path} line {line}, column '{column}': {message} (value: {value:?})")]
    Value {
        path: String,
        line: u64,
        column: String,
        value: String,
        message: String,
    },
    #[error(
        "BML-D006 {path} line {line}: missing value in feature '{column}' and missing policy is 'reject'"
    )]
    Missing {
        path: String,
        line: u64,
        column: String,
    },
    #[error("BML-D007 {0}")]
    Empty(String),
    #[error("BML-D008 preprocessing mismatch: {0}")]
    Preprocess(String),
    #[error("BML-D009 resume position invalid: {0}")]
    Resume(String),
}

pub type Result<T> = std::result::Result<T, DataError>;
