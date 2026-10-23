//! Typed errors for the computational core.
//!
//! Every variant carries a stable code (`BML-Cxxx`) documented in `docs/errors.md`.

use thiserror::Error;

/// One large contributor to a memory plan, reported by budget errors.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Contributor {
    pub name: String,
    pub bytes: u64,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum CoreError {
    #[error("BML-C001 shape error: {0}")]
    Shape(String),
    #[error("BML-C002 size overflow while computing {0}")]
    Overflow(&'static str),
    #[error("BML-C003 invalid graph: {0}")]
    Graph(String),
    #[error("BML-C004 batch size {got} outside the planned range 1..={max}")]
    BatchOutOfRange { got: usize, max: usize },
    #[error(
        "BML-C005 tensor budget exceeded: plan needs {requested} bytes but the budget is {available} bytes; largest contributors: {}",
        fmt_contributors(.contributors)
    )]
    BudgetExceeded {
        requested: u64,
        available: u64,
        contributors: Vec<Contributor>,
    },
    #[error("BML-C006 buffer length mismatch for {what}: expected {expected}, got {got}")]
    Length {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("BML-C007 non-finite value encountered in {0}")]
    NonFinite(&'static str),
    #[error("BML-C008 class index {value} invalid for {classes} classes (row {row})")]
    ClassIndex {
        value: f32,
        classes: usize,
        row: usize,
    },
    #[error("BML-C009 invalid optimizer configuration: {0}")]
    Optimizer(String),
}

fn fmt_contributors(c: &[Contributor]) -> String {
    c.iter()
        .map(|c| format!("{} ({} bytes)", c.name, c.bytes))
        .collect::<Vec<_>>()
        .join(", ")
}

pub type Result<T> = std::result::Result<T, CoreError>;
