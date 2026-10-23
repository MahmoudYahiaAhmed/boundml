#![doc = include_str!("../README.md")]
//!
//! BoundML artifacts: versioned inference packages, the embeddable [`Predictor`],
//! package comparison, and file utilities used by checkpoints.
//!
//! Trust model: SHA-256 hashes detect accidental corruption and truncation. They do
//! **not** authenticate the publisher; anyone who can write the package can rewrite
//! its hashes. Packages contain no executable code. See `docs/package-format.md`.

pub mod compare;
mod error;
pub mod fsutil;
pub mod manifest;
mod package;
mod predictor;

pub use compare::{CompareOptions, CompareReport, Verdict, compare};
pub use error::{ArtifactError, Result};
pub use manifest::{Bounds, DataRef, Manifest, Provenance, TestVectors};
pub use package::{Package, PackageSpec, VerifyReport};
pub use predictor::Predictor;
