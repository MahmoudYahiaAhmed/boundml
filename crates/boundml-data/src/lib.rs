//! Numeric CSV input with an explicit schema, train-only fitted preprocessing and
//! bounded-memory, deterministic, resumable minibatch streaming.
//!
//! Policies (see `docs/data.md`):
//! * Features are mapped by **name**, never by position. Missing or duplicated
//!   header columns are errors; unexpected columns follow [`ExtraColumns`].
//! * An empty cell is *missing*. Missing features follow [`MissingPolicy`]; a
//!   missing label is always an error.
//! * Literal `NaN`/`inf` or any non-numeric text in a feature cell is an error.
//! * Normalization statistics are fitted once on training data and then reused
//!   unchanged for validation, testing and production.

mod csvsrc;
mod error;
mod limit;
mod preprocess;
mod reserve;
mod schema;
mod split;
mod stream;

pub use csvsrc::{CsvSource, MAX_COLUMNS, MAX_RECORD_BYTES, RawRow};
pub use error::{DataError, Result};
pub use preprocess::{FeatureStats, Normalize, Preprocessor};
pub use reserve::{ReservedFile, same_file};
pub use schema::{ExtraColumns, LabelKind, MissingPolicy, Schema};
pub use split::{SplitCounts, split_csv};
pub use stream::{BatchReader, ReaderPosition};
