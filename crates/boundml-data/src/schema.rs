use crate::error::{DataError, Result};
use boundml_core::Task;
use serde::{Deserialize, Serialize};

pub const MAX_FEATURES: usize = 65_536;
pub const MAX_CLASSES: usize = 4096;
pub const MAX_NAME: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MissingPolicy {
    /// Any missing feature value is an error.
    #[default]
    Reject,
    /// Replace a missing value with the training-set mean of that feature.
    ImputeMean,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExtraColumns {
    /// Columns that are neither features nor the label are ignored.
    #[default]
    Ignore,
    /// Columns that are neither features nor the label are an error.
    Reject,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LabelKind {
    /// Continuous numeric label.
    Regression,
    /// Label text is matched exactly (after trimming) against `classes`; the index is
    /// the class id. Two classes => binary task, positive class is `classes[1]`.
    Classes { classes: Vec<String> },
}

/// The input contract of a model: which named columns, in which order, and how
/// to treat missing or unexpected data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Schema {
    pub features: Vec<String>,
    pub label: String,
    pub label_kind: LabelKind,
    #[serde(default)]
    pub missing: MissingPolicy,
    #[serde(default)]
    pub extra_columns: ExtraColumns,
}

impl Schema {
    pub fn validate(&self) -> Result<()> {
        let err = |m: String| Err(DataError::Schema(m));
        if self.features.is_empty() || self.features.len() > MAX_FEATURES {
            return err(format!("feature count must be 1..={MAX_FEATURES}"));
        }
        let mut seen = std::collections::HashSet::new();
        for f in self.features.iter().chain(std::iter::once(&self.label)) {
            if f.trim().is_empty() || f.len() > MAX_NAME || f.trim() != f {
                return err(format!(
                    "column name {f:?} must be non-empty, <= {MAX_NAME} bytes, without surrounding spaces"
                ));
            }
            if !seen.insert(f.as_str()) {
                return err(format!(
                    "column '{f}' is declared twice (features and label must be distinct)"
                ));
            }
        }
        if let LabelKind::Classes { classes } = &self.label_kind {
            if classes.len() < 2 || classes.len() > MAX_CLASSES {
                return err(format!("classification needs 2..={MAX_CLASSES} classes"));
            }
            let mut s = std::collections::HashSet::new();
            for c in classes {
                if c.trim() != c || c.is_empty() || !s.insert(c.as_str()) {
                    return err(format!(
                        "class labels must be unique, non-empty and trimmed; bad: {c:?}"
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn task(&self) -> Task {
        match &self.label_kind {
            LabelKind::Regression => Task::Regression,
            LabelKind::Classes { classes } if classes.len() == 2 => Task::Binary,
            LabelKind::Classes { classes } => Task::Multiclass {
                classes: classes.len(),
            },
        }
    }

    pub fn classes(&self) -> Option<&[String]> {
        match &self.label_kind {
            LabelKind::Classes { classes } => Some(classes),
            LabelKind::Regression => None,
        }
    }
}
