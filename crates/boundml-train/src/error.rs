use boundml_artifact::ArtifactError;
use boundml_core::CoreError;
use boundml_data::DataError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TrainError {
    #[error("BML-T001 configuration error: {0}")]
    Config(String),
    #[error("BML-T002 checkpoint error: {0}")]
    Checkpoint(String),
    #[error("BML-T003 training data changed since the checkpoint: {0}")]
    DataChanged(String),
    #[error(transparent)]
    Core(#[from] CoreError),
    #[error(transparent)]
    Data(#[from] DataError),
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
}

pub type Result<T> = std::result::Result<T, TrainError>;
