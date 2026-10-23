//! BoundML training: declarative configuration, the training loop, metrics,
//! resumable checkpoints and packaging into inference artifacts.

pub mod checkpoint;
pub mod config;
mod error;
pub mod metrics;
pub mod pack;
pub mod trainer;

pub use checkpoint::{Checkpoint, EpochRecord};
pub use config::TrainConfig;
pub use error::{Result, TrainError};
pub use pack::{Evaluation, evaluate, pack};
pub use trainer::{TrainEvent, TrainOptions, TrainOutcome, plan_inference, plan_training, train};
