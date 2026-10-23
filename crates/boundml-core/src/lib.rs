//! BoundML core: graph IR, CPU kernels, reverse-mode differentiation, static
//! tensor-memory planning and optimizers for small dense models.
//!
//! This crate has no I/O, parsing, or networking. See the workspace `README.md` for
//! the end-to-end workflow and `docs/architecture.md` for semantics.

mod arena;
pub mod error;
pub mod exec;
pub mod graph;
pub mod kernels;
pub mod model;
pub mod optim;
pub mod plan;
pub mod rng;
pub mod shape;

pub use error::{CoreError, Result};
pub use exec::Executor;
pub use graph::{Graph, GraphBuilder, GraphDef, Op};
pub use model::{Activation, DenseSpec, InferenceSession, Model, Task};
pub use optim::{Optimizer, OptimizerConfig, OptimizerState};
pub use plan::{MemoryReport, Mode, Plan, PlannerKind};
pub use rng::Rng;
pub use shape::{Dim, Shape};

/// Runtime version, recorded in packages and checked against `min_runtime_version`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
