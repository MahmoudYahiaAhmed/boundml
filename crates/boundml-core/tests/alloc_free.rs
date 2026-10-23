//! Counts heap allocations on the current thread inside the initialized prediction
//! path and training-step path (forward + backward + optimizer update). Both must be
//! zero after construction.

use boundml_core::{
    Activation, DenseSpec, Executor, InferenceSession, Mode, Model, Optimizer, OptimizerConfig,
    PlannerKind, Rng, Task,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

struct Counting;

thread_local! {
    static COUNT: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = COUNT.try_with(|c| c.set(c.get() + 1));
        // SAFETY: forwarding to the system allocator with the caller's layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by `System` with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new: usize) -> *mut u8 {
        let _ = COUNT.try_with(|c| c.set(c.get() + 1));
        // SAFETY: forwarding to the system allocator.
        unsafe { System.realloc(ptr, layout, new) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocations<F: FnMut()>(mut f: F) -> usize {
    let before = COUNT.with(|c| c.get());
    f();
    COUNT.with(|c| c.get()) - before
}

fn spec(task: Task) -> DenseSpec {
    DenseSpec {
        inputs: 6,
        hidden: vec![16, 8],
        activation: Activation::Relu,
        task,
    }
}

#[test]
fn prediction_and_training_paths_do_not_allocate() {
    for task in [
        Task::Regression,
        Task::Binary,
        Task::Multiclass { classes: 3 },
    ] {
        let spec = spec(task);
        let graph = spec.graph().unwrap();
        let params = spec.init_params(&graph, &mut Rng::seed(1));
        let model = Arc::new(Model::new(graph.clone(), params.clone()).unwrap());
        let mut session = InferenceSession::new(Arc::clone(&model), 32).unwrap();
        let x = vec![0.25f32; 32 * 6];
        let mut out = vec![0.0f32; 32 * model.output_cols()];

        session.predict_into(&x, 32, &mut out).unwrap();
        let n = allocations(|| {
            for b in [1usize, 7, 32] {
                session
                    .predict_into(&x[..b * 6], b, &mut out[..b * model.output_cols()])
                    .unwrap();
            }
        });
        assert_eq!(n, 0, "prediction allocated {n} times for {task:?}");

        let graph = Arc::new(graph);
        let mut ex =
            Executor::new(Arc::clone(&graph), Mode::Training, 32, PlannerKind::Reuse).unwrap();
        let mut opt = Optimizer::new(
            OptimizerConfig::AdamW {
                lr: 1e-3,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.01,
            },
            &graph,
        )
        .unwrap();
        let mut p = params.clone();
        let y = vec![1.0f32; 32];
        ex.forward_backward(&p, &x, &y, 32).unwrap();
        let n = allocations(|| {
            for b in [32usize, 5] {
                ex.forward_backward(&p, &x[..b * 6], &y[..b], b).unwrap();
                opt.step(&mut p, ex.param_grads()).unwrap();
            }
        });
        assert_eq!(n, 0, "training step allocated {n} times for {task:?}");
    }
}

#[test]
fn counter_detects_allocation() {
    let n = allocations(|| {
        let v = vec![1u8; 10];
        std::hint::black_box(v);
    });
    assert!(n >= 1);
}
