//! Native core of the `boundml` Python package. The Python layer (`boundml/__init__.py`)
//! converts NumPy arrays, pandas DataFrames and dict rows into the two entry points
//! here; everything else (validation, preprocessing, planned execution, training)
//! is the same Rust code the CLI uses.
//!
//! Errors are raised as `boundml_rs.BoundMLError(message, code)`. Computation runs with
//! the GIL released, so other Python threads keep running.

use boundml_artifact::{Package, Predictor};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// `BML-xxxx` code at the start of an error message (`BML-E000` if none).
fn code_of(message: &str) -> &str {
    message
        .split_whitespace()
        .next()
        .filter(|w| w.starts_with("BML-"))
        .unwrap_or("BML-E000")
}

/// Build a `boundml_rs.BoundMLError(message, code)`.
fn error(py: Python<'_>, message: impl Into<String>) -> PyErr {
    let message = message.into();
    let code = code_of(&message).to_string();
    match py
        .import("boundml_rs._errors")
        .and_then(|m| m.getattr("BoundMLError"))
        .and_then(|cls| cls.call1((message.clone(), code)))
    {
        Ok(exc) => PyErr::from_value(exc),
        Err(e) => e,
    }
}

fn json<T: serde::Serialize>(py: Python<'_>, value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(|e| error(py, format!("BML-E000 serialization: {e}")))
}

/// A loaded, validated package plus one lazily created predictor (one session).
#[pyclass(module = "boundml_rs._native", frozen)]
struct NativeModel {
    pkg: Arc<Package>,
    predictor: Mutex<Option<Predictor>>,
}

impl NativeModel {
    fn new(pkg: Package) -> Self {
        Self {
            pkg: Arc::new(pkg),
            predictor: Mutex::new(None),
        }
    }

    /// Run `f` with this model's predictor, creating it on first use.
    fn with_predictor<R>(
        &self,
        f: impl FnOnce(&mut Predictor) -> boundml_artifact::Result<R>,
    ) -> Result<R, String> {
        let mut slot = self.predictor.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            let batch = self.pkg.manifest().bounds.max_batch;
            *slot = Some(self.pkg.predictor(batch).map_err(|e| e.to_string())?);
        }
        f(slot.as_mut().expect("created above")).map_err(|e| e.to_string())
    }

    fn output_array<'py>(
        py: Python<'py>,
        out: Vec<f32>,
        cols: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let rows: Vec<Vec<f32>> = out.chunks(cols.max(1)).map(<[f32]>::to_vec).collect();
        if rows.is_empty() {
            return Ok(PyArray2::zeros(py, [0, cols], false));
        }
        PyArray2::from_vec2(py, &rows).map_err(|e| error(py, format!("BML-E000 output array: {e}")))
    }
}

/// NaN means "missing"; every other value (including infinities, which the runtime
/// rejects) is passed through.
fn cell(v: f64) -> Option<f64> {
    if v.is_nan() { None } else { Some(v) }
}

#[pymethods]
impl NativeModel {
    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let pkg = py
            .detach(|| Package::load(&path))
            .map_err(|e| error(py, e.to_string()))?;
        Ok(Self::new(pkg))
    }

    /// The full manifest as JSON.
    fn manifest_json(&self, py: Python<'_>) -> PyResult<String> {
        json(py, self.pkg.manifest())
    }

    fn output_names(&self) -> Vec<String> {
        self.pkg.manifest().output_names()
    }

    /// Planned tensor memory for `sessions` sessions, as JSON.
    fn memory_report_json(&self, py: Python<'_>, sessions: usize) -> PyResult<String> {
        let r = self
            .pkg
            .memory_report(sessions)
            .map_err(|e| error(py, e.to_string()))?;
        json(py, &r)
    }

    /// Re-run the bundled self-test. The model's own predictor is released first so
    /// that a package declaring one session can still be verified.
    fn verify_json(&self, py: Python<'_>) -> PyResult<String> {
        let report = py.detach(|| {
            self.predictor
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            self.pkg.verify().map_err(|e| e.to_string())
        });
        let report = report.map_err(|e| error(py, e))?;
        json(py, &report)
    }

    /// Predict from raw values in schema order: `x` is `[rows, n_features]` float64,
    /// NaN = missing. Returns `[rows, outputs]` float32.
    fn predict_array<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = x.as_array().shape().to_vec();
        let nf = self.pkg.manifest().schema.features.len();
        if shape[1] != nf {
            return Err(error(
                py,
                format!(
                    "BML-D002 expected {nf} feature columns in schema order, got {}",
                    shape[1]
                ),
            ));
        }
        let raw: Vec<Option<f64>> = x.as_array().iter().copied().map(cell).collect();
        let no = self.pkg.manifest().task.output_cols();
        let out = py
            .detach(|| self.predict_flat(&raw, shape[0], nf, no))
            .map_err(|e| error(py, e))?;
        Self::output_array(py, out, no)
    }

    /// Predict from named columns: `names[j]` labels column `j` of `x`. Columns are
    /// mapped by name; absent features are missing; unknown names follow the schema's
    /// `extra_columns` policy. NaN = missing.
    fn predict_named<'py>(
        &self,
        py: Python<'py>,
        names: Vec<String>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = x.as_array().shape().to_vec();
        if shape[1] != names.len() {
            return Err(error(
                py,
                format!(
                    "BML-D002 {} column names for {} columns",
                    names.len(),
                    shape[1]
                ),
            ));
        }
        let values: Vec<f64> = x.as_array().iter().copied().collect();
        let nf = self.pkg.manifest().schema.features.len();
        let no = self.pkg.manifest().task.output_cols();
        let out = py
            .detach(|| {
                let mut raw = vec![None; shape[0] * nf];
                self.with_predictor(|p| {
                    for (r, dst) in raw.chunks_mut(nf).enumerate() {
                        let row = &values[r * shape[1]..(r + 1) * shape[1]];
                        let pairs = names
                            .iter()
                            .map(String::as_str)
                            .zip(row.iter().map(|v| cell(*v)));
                        p.fill_named(pairs, dst, r + 1)?;
                    }
                    Ok(())
                })?;
                self.predict_flat(&raw, shape[0], nf, no)
            })
            .map_err(|e| error(py, e))?;
        Self::output_array(py, out, no)
    }
}

impl NativeModel {
    fn predict_flat(
        &self,
        raw: &[Option<f64>],
        rows: usize,
        nf: usize,
        no: usize,
    ) -> Result<Vec<f32>, String> {
        let mut out = vec![0.0f32; rows * no];
        self.with_predictor(|p| {
            let batch = p.max_batch();
            for (i, chunk) in raw.chunks(batch * nf.max(1)).enumerate() {
                let n = chunk.len() / nf.max(1);
                let at = i * batch * no;
                p.predict_raw(chunk, n, &mut out[at..at + n * no])?;
            }
            Ok(())
        })?;
        Ok(out)
    }
}

/// Train from a TOML config into a checkpoint directory. Returns the outcome as JSON.
/// Ctrl-C (KeyboardInterrupt) stops after the next step and saves a checkpoint.
#[pyfunction]
#[pyo3(signature = (config, checkpoint, resume = false, max_steps = None, verbose = false))]
fn train_json(
    py: Python<'_>,
    config: PathBuf,
    checkpoint: PathBuf,
    resume: bool,
    max_steps: Option<u64>,
    verbose: bool,
) -> PyResult<String> {
    let (cfg, base) =
        boundml_train::TrainConfig::load(&config).map_err(|e| error(py, e.to_string()))?;
    let interrupted = AtomicBool::new(false);
    let should_stop = || {
        if !interrupted.load(Ordering::Relaxed) && Python::attach(|py| py.check_signals()).is_err()
        {
            interrupted.store(true, Ordering::Relaxed);
        }
        interrupted.load(Ordering::Relaxed)
    };
    let mut on_event = |e: &boundml_train::TrainEvent| {
        if let (true, boundml_train::TrainEvent::Epoch(r)) = (verbose, e) {
            eprintln!("epoch {:>3}  train_loss={:.5}", r.epoch + 1, r.train_loss);
        }
    };
    let outcome = py.detach(|| {
        boundml_train::train(
            &cfg,
            &base,
            boundml_train::TrainOptions {
                checkpoint_dir: checkpoint,
                resume,
                max_steps,
                should_stop: &should_stop,
                on_event: &mut on_event,
            },
        )
        .map_err(|e| e.to_string())
    });
    if interrupted.load(Ordering::Relaxed) {
        return Err(pyo3::exceptions::PyKeyboardInterrupt::new_err(
            "training interrupted; the checkpoint was saved and can be resumed",
        ));
    }
    let o = outcome.map_err(|e| error(py, e))?;
    let history: Vec<_> = o
        .history
        .iter()
        .map(|r| serde_json::json!({"epoch": r.epoch + 1, "train_loss": r.train_loss, "valid": r.valid}))
        .collect();
    json(
        py,
        &serde_json::json!({"completed": o.completed, "global_step": o.global_step,
            "history": history, "memory": o.report}),
    )
}

/// Metrics of a checkpoint or package on a labelled CSV, with the constant baseline.
#[pyfunction]
fn evaluate_json(py: Python<'_>, path: PathBuf, data: PathBuf) -> PyResult<String> {
    let e = py
        .detach(|| boundml_train::evaluate(&path, &data))
        .map_err(|e| error(py, e.to_string()))?;
    json(
        py,
        &serde_json::json!({"model": e.model, "baseline": e.baseline}),
    )
}

/// Pack a checkpoint into a package directory and return it, loaded.
#[pyfunction]
#[pyo3(signature = (checkpoint, out, name = None))]
fn pack(
    py: Python<'_>,
    checkpoint: PathBuf,
    out: PathBuf,
    name: Option<String>,
) -> PyResult<NativeModel> {
    let pkg = py
        .detach(|| boundml_train::pack(&checkpoint, &out, name.as_deref()))
        .map_err(|e| error(py, e.to_string()))?;
    Ok(NativeModel::new(pkg))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeModel>()?;
    m.add_function(wrap_pyfunction!(train_json, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_json, m)?)?;
    m.add_function(wrap_pyfunction!(pack, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
