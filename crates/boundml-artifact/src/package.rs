//! Inference packages: a directory with `manifest.json`, `MANIFEST.sha256` and
//! `weights.safetensors`. Loading re-validates everything and recomputes the memory
//! plan under the installed runtime; nothing in the manifest is trusted blindly.

use crate::error::{ArtifactError, Result};
use crate::fsutil::{
    check_dir, decode_tensors, encode_tensors, read_limited, sha256_hex, unix_now, write_dir_atomic,
};
use crate::manifest::*;
use crate::predictor::Predictor;
use boundml_core::{Graph, GraphDef, MemoryReport, Model};
use boundml_data::{Preprocessor, Schema};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

pub const MANIFEST_FILE: &str = "manifest.json";
pub const MANIFEST_HASH_FILE: &str = "MANIFEST.sha256";
pub const WEIGHTS_FILE: &str = "weights.safetensors";
pub const MAX_MANIFEST_BYTES: u64 = 16 << 20;
pub const MAX_WEIGHTS_BYTES: u64 = 1 << 30;
/// Hard ceiling on a package's planned inference tensor memory (all declared
/// sessions plus weights), enforced on save and load whether or not the package
/// declares its own budget. BoundML targets small models; this stops a package from
/// making an application or server preallocate an arbitrary amount of memory.
pub const MAX_PLANNED_TENSOR_BYTES: u64 = 1 << 30;
/// Upper bound on `bounds.sessions`.
pub const MAX_SESSIONS: usize = 256;

/// Everything needed to write a package (the manifest is derived from it).
#[derive(Debug, Clone)]
pub struct PackageSpec {
    pub name: String,
    pub schema: Schema,
    pub preprocessing: Preprocessor,
    pub graph: GraphDef,
    pub weights: Vec<f32>,
    pub bounds: Bounds,
    pub provenance: Provenance,
    /// Raw input rows for self-tests. If `expected` is `None`, the expected outputs
    /// are computed by this runtime at pack time (regression detection only).
    pub test_inputs: Vec<Vec<Option<f64>>>,
    pub test_expected: Option<Vec<Vec<f32>>>,
    pub test_origin: String,
    pub test_atol: f32,
    pub test_rtol: f32,
}

/// A loaded, validated package. Immutable; share with `Arc`.
#[derive(Debug)]
pub struct Package {
    manifest: Manifest,
    model: Arc<Model>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifyReport {
    pub vectors: usize,
    pub max_abs_error: f32,
    pub passed: bool,
    /// First failing vector or evaluation error, when `passed` is false.
    #[serde(default)]
    pub failure: Option<String>,
}

impl Package {
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    /// Validate `spec`, compute test-vector outputs if needed and write atomically.
    pub fn save(dir: &Path, spec: PackageSpec) -> Result<Package> {
        let graph = Graph::new(spec.graph.clone())?;
        let tensors: Vec<TensorInfo> = graph
            .params()
            .iter()
            .map(|p| TensorInfo {
                name: p.name.clone(),
                shape: [p.rows, p.cols],
                dtype: "f32".into(),
                layout: "row_major".into(),
            })
            .collect();
        let model = Arc::new(Model::new(graph, spec.weights)?);
        let manifest = Manifest {
            format: PACKAGE_FORMAT.into(),
            format_version: PACKAGE_FORMAT_VERSION,
            min_runtime_version: MIN_RUNTIME_VERSION.into(),
            name: spec.name,
            task: spec.schema.task(),
            schema: spec.schema,
            preprocessing: spec.preprocessing,
            graph: spec.graph,
            tensors,
            bounds: spec.bounds,
            provenance: Provenance {
                created_unix: unix_now(),
                created_by: format!("boundml {}", boundml_core::VERSION),
                ..spec.provenance
            },
            files: BTreeMap::new(),
            test_vectors: None,
        };
        check_consistency(&manifest, &model)?;
        let pkg = Package { manifest, model };
        let vectors = if spec.test_inputs.is_empty() {
            None
        } else {
            let expected = match spec.test_expected {
                Some(e) => e,
                None => pkg.compute_outputs(&spec.test_inputs)?,
            };
            Some(TestVectors {
                origin: spec.test_origin,
                inputs: spec.test_inputs,
                expected,
                atol: spec.test_atol,
                rtol: spec.test_rtol,
            })
        };
        let Package {
            mut manifest,
            model,
        } = pkg;
        manifest.test_vectors = vectors;
        check_test_vectors(&manifest)?;
        // Self-test in memory first so a failing package is never written.
        let pkg = Package { manifest, model };
        let pre_write = pkg.verify()?;
        if !pre_write.passed {
            return Err(ArtifactError::SelfTest(format!(
                "outputs differ from the expected test vectors ({}); nothing was written",
                pre_write.failure.as_deref().unwrap_or("unknown failure")
            )));
        }
        let Package {
            mut manifest,
            model,
        } = pkg;

        let g = model.graph();
        let named: Vec<(String, [usize; 2], &[f32])> = (0..g.params().len())
            .map(|i| {
                let p = &g.params()[i];
                let o = g.param_offset(i);
                (
                    p.name.clone(),
                    [p.rows, p.cols],
                    &model.weights()[o..o + g.param_len(i)],
                )
            })
            .collect();
        let weights = encode_tensors(&named)?;
        manifest.files.insert(
            WEIGHTS_FILE.into(),
            FileHash {
                sha256: sha256_hex(&weights),
                bytes: weights.len() as u64,
            },
        );
        let manifest_json = serde_json::to_vec_pretty(&manifest)
            .map_err(|e| ArtifactError::Format(e.to_string()))?;
        let manifest_hash = format!("{}\n", sha256_hex(&manifest_json));
        write_dir_atomic(
            dir,
            &[
                (MANIFEST_FILE, &manifest_json),
                (MANIFEST_HASH_FILE, manifest_hash.as_bytes()),
                (WEIGHTS_FILE, &weights),
            ],
        )?;
        drop(model);
        // Re-read what was written so the self-test covers the persisted bytes.
        let pkg = Package::load(dir)?;
        let report = pkg.verify()?;
        if !report.passed {
            return Err(ArtifactError::SelfTest(format!(
                "package written to {} failed its own test vectors (max error {})",
                dir.display(),
                report.max_abs_error
            )));
        }
        Ok(pkg)
    }

    /// Load and fully validate a package directory.
    pub fn load(dir: &Path) -> Result<Package> {
        check_dir(dir)?;
        let manifest_bytes = read_limited(&dir.join(MANIFEST_FILE), MAX_MANIFEST_BYTES)?;
        let hash_text = read_limited(&dir.join(MANIFEST_HASH_FILE), 1024)?;
        let expected = String::from_utf8_lossy(&hash_text).trim().to_string();
        let found = sha256_hex(&manifest_bytes);
        if expected != found {
            return Err(ArtifactError::Integrity {
                file: MANIFEST_FILE.into(),
                expected,
                found,
            });
        }
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|e| ArtifactError::Format(format!("manifest.json: {e}")))?;
        manifest.check_version()?;
        let unexpected: Vec<&String> = manifest
            .files
            .keys()
            .filter(|k| *k != WEIGHTS_FILE)
            .collect();
        if !unexpected.is_empty() {
            return Err(ArtifactError::Format(format!(
                "unexpected files listed in manifest: {unexpected:?}"
            )));
        }
        let fh = manifest.files.get(WEIGHTS_FILE).ok_or_else(|| {
            ArtifactError::Format("manifest does not list weights.safetensors".into())
        })?;
        if fh.bytes > MAX_WEIGHTS_BYTES {
            return Err(ArtifactError::TooLarge {
                what: "declared weights".into(),
                bytes: fh.bytes,
                limit: MAX_WEIGHTS_BYTES,
            });
        }
        let weights_bytes = read_limited(&dir.join(WEIGHTS_FILE), fh.bytes)?;
        let found = sha256_hex(&weights_bytes);
        if found != fh.sha256 || weights_bytes.len() as u64 != fh.bytes {
            return Err(ArtifactError::Integrity {
                file: WEIGHTS_FILE.into(),
                expected: fh.sha256.clone(),
                found,
            });
        }
        let graph = Graph::new(manifest.graph.clone()).map_err(rejected)?;
        let decoded = decode_tensors(&weights_bytes, boundml_core::graph::MAX_PARAMS)?;
        let weights = assemble_weights(&graph, &manifest.tensors, decoded)?;
        let model = Arc::new(Model::new(graph, weights).map_err(rejected)?);
        check_consistency(&manifest, &model).map_err(|e| match e {
            ArtifactError::Core(c) => rejected(c),
            ArtifactError::Data(d) => ArtifactError::Inconsistent(d.to_string()),
            other => other,
        })?;
        check_test_vectors(&manifest)?;
        Ok(Package { manifest, model })
    }

    /// Memory report for `sessions` sessions at the package's `max_batch`.
    pub fn memory_report(&self, sessions: usize) -> Result<MemoryReport> {
        Ok(self.model.inference_report(
            self.manifest.bounds.max_batch,
            sessions,
            self.manifest.bounds.planner,
        )?)
    }

    pub fn predictor(self: &Arc<Self>, max_batch: usize) -> Result<Predictor> {
        Predictor::new(Arc::clone(self), max_batch)
    }

    fn compute_outputs(&self, inputs: &[Vec<Option<f64>>]) -> Result<Vec<Vec<f32>>> {
        let tmp = Arc::new(Package {
            manifest: self.manifest.clone(),
            model: Arc::clone(&self.model),
        });
        let mut p = Predictor::new(tmp, 1)?;
        inputs.iter().map(|row| p.predict_one(row)).collect()
    }

    /// Run the bundled test vectors through the full preprocessing + model path.
    pub fn verify(&self) -> Result<VerifyReport> {
        let Some(tv) = &self.manifest.test_vectors else {
            return Ok(VerifyReport {
                vectors: 0,
                max_abs_error: 0.0,
                passed: true,
                failure: None,
            });
        };
        let got = match self.compute_outputs(&tv.inputs) {
            Ok(g) => g,
            // A vector the model cannot evaluate (e.g. its output overflows to inf/NaN)
            // is a failed self-test, never a pass.
            Err(e) => {
                return Ok(VerifyReport {
                    vectors: tv.inputs.len(),
                    max_abs_error: f32::INFINITY,
                    passed: false,
                    failure: Some(e.to_string()),
                });
            }
        };
        let mut max_err = 0.0f32;
        let mut failure = None;
        for (row, (g, e)) in got.iter().zip(&tv.expected).enumerate() {
            for (a, b) in g.iter().zip(e) {
                let err = (a - b).abs();
                // NaN anywhere fails: a value only passes if it is finite and within tolerance.
                if !err.is_finite() || !a.is_finite() || !b.is_finite() {
                    max_err = f32::INFINITY;
                    failure.get_or_insert(format!(
                        "vector {row}: non-finite output {a} or expected {b}"
                    ));
                } else {
                    max_err = max_err.max(err);
                }
                if !(err.is_finite() && err <= tv.atol + tv.rtol * b.abs()) {
                    failure.get_or_insert(format!("vector {row}: got {a}, expected {b}"));
                }
            }
        }
        Ok(VerifyReport {
            vectors: tv.inputs.len(),
            max_abs_error: max_err,
            passed: failure.is_none(),
            failure,
        })
    }
}

/// A loaded package whose graph or contents violate core rules is a rejected
/// artifact (exit code 5), not invalid user input. Budget overruns keep their code.
fn rejected(e: boundml_core::CoreError) -> ArtifactError {
    match e {
        e @ boundml_core::CoreError::BudgetExceeded { .. } => ArtifactError::Core(e),
        e => ArtifactError::Inconsistent(e.to_string()),
    }
}

fn assemble_weights(
    graph: &Graph,
    declared: &[TensorInfo],
    decoded: Vec<(String, [usize; 2], Vec<f32>)>,
) -> Result<Vec<f32>> {
    let params = graph.params();
    if declared.len() != params.len() || decoded.len() != params.len() {
        return Err(ArtifactError::Inconsistent(format!(
            "graph has {} parameters, manifest declares {}, weights file has {}",
            params.len(),
            declared.len(),
            decoded.len()
        )));
    }
    let mut out = vec![0.0f32; graph.param_count()];
    for (i, p) in params.iter().enumerate() {
        let d = &declared[i];
        if d.name != p.name
            || d.shape != [p.rows, p.cols]
            || d.dtype != "f32"
            || d.layout != "row_major"
        {
            return Err(ArtifactError::Inconsistent(format!(
                "tensor declaration {:?} does not match graph parameter '{}' [{}, {}] f32 row_major",
                d, p.name, p.rows, p.cols
            )));
        }
        let (_, shape, values) =
            decoded
                .iter()
                .find(|(n, _, _)| *n == p.name)
                .ok_or_else(|| {
                    ArtifactError::Inconsistent(format!("weights file lacks tensor '{}'", p.name))
                })?;
        if *shape != [p.rows, p.cols] {
            return Err(ArtifactError::Inconsistent(format!(
                "tensor '{}' has shape {shape:?}, graph expects [{}, {}]",
                p.name, p.rows, p.cols
            )));
        }
        let o = graph.param_offset(i);
        out[o..o + values.len()].copy_from_slice(values);
    }
    Ok(out)
}

fn check_consistency(m: &Manifest, model: &Model) -> Result<()> {
    m.schema.validate()?;
    m.preprocessing.validate(&m.schema)?;
    let inc = |s: String| Err(ArtifactError::Inconsistent(s));
    if m.task != m.schema.task() {
        return inc(format!(
            "task {:?} does not match schema label kind ({:?})",
            m.task,
            m.schema.task()
        ));
    }
    if model.input_cols() != m.schema.features.len() {
        return inc(format!(
            "graph input has {} columns, schema has {} features",
            model.input_cols(),
            m.schema.features.len()
        ));
    }
    if model.output_cols() != m.task.output_cols() {
        return inc(format!(
            "graph output has {} columns, task needs {}",
            model.output_cols(),
            m.task.output_cols()
        ));
    }
    if m.name.is_empty() || m.name.len() > 128 {
        return inc("package name must be 1..=128 bytes".into());
    }
    let b = &m.bounds;
    if b.sessions == 0 || b.sessions > MAX_SESSIONS {
        return inc(format!("bounds.sessions must be 1..={MAX_SESSIONS}"));
    }
    let report = model.inference_report(b.max_batch, b.sessions, b.planner)?;
    if let Some(budget) = b.tensor_budget_bytes {
        report.check_budget(budget)?;
    }
    if report.total_planned_bytes > MAX_PLANNED_TENSOR_BYTES {
        return Err(ArtifactError::TooLarge {
            what: format!(
                "planned inference tensor memory for max_batch {} x {} session(s)",
                b.max_batch, b.sessions
            ),
            bytes: report.total_planned_bytes,
            limit: MAX_PLANNED_TENSOR_BYTES,
        });
    }
    Ok(())
}

fn check_test_vectors(m: &Manifest) -> Result<()> {
    let Some(tv) = &m.test_vectors else {
        return Ok(());
    };
    let nf = m.schema.features.len();
    let no = m.task.output_cols();
    if tv.inputs.len() > MAX_TEST_VECTORS || tv.inputs.len() != tv.expected.len() {
        return Err(ArtifactError::Inconsistent(format!(
            "test vectors: {} inputs vs {} expected (max {MAX_TEST_VECTORS})",
            tv.inputs.len(),
            tv.expected.len()
        )));
    }
    if tv.inputs.iter().any(|r| r.len() != nf) || tv.expected.iter().any(|r| r.len() != no) {
        return Err(ArtifactError::Inconsistent(
            "test vector row width mismatch".into(),
        ));
    }
    if !(tv.atol.is_finite() && tv.atol >= 0.0 && tv.rtol.is_finite() && tv.rtol >= 0.0) {
        return Err(ArtifactError::Inconsistent(
            "test vector tolerances must be finite and >= 0".into(),
        ));
    }
    let bad_input = tv
        .inputs
        .iter()
        .flatten()
        .any(|v| v.is_some_and(|x| !x.is_finite()));
    let bad_expected = tv.expected.iter().flatten().any(|v| !v.is_finite());
    if bad_input || bad_expected {
        return Err(ArtifactError::Inconsistent(
            "test vectors must contain only finite inputs and expected outputs".into(),
        ));
    }
    Ok(())
}
