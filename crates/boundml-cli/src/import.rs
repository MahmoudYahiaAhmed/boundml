//! Import a dense sequential model exported from PyTorch or Keras by the scripts in
//! `migrate/`. The bundle declares the architecture explicitly; weights are only
//! data. Unsupported layers are rejected by the exporter, and anything this importer
//! does not understand is rejected here too. Source-framework outputs on reference
//! inputs become the package's test vectors, so `Package::save` fails unless the
//! imported model reproduces them within the declared tolerance.

use boundml_artifact::fsutil::{decode_tensors, read_limited};
use boundml_artifact::{Bounds, Package, PackageSpec, Provenance};
use boundml_core::{GraphBuilder, PlannerKind};
use boundml_data::{
    ExtraColumns, FeatureStats, LabelKind, MissingPolicy, Normalize, Preprocessor, Schema,
};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Bundle {
    format: String,
    version: u32,
    source: String,
    framework_version: String,
    name: String,
    features: Vec<String>,
    label: String,
    #[serde(default)]
    classes: Vec<String>,
    layers: Vec<Layer>,
    output_activation: String,
    #[serde(default)]
    preprocessing: Option<Standardization>,
    reference: Reference,
    #[serde(default = "default_tol")]
    atol: f32,
    #[serde(default = "default_rtol")]
    rtol: f32,
}

const MAX_ATOL: f32 = 1e-3;
const MAX_RTOL: f32 = 1e-2;
const MIN_REFERENCE_ROWS: usize = 4;

fn default_tol() -> f32 {
    1e-5
}
fn default_rtol() -> f32 {
    1e-4
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum Layer {
    Dense {
        weight: String,
        bias: Option<String>,
        /// `out_in` (PyTorch `nn.Linear`) or `in_out` (Keras `Dense`).
        weight_layout: String,
    },
    Activation {
        #[serde(rename = "fn")]
        func: String,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Standardization {
    mean: Vec<f64>,
    std: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reference {
    inputs: Vec<Vec<f64>>,
    outputs: Vec<Vec<f32>>,
}

pub struct ImportReport {
    pub vectors: usize,
    pub max_abs_error: f32,
}

fn bad(m: impl Into<String>) -> String {
    format!("BML-M001 import error: {}", m.into())
}

pub fn import(bundle_dir: &Path, out: &Path, max_batch: usize) -> Result<ImportReport, String> {
    let json =
        read_limited(&bundle_dir.join("bundle.json"), 16 << 20).map_err(|e| e.to_string())?;
    let b: Bundle = serde_json::from_slice(&json).map_err(|e| bad(format!("bundle.json: {e}")))?;
    if b.format != "boundml-import" || b.version != 1 {
        return Err(bad(format!(
            "unsupported bundle format {} v{}",
            b.format, b.version
        )));
    }
    // Loose tolerances would make the source-framework check meaningless.
    if !(0.0..=MAX_ATOL).contains(&b.atol) || !(0.0..=MAX_RTOL).contains(&b.rtol) {
        return Err(bad(format!(
            "tolerances atol={} rtol={} exceed the allowed maximum (atol <= {MAX_ATOL}, rtol <= {MAX_RTOL})",
            b.atol, b.rtol
        )));
    }
    if b.reference.inputs.len() < MIN_REFERENCE_ROWS {
        return Err(bad(format!(
            "at least {MIN_REFERENCE_ROWS} reference rows are required, got {}",
            b.reference.inputs.len()
        )));
    }
    let wbytes = read_limited(&bundle_dir.join("weights.safetensors"), 1 << 30)
        .map_err(|e| e.to_string())?;
    let tensors = decode_tensors(&wbytes, 4096).map_err(|e| e.to_string())?;
    let get = |name: &str| {
        tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .ok_or_else(|| bad(format!("tensor '{name}' not found in weights.safetensors")))
    };

    let nf = b.features.len();
    let mut g = GraphBuilder::new();
    let x = g.input("x", nf);
    let mut h = x;
    let mut width = nf;
    let mut weights: Vec<f32> = Vec::new();
    let mut dense_i = 0;
    for (li, layer) in b.layers.iter().enumerate() {
        match layer {
            Layer::Dense {
                weight,
                bias,
                weight_layout,
            } => {
                let (_, shape, w) = get(weight)?;
                let (fan_in, fan_out, w_in_out) = match weight_layout.as_str() {
                    "out_in" => (shape[1], shape[0], transpose(w, shape[0], shape[1])),
                    "in_out" => (shape[0], shape[1], w.clone()),
                    other => {
                        return Err(bad(format!("layer {li}: unknown weight_layout '{other}'")));
                    }
                };
                if fan_in != width {
                    return Err(bad(format!(
                        "layer {li}: expects {fan_in} inputs but receives {width}"
                    )));
                }
                let wn = g.param(&format!("dense{dense_i}.weight"), fan_in, fan_out, true);
                weights.extend_from_slice(&w_in_out);
                h = g.matmul(&format!("dense{dense_i}.matmul"), h, wn);
                if let Some(bname) = bias {
                    let (_, bshape, bv) = get(bname)?;
                    if *bshape != [1, fan_out] {
                        return Err(bad(format!(
                            "layer {li}: bias shape {bshape:?}, expected [1, {fan_out}]"
                        )));
                    }
                    let bn = g.param(&format!("dense{dense_i}.bias"), 1, fan_out, false);
                    weights.extend_from_slice(bv);
                    h = g.add(&format!("dense{dense_i}.pre"), h, bn);
                }
                width = fan_out;
                dense_i += 1;
            }
            Layer::Activation { func } => {
                let name = format!("act{li}");
                h = match func.as_str() {
                    "relu" => g.relu(&name, h),
                    "tanh" => g.tanh(&name, h),
                    "sigmoid" => g.sigmoid(&name, h),
                    other => {
                        return Err(bad(format!(
                            "layer {li}: unsupported activation '{other}' (supported: relu, tanh, sigmoid; output softmax via output_activation)"
                        )));
                    }
                };
            }
        }
    }
    let (label_kind, expect) = match (b.classes.len(), b.output_activation.as_str()) {
        (0, "none") => (LabelKind::Regression, 1),
        (2, "sigmoid") => (
            LabelKind::Classes {
                classes: b.classes.clone(),
            },
            1,
        ),
        (k, "softmax") if k > 2 => (
            LabelKind::Classes {
                classes: b.classes.clone(),
            },
            k,
        ),
        (k, act) => {
            return Err(bad(format!(
                "{k} classes with output_activation '{act}' is unsupported: use regression+none, 2 classes+sigmoid, or >2 classes+softmax"
            )));
        }
    };
    if width != expect {
        return Err(bad(format!(
            "final layer width {width} does not match the task's {expect} outputs"
        )));
    }
    let output = match b.output_activation.as_str() {
        "sigmoid" => g.sigmoid("output", h),
        "softmax" => g.softmax("output", h),
        _ => h,
    };
    let schema = Schema {
        features: b.features.clone(),
        label: b.label.clone(),
        label_kind,
        missing: MissingPolicy::Reject,
        extra_columns: ExtraColumns::Ignore,
    };
    let preprocessing = match &b.preprocessing {
        Some(s) if s.mean.len() == nf && s.std.len() == nf => Preprocessor {
            normalize: Normalize::Standardize,
            missing: MissingPolicy::Reject,
            features: b
                .features
                .iter()
                .enumerate()
                .map(|(i, f)| FeatureStats {
                    name: f.clone(),
                    mean: s.mean[i],
                    std: s.std[i],
                    count: 0,
                    missing: 0,
                    constant: false,
                })
                .collect(),
            rows_fitted: 0,
        },
        Some(_) => return Err(bad("preprocessing mean/std length differs from features")),
        None => Preprocessor {
            normalize: Normalize::None,
            missing: MissingPolicy::Reject,
            features: b
                .features
                .iter()
                .map(|f| FeatureStats {
                    name: f.clone(),
                    mean: 0.0,
                    std: 1.0,
                    count: 0,
                    missing: 0,
                    constant: false,
                })
                .collect(),
            rows_fitted: 0,
        },
    };
    if b.reference.inputs.is_empty() || b.reference.inputs.len() != b.reference.outputs.len() {
        return Err(bad(
            "reference inputs/outputs must be non-empty and the same length",
        ));
    }
    let spec = PackageSpec {
        name: b.name.clone(),
        schema,
        preprocessing,
        graph: g.into_def(output, None),
        weights,
        bounds: Bounds {
            max_batch,
            tensor_budget_bytes: None,
            sessions: 1,
            planner: PlannerKind::Reuse,
        },
        provenance: Provenance {
            source: format!("import:{}", b.source),
            notes: Some(format!("{} {}", b.source, b.framework_version)),
            ..Default::default()
        },
        test_inputs: b
            .reference
            .inputs
            .iter()
            .map(|r| r.iter().map(|v| Some(*v)).collect())
            .collect(),
        test_expected: Some(b.reference.outputs.clone()),
        test_origin: format!("source-framework:{}-{}", b.source, b.framework_version),
        test_atol: b.atol,
        test_rtol: b.rtol,
    };
    let pkg = Package::save(out, spec).map_err(|e| e.to_string())?;
    let report = pkg.verify().map_err(|e| e.to_string())?;
    Ok(ImportReport {
        vectors: report.vectors,
        max_abs_error: report.max_abs_error,
    })
}

fn transpose(w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0; w.len()];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = w[i * cols + j];
        }
    }
    t
}
