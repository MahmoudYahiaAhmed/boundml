//! Burn (ndarray CPU backend) training baseline for the same BoundML config.
//!
//! The setup follows `bench/py_train.py`: same layer widths and activation, AdamW
//! with the config's lr/betas/eps/weight decay, the same batch size and epochs, the
//! same train-only standardization, in-memory data with a seeded per-epoch shuffle,
//! and an output bias initialized from the training labels. The differences are
//! Burn-specific: its default Kaiming-uniform weight initialization, and AdamW weight
//! decay applied to biases too (Burn's AdamW has no per-parameter groups). Only the
//! training loop is timed.
//!
//! With a control file written by `train_core` (`train_burn <config> <test> <control.json>`),
//! the standardized data, the initial weights and biases, and every epoch's batch
//! order come from that file, so only the optimizer implementation and the bias-decay
//! policy (Burn decays biases too) differ from BoundML's in-memory trainer.
//!
//! Usage: `train_burn <config.toml> <test.csv> [<control.json>]`

use boundml_core::{OptimizerConfig, Rng, Task};
use boundml_data::{BatchReader, Normalize, Preprocessor};
use boundml_train::TrainConfig;
use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData, activation};
use serde_json::json;
use std::path::Path;
use std::time::Instant;

type Ad = Autodiff<NdArray<f32>>;

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    layers: Vec<Linear<B>>,
}

fn forward<B: Backend>(m: &Mlp<B>, x: Tensor<B, 2>, act: &str) -> Tensor<B, 2> {
    let n = m.layers.len();
    let mut h = x;
    for (i, l) in m.layers.iter().enumerate() {
        h = l.forward(h);
        if i + 1 < n {
            h = match act {
                "tanh" => activation::tanh(h),
                "sigmoid" => activation::sigmoid(h),
                _ => activation::relu(h),
            };
        }
    }
    h
}

fn load(path: &Path, cfg: &TrainConfig, pre: &Preprocessor) -> (Vec<f32>, Vec<f32>) {
    let schema = cfg.schema();
    let nf = schema.features.len();
    let mut r = BatchReader::open(path, &schema, pre, 1, Rng::seed(0), true).unwrap();
    let (mut x, mut y) = (Vec::new(), Vec::new());
    let (mut bx, mut by, mut bl) = (vec![0.0; 256 * nf], vec![0.0; 256], vec![0u64; 256]);
    loop {
        let n = r.next_batch(256, &mut bx, &mut by, &mut bl).unwrap();
        if n == 0 {
            break;
        }
        x.extend_from_slice(&bx[..n * nf]);
        y.extend_from_slice(&by[..n]);
    }
    (x, y)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let (cfg, base) = TrainConfig::load(Path::new(&a[1]))?;
    let test = Path::new(&a[2]);
    let schema = cfg.schema();
    let task = schema.task();
    let nf = schema.features.len();
    let train_path = base.join(&cfg.data.train);
    let pre = Preprocessor::fit(&train_path, &schema, Normalize::Standardize)?;
    let control: Option<serde_json::Value> = match a.get(3) {
        Some(p) => Some(serde_json::from_slice(&std::fs::read(p)?)?),
        None => None,
    };
    let floats = |v: &serde_json::Value| -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    };
    let ((xtr, ytr), (xte, yte)) = match &control {
        Some(c) => (
            (floats(&c["x_train"]), floats(&c["y_train"])),
            (floats(&c["x_test"]), floats(&c["y_test"])),
        ),
        None => (load(&train_path, &cfg, &pre), load(test, &cfg, &pre)),
    };
    let n = ytr.len();
    let out = task.output_cols();
    let device = Default::default();
    Ad::seed(&device, cfg.seed);
    let mut widths = vec![nf];
    widths.extend(&cfg.model.hidden);
    widths.push(out);
    let mut layers: Vec<Linear<Ad>> = widths
        .windows(2)
        .map(|w| LinearConfig::new(w[0], w[1]).init(&device))
        .collect();
    // Output bias from label statistics, as BoundML and the PyTorch baseline do.
    let mean = ytr.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
    let bias: Vec<f32> = match task {
        Task::Regression => vec![mean as f32],
        Task::Binary => {
            let p = mean.clamp(1e-3, 1.0 - 1e-3);
            vec![(p / (1.0 - p)).ln() as f32]
        }
        Task::Multiclass { classes } => (0..classes)
            .map(|c| {
                (ytr.iter().filter(|v| **v as usize == c).count() as f64 / n as f64)
                    .max(1e-3)
                    .ln() as f32
            })
            .collect(),
    };
    let last = layers.len() - 1;
    layers[last].bias = Some(Param::from_tensor(Tensor::from_data(
        TensorData::new(bias, [out]),
        &device,
    )));
    if let Some(c) = &control {
        for (i, l) in layers.iter_mut().enumerate() {
            let w = &c["init"][format!("dense{i}.weight")];
            let shape = [
                w["rows"].as_u64().unwrap() as usize,
                w["cols"].as_u64().unwrap() as usize,
            ];
            // Burn's Linear stores its weight as [d_input, d_output], the BoundML layout.
            l.weight = Param::from_tensor(Tensor::from_data(
                TensorData::new(floats(&w["values"]), shape),
                &device,
            ));
            let b = floats(&c["init"][format!("dense{i}.bias")]["values"]);
            let nb = b.len();
            l.bias = Some(Param::from_tensor(Tensor::from_data(
                TensorData::new(b, [nb]),
                &device,
            )));
        }
    }
    let mut model = Mlp { layers };
    let OptimizerConfig::AdamW {
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
    } = cfg.training.optimizer.clone()
    else {
        return Err("train_burn supports AdamW configs only".into());
    };
    let mut opt = AdamWConfig::new()
        .with_beta_1(beta1)
        .with_beta_2(beta2)
        .with_epsilon(eps)
        .with_weight_decay(weight_decay)
        .init::<Ad, Mlp<Ad>>();
    let act = format!("{:?}", cfg.model.activation).to_lowercase();
    let bs = cfg.training.batch_size;
    let mut rng = Rng::seed(cfg.seed);
    let mut idx: Vec<usize> = (0..n).collect();
    let perms: Vec<Vec<usize>> = match &control {
        Some(c) => c["perms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                p.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect()
            })
            .collect(),
        None => (0..cfg.training.epochs)
            .map(|_| {
                for i in (1..n).rev() {
                    idx.swap(i, rng.below(i as u64 + 1) as usize);
                }
                idx.clone()
            })
            .collect(),
    };
    let started = Instant::now();
    for perm in &perms {
        for chunk in perm.chunks(bs) {
            let m = chunk.len();
            let bx: Vec<f32> = chunk
                .iter()
                .flat_map(|&r| xtr[r * nf..(r + 1) * nf].iter().copied())
                .collect();
            let by: Vec<f32> = chunk.iter().map(|&r| ytr[r]).collect();
            let x = Tensor::<Ad, 2>::from_data(TensorData::new(bx, [m, nf]), &device);
            let z = forward(&model, x, &act);
            let loss = match task {
                Task::Regression => {
                    let y = Tensor::<Ad, 2>::from_data(TensorData::new(by, [m, 1]), &device);
                    (z - y).powf_scalar(2.0).mean()
                }
                Task::Binary => {
                    let y = Tensor::<Ad, 2>::from_data(TensorData::new(by, [m, 1]), &device);
                    // Stable BCE with logits: max(z,0) - z*y + log(1 + exp(-|z|)).
                    (z.clone().clamp_min(0.0) - z.clone() * y
                        + z.abs().neg().exp().add_scalar(1.0).log())
                    .mean()
                }
                Task::Multiclass { .. } => {
                    let targets = burn::tensor::Tensor::<Ad, 1, burn::tensor::Int>::from_data(
                        TensorData::new(by.iter().map(|v| *v as i64).collect::<Vec<_>>(), [m]),
                        &device,
                    );
                    burn::nn::loss::CrossEntropyLossConfig::new()
                        .init(&device)
                        .forward(z, targets)
                }
            };
            let grads = GradientsParams::from_grads::<Ad, Mlp<Ad>>(loss.backward(), &model);
            model = opt.step(lr as f64, model, grads);
        }
    }
    let secs = started.elapsed().as_secs_f64();
    let valid = <Mlp<Ad> as AutodiffModule<Ad>>::valid(&model);
    let m = yte.len();
    let x = Tensor::from_data(TensorData::new(xte, [m, nf]), &device);
    let z: Vec<f32> = forward(&valid, x, &act)
        .into_data()
        .to_vec()
        .map_err(|e| format!("{e:?}"))?;
    let metric = match task {
        Task::Regression => {
            json!({"rmse": (z.iter().zip(&yte).map(|(p, y)| ((p - y) as f64).powi(2)).sum::<f64>() / m as f64).sqrt()})
        }
        Task::Binary => {
            json!({"accuracy": z.iter().zip(&yte).filter(|(p, y)| (**p >= 0.0) == (**y == 1.0)).count() as f64 / m as f64})
        }
        Task::Multiclass { classes } => {
            let correct = (0..m)
                .filter(|&r| {
                    let row = &z[r * classes..(r + 1) * classes];
                    let best = (0..classes)
                        .max_by(|&i, &j| row[i].total_cmp(&row[j]))
                        .unwrap();
                    best == yte[r] as usize
                })
                .count();
            json!({"accuracy": correct as f64 / m as f64})
        }
    };
    println!(
        "{}",
        json!({"runtime": "burn-ndarray", "burn": "0.21.0", "threads": "ndarray default", "controlled": control.is_some(),
        "bias_weight_decay": "applied (Burn AdamW has no per-parameter groups)", "train_seconds": secs, "test": metric})
    );
    Ok(())
}
