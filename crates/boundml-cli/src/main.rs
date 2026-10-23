use boundml_artifact::CompareOptions;
use boundml_cli::commands::{self, CmdResult, PlanArgs};
use boundml_cli::{exit, exit_code_for};
use boundml_core::{Mode, PlannerKind};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// Train, package and embed small dense models with an enforceable contract.
#[derive(Parser)]
#[command(name = "boundml", version, about)]
struct Cli {
    /// Print machine-readable JSON on stdout instead of human-readable text.
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum ModeArg {
    Train,
    Infer,
}

#[derive(Clone, Copy, ValueEnum)]
enum PlannerArg {
    Reuse,
    Conservative,
}

#[derive(Subcommand)]
enum Cmd {
    /// Scaffold an example project (config + synthetic data) in DIR.
    Init { dir: PathBuf },
    /// Deterministically split a CSV into train/valid/test files.
    Split {
        csv: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 0.15)]
        valid: f64,
        #[arg(long, default_value_t = 0.15)]
        test: f64,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Validate a configuration, its data headers and its memory plans.
    Validate { config: PathBuf },
    /// Report the planned tensor memory for training or inference.
    Plan {
        config: PathBuf,
        #[arg(long, value_enum, default_value = "train")]
        mode: ModeArg,
        /// Override the configured budget, e.g. 64MiB.
        #[arg(long)]
        tensor_budget: Option<String>,
        /// Override batch size (training) or max batch (inference).
        #[arg(long)]
        batch: Option<usize>,
        #[arg(long, value_enum)]
        planner: Option<PlannerArg>,
        /// Override the number of inference sessions.
        #[arg(long)]
        sessions: Option<usize>,
    },
    /// Train a model and write a resumable checkpoint.
    Train {
        config: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
        /// Resume from the checkpoint (requires the identical config and data).
        #[arg(long)]
        resume: bool,
        /// Stop after this many total optimizer steps (checkpoint is saved).
        #[arg(long)]
        max_steps: Option<u64>,
        #[arg(long)]
        quiet: bool,
    },
    /// Evaluate a checkpoint or package on a labelled CSV.
    Evaluate {
        path: PathBuf,
        #[arg(long)]
        data: PathBuf,
    },
    /// Package a checkpoint for inference.
    Pack {
        checkpoint: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        name: Option<String>,
    },
    /// Show a package's contract, provenance and planned memory.
    Inspect { package: PathBuf },
    /// Check integrity, compatibility and bundled self-tests.
    Verify { package: PathBuf },
    /// Batch prediction from CSV to CSV.
    Predict {
        package: PathBuf,
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 256)]
        batch: usize,
    },
    /// Compare a candidate package with the current one before an update.
    Compare {
        current: PathBuf,
        candidate: PathBuf,
        #[arg(long)]
        data: Option<PathBuf>,
        /// Treat a larger max |output difference| as breaking.
        #[arg(long)]
        max_abs_diff: Option<f32>,
        #[arg(long)]
        max_rows: Option<u64>,
    },
    /// Serve a package over local HTTP.
    Serve {
        package: PathBuf,
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: String,
        /// Concurrent computations (default: the package's planned sessions).
        #[arg(long)]
        workers: Option<usize>,
        /// Requests allowed to wait beyond the workers; more are rejected with 503.
        #[arg(long, default_value_t = 16)]
        queue: usize,
        #[arg(long, default_value = "1MiB")]
        max_body: String,
        /// Rows per request (default: the package's max_batch).
        #[arg(long)]
        max_rows: Option<usize>,
        #[arg(long, default_value_t = 5000)]
        timeout_ms: u64,
    },
    /// Import a PyTorch/Keras export bundle (see migrate/) into a package.
    Import {
        bundle: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 256)]
        max_batch: usize,
    },
}

fn run(cmd: Cmd) -> CmdResult {
    match cmd {
        Cmd::Init { dir } => commands::init(&dir),
        Cmd::Split {
            csv,
            out,
            valid,
            test,
            seed,
        } => commands::split(&csv, &out, valid, test, seed),
        Cmd::Validate { config } => commands::validate(&config),
        Cmd::Plan {
            config,
            mode,
            tensor_budget,
            batch,
            planner,
            sessions,
        } => commands::plan(PlanArgs {
            config: &config,
            mode: match mode {
                ModeArg::Train => Mode::Training,
                ModeArg::Infer => Mode::Inference,
            },
            budget: tensor_budget.as_deref(),
            batch,
            planner: planner.map(|p| match p {
                PlannerArg::Reuse => PlannerKind::Reuse,
                PlannerArg::Conservative => PlannerKind::Conservative,
            }),
            sessions,
        }),
        Cmd::Train {
            config,
            checkpoint,
            resume,
            max_steps,
            quiet,
        } => commands::train(&config, &checkpoint, resume, max_steps, quiet),
        Cmd::Evaluate { path, data } => commands::evaluate(&path, &data),
        Cmd::Pack {
            checkpoint,
            out,
            name,
        } => commands::pack(&checkpoint, &out, name.as_deref()),
        Cmd::Inspect { package } => commands::inspect(&package),
        Cmd::Verify { package } => commands::verify(&package),
        Cmd::Predict {
            package,
            data,
            out,
            batch,
        } => commands::predict(&package, &data, &out, batch),
        Cmd::Compare {
            current,
            candidate,
            data,
            max_abs_diff,
            max_rows,
        } => commands::compare_cmd(
            &current,
            &candidate,
            CompareOptions {
                data,
                max_abs_diff,
                max_rows,
            },
        ),
        Cmd::Serve {
            package,
            bind,
            workers,
            queue,
            max_body,
            max_rows,
            timeout_ms,
        } => {
            let p = boundml_artifact::Package::load(&package).map_err(|e| e.to_string())?;
            let b = &p.manifest().bounds;
            let cfg = commands::serve_config(
                workers,
                queue,
                &max_body,
                max_rows.unwrap_or(b.max_batch),
                timeout_ms,
                b.sessions,
            )?;
            commands::serve(&package, &bind, cfg)
        }
        Cmd::Import {
            bundle,
            out,
            max_batch,
        } => commands::import(&bundle, &out, max_batch),
    }
}

fn main() {
    let cli = match Cli::try_parse() {
        Ok(c) => c,
        Err(e) => {
            let code = if e.use_stderr() {
                exit::USAGE
            } else {
                exit::OK
            };
            let _ = e.print();
            std::process::exit(code);
        }
    };
    let json = cli.json;
    let code = match run(cli.cmd) {
        Ok(o) => {
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &serde_json::json!({"ok": true, "result": o.json})
                    )
                    .unwrap_or_default()
                );
            } else {
                println!("{}", o.text.trim_end());
            }
            o.exit
        }
        Err(msg) => {
            let code = exit_code_for(&msg);
            if json {
                let c = boundml_cli::serve::code_of(&msg);
                println!(
                    "{}",
                    serde_json::json!({"ok": false, "error": {"code": c, "message": msg, "exit_code": code}})
                );
            } else {
                eprintln!("error: {msg}");
            }
            code
        }
    };
    std::process::exit(code);
}
