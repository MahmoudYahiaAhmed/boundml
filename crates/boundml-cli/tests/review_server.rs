//! Server recovery: a timed-out computation keeps its slot until it really ends and
//! the same server then serves again; a panic in the compute section does not lose
//! capacity silently.

use boundml_artifact::{Bounds, Package, PackageSpec, Provenance};
use boundml_cli::serve::{AppState, ServeConfig, TestHooks, run};
use boundml_core::{Activation, DenseSpec, PlannerKind, Task};
use boundml_data::*;
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

fn package(dir: &std::path::Path, sessions: usize) -> Arc<Package> {
    let schema = Schema {
        features: vec!["a".into(), "b".into()],
        label: "y".into(),
        label_kind: LabelKind::Regression,
        missing: MissingPolicy::Reject,
        extra_columns: ExtraColumns::Ignore,
    };
    let dense = DenseSpec {
        inputs: 2,
        hidden: vec![],
        activation: Activation::Relu,
        task: Task::Regression,
    };
    let pre = Preprocessor {
        normalize: Normalize::None,
        missing: MissingPolicy::Reject,
        features: ["a", "b"]
            .iter()
            .map(|n| FeatureStats {
                name: n.to_string(),
                mean: 0.0,
                std: 1.0,
                count: 1,
                missing: 0,
                constant: false,
            })
            .collect(),
        rows_fitted: 1,
    };
    Arc::new(
        Package::save(
            &dir.join("p"),
            PackageSpec {
                name: "srv".into(),
                schema,
                preprocessing: pre,
                graph: dense.graph().unwrap().def().clone(),
                weights: vec![1.0, 1.0, 0.0],
                bounds: Bounds {
                    max_batch: 8,
                    tensor_budget_bytes: None,
                    sessions,
                    planner: PlannerKind::Reuse,
                },
                provenance: Provenance::default(),
                test_inputs: vec![],
                test_expected: None,
                test_origin: String::new(),
                test_atol: 0.0,
                test_rtol: 0.0,
            },
        )
        .unwrap(),
    )
}

struct Server {
    rt: tokio::runtime::Runtime,
    addr: SocketAddr,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<std::io::Result<()>>>,
    state: Arc<AppState>,
}

fn start(pkg: Arc<Package>, cfg: ServeConfig) -> Server {
    let state = AppState::new(pkg, cfg).unwrap();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let listener = rt
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let handle = rt.spawn(run(Arc::clone(&state), listener, async {
        let _ = rx.await;
    }));
    Server {
        rt,
        addr,
        stop: Some(tx),
        handle: Some(handle),
        state,
    }
}

impl Server {
    fn shutdown(mut self) {
        let _ = self.stop.take().unwrap().send(());
        self.rt
            .block_on(self.handle.take().unwrap())
            .unwrap()
            .unwrap();
    }
}

fn http(addr: SocketAddr, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut resp = String::new();
    s.read_to_string(&mut resp).unwrap();
    (
        resp[9..12].parse().unwrap(),
        resp.split("\r\n\r\n").nth(1).unwrap_or("").to_string(),
    )
}

const ONE: &str = r#"{"rows":[{"a":1.0,"b":2.0}]}"#;

fn cfg(hooks: &Arc<TestHooks>) -> ServeConfig {
    ServeConfig {
        workers: 1,
        queue: 1,
        max_rows: 8,
        timeout: Duration::from_millis(300),
        hooks: Some(Arc::clone(hooks)),
        ..Default::default()
    }
}

#[test]
fn compute_timeout_keeps_the_slot_until_done_then_the_server_serves_again() {
    let d = tempfile::tempdir().unwrap();
    let hooks = Arc::new(TestHooks::default());
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    *hooks.hold_next.lock().unwrap() = Some((entered_tx, release_rx));
    let s = start(package(d.path(), 1), cfg(&hooks));
    let a = s.addr;
    let first = std::thread::spawn(move || http(a, "POST", "/v1/predict", ONE).0);
    entered_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("first request entered compute");
    assert_eq!(first.join().unwrap(), 408, "the client gets its timeout");
    // The held computation still owns the only compute slot: a second request cannot
    // run concurrently with it, so it times out waiting instead of being served.
    assert_ne!(http(a, "POST", "/v1/predict", ONE).0, 200);
    // Let the timed-out computation finish; its result is discarded.
    release_tx.send(()).unwrap();
    // Same server, same single worker: a valid request now succeeds.
    let (code, body) = http(a, "POST", "/v1/predict", ONE);
    assert_eq!(code, 200, "{body}");
    assert!(body.contains("[3.0]"), "{body}");
    assert_eq!(http(a, "GET", "/readyz", "").0, 200);
    assert_eq!(
        s.state.served.load(Ordering::Relaxed),
        1,
        "timed-out results are not counted as served"
    );
    s.shutdown();
}

#[test]
fn a_panicking_computation_is_replaced_and_capacity_is_kept() {
    let d = tempfile::tempdir().unwrap();
    let hooks = Arc::new(TestHooks::default());
    hooks.panic_next.store(true, Ordering::SeqCst);
    let s = start(package(d.path(), 1), cfg(&hooks));
    let (code, body) = http(s.addr, "POST", "/v1/predict", ONE);
    assert_eq!(code, 500, "{body}");
    assert!(body.contains("BML-S004"));
    let (code, body) = http(s.addr, "GET", "/readyz", "");
    assert_eq!(code, 200, "{body}");
    assert!(body.contains("\"usable_workers\":1"), "{body}");
    for _ in 0..3 {
        assert_eq!(http(s.addr, "POST", "/v1/predict", ONE).0, 200);
    }
    assert_eq!(s.state.replaced_after_panic.load(Ordering::Relaxed), 1);
    s.shutdown();
}

#[test]
fn an_unreplaceable_predictor_fails_readiness_explicitly() {
    let d = tempfile::tempdir().unwrap();
    let hooks = Arc::new(TestHooks::default());
    hooks.panic_next.store(true, Ordering::SeqCst);
    hooks.fail_next_replacement.store(true, Ordering::SeqCst);
    let s = start(package(d.path(), 1), cfg(&hooks));
    assert_eq!(http(s.addr, "POST", "/v1/predict", ONE).0, 500);
    let (code, body) = http(s.addr, "GET", "/readyz", "");
    assert_eq!(code, 503, "{body}");
    assert!(body.contains("no usable workers"), "{body}");
    // No request is admitted to a slot that has no predictor: it times out waiting
    // rather than failing with "no idle predictor".
    let (code, body) = http(s.addr, "POST", "/v1/predict", ONE);
    assert_eq!(code, 408, "{body}");
    s.shutdown();
}
