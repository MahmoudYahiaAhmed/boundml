//! Server limits: overload, timeout, malformed input, bounded work, shutdown.

use boundml_artifact::{Bounds, Package, PackageSpec, Provenance};
use boundml_cli::serve::{AppState, ServeConfig, run};
use boundml_core::{Activation, DenseSpec, PlannerKind, Rng};
use boundml_data::*;
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

fn package(dir: &std::path::Path, sessions: usize) -> Arc<Package> {
    let schema = Schema {
        features: vec!["a".into(), "b".into()],
        label: "y".into(),
        label_kind: LabelKind::Classes {
            classes: vec!["n".into(), "p".into()],
        },
        missing: MissingPolicy::Reject,
        extra_columns: ExtraColumns::Reject,
    };
    let spec = DenseSpec {
        inputs: 2,
        hidden: vec![4],
        activation: Activation::Relu,
        task: schema.task(),
    };
    let g = spec.graph().unwrap();
    let w = spec.init_params(&g, &mut Rng::seed(1));
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
                graph: g.def().clone(),
                weights: w,
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
    fn shutdown(&mut self) {
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
    write!(s, "{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    let mut resp = String::new();
    s.read_to_string(&mut resp).unwrap();
    (
        resp[9..12].parse().unwrap(),
        resp.split("\r\n\r\n").nth(1).unwrap_or("").to_string(),
    )
}

const ONE: &str = r#"{"rows":[{"a":1.0,"b":2.0}]}"#;

#[test]
fn malformed_and_policy_violations() {
    let d = tempfile::tempdir().unwrap();
    let mut s = start(
        package(d.path(), 2),
        ServeConfig {
            workers: 2,
            max_rows: 8,
            ..Default::default()
        },
    );
    let a = s.addr;
    assert_eq!(http(a, "GET", "/healthz", "").0, 200);
    assert_eq!(http(a, "GET", "/readyz", "").0, 200);
    let (c, b) = http(a, "POST", "/v1/predict", ONE);
    assert_eq!(c, 200, "{b}");
    assert!(b.contains("\"predicted\""));
    for (body, code, needle) in [
        ("{", 400, "BML-S002"),
        (r#"{"rows":[]}"#, 422, "BML-C004"),
        (r#"{"rows":[{"a":1}]}"#, 422, "missing"),
        (r#"{"rows":[{"a":1,"b":2,"zzz":3}]}"#, 422, "unknown field"),
        (r#"{"rows":[{"a":"x","b":2}]}"#, 422, "number or null"),
        (r#"{"rows":[{"a":1,"b":2}],"extra":1}"#, 400, "BML-S002"),
        (r#"{"rows":[{"a":1e400,"b":2}]}"#, 400, "BML-S002"),
    ] {
        let (c, b) = http(a, "POST", "/v1/predict", body);
        assert_eq!(c, code, "{body}: {b}");
        assert!(b.contains(needle), "{body}: {b}");
    }
    let nine = format!(r#"{{"rows":[{}]}}"#, vec![r#"{"a":1,"b":2}"#; 9].join(","));
    assert_eq!(http(a, "POST", "/v1/predict", &nine).0, 422);
    assert_eq!(http(a, "GET", "/nope", "").0, 404);
    s.shutdown();
}

#[test]
fn workers_cannot_exceed_planned_sessions() {
    let d = tempfile::tempdir().unwrap();
    assert!(
        AppState::new(
            package(d.path(), 1),
            ServeConfig {
                workers: 2,
                max_rows: 8,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn overload_is_rejected_not_queued_without_bound() {
    let d = tempfile::tempdir().unwrap();
    let cfg = ServeConfig {
        workers: 1,
        queue: 1,
        max_rows: 8,
        simulate_compute_delay: Duration::from_millis(400),
        ..Default::default()
    };
    let mut s = start(package(d.path(), 1), cfg);
    let a = s.addr;
    let handles: Vec<_> = (0..6)
        .map(|_| std::thread::spawn(move || http(a, "POST", "/v1/predict", ONE).0))
        .collect();
    let codes: Vec<u16> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let ok = codes.iter().filter(|&&c| c == 200).count();
    let rejected = codes.iter().filter(|&&c| c == 503).count();
    assert!(
        ok >= 1 && ok <= 2,
        "at most workers+queue admitted: {codes:?}"
    );
    assert_eq!(ok + rejected, 6, "{codes:?}");
    assert_eq!(
        s.state.rejected_overload.load(Ordering::Relaxed) as usize,
        rejected
    );
    s.shutdown();
}

#[test]
fn timeout_returns_408_and_work_stays_bounded() {
    let d = tempfile::tempdir().unwrap();
    let cfg = ServeConfig {
        workers: 1,
        queue: 4,
        max_rows: 8,
        timeout: Duration::from_millis(150),
        simulate_compute_delay: Duration::from_millis(500),
        ..Default::default()
    };
    let mut s = start(package(d.path(), 1), cfg);
    let a = s.addr;
    let t = Instant::now();
    let handles: Vec<_> = (0..3)
        .map(|_| std::thread::spawn(move || http(a, "POST", "/v1/predict", ONE).0))
        .collect();
    let codes: Vec<u16> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert!(codes.iter().all(|&c| c == 408), "{codes:?}");
    assert!(
        t.elapsed() < Duration::from_millis(450),
        "clients get timeouts promptly"
    );
    // Only the request that had already started computing runs; the queued ones were
    // cancelled before computing. Wait for it to finish, then check served == 0
    // (timed-out results are discarded) and the worker is free again.
    std::thread::sleep(Duration::from_millis(700));
    assert_eq!(s.state.served.load(Ordering::Relaxed), 0);
    s.shutdown();
}

#[test]
fn graceful_shutdown_drains_in_flight_requests() {
    let d = tempfile::tempdir().unwrap();
    let cfg = ServeConfig {
        workers: 1,
        queue: 0,
        max_rows: 8,
        simulate_compute_delay: Duration::from_millis(300),
        ..Default::default()
    };
    let mut s = start(package(d.path(), 1), cfg);
    let a = s.addr;
    let inflight = std::thread::spawn(move || http(a, "POST", "/v1/predict", ONE).0);
    std::thread::sleep(Duration::from_millis(100));
    s.shutdown();
    assert_eq!(
        inflight.join().unwrap(),
        200,
        "in-flight request completed during drain"
    );
    assert!(
        std::net::TcpStream::connect(a).is_err(),
        "listener closed after shutdown"
    );
}
