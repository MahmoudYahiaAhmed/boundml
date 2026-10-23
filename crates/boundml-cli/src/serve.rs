//! Bounded local HTTP inference service.
//!
//! Every request uses the same validated path as the SDK: `Predictor::fill_named`
//! (name mapping + policies) then `Predictor::predict_raw` (fitted preprocessing +
//! planned execution). Resource bounds:
//!
//! * request body <= `max_body_bytes` (413), rows per request <= `max_rows` (422);
//! * at most `workers` computations run at once, each on its own preallocated
//!   predictor; at most `queue` further requests may wait. Anything beyond is
//!   rejected immediately with 503 (no unbounded queue);
//! * each request (including body read and queue wait) has a deadline (408);
//! * a request whose client disconnects or times out while waiting never starts
//!   computing; one that already started finishes its bounded batch and returns
//!   its predictor to the pool, so CPU work per admitted request is bounded;
//! * SIGINT/Ctrl-C stops accepting connections and drains in-flight requests.
//!
//! The service binds to localhost by default and has no authentication or TLS.
//! Put it behind a reverse proxy that provides those before exposing it.

use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use boundml_artifact::{Package, Predictor};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Semaphore;
use tower_http::timeout::TimeoutLayer;

#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub workers: usize,
    pub queue: usize,
    pub max_body_bytes: usize,
    pub max_rows: usize,
    pub timeout: Duration,
    /// Test hook: extra sleep inside the compute section (default zero).
    #[doc(hidden)]
    pub simulate_compute_delay: Duration,
}

impl Default for ServeConfig {
    fn default() -> Self {
        ServeConfig {
            workers: 2,
            queue: 16,
            max_body_bytes: 1 << 20,
            max_rows: 256,
            timeout: Duration::from_secs(5),
            simulate_compute_delay: Duration::ZERO,
        }
    }
}

pub struct AppState {
    pkg: Arc<Package>,
    pool: Mutex<Vec<Predictor>>,
    compute: Arc<Semaphore>,
    admission: Semaphore,
    cfg: ServeConfig,
    shutting_down: AtomicBool,
    pub served: AtomicU64,
    pub rejected_overload: AtomicU64,
}

impl AppState {
    /// Allocates one predictor (arena) per worker up front.
    pub fn new(pkg: Arc<Package>, cfg: ServeConfig) -> Result<Arc<AppState>, String> {
        if cfg.workers == 0 || cfg.workers > 256 || cfg.queue > 65_536 || cfg.max_rows == 0 {
            return Err("workers must be 1..=256, queue <= 65536, max_rows >= 1".into());
        }
        let bound = pkg.manifest().bounds.max_batch;
        if cfg.max_rows > bound {
            return Err(format!(
                "max_rows {} exceeds the package max_batch {bound}",
                cfg.max_rows
            ));
        }
        if cfg.workers > pkg.manifest().bounds.sessions {
            return Err(format!(
                "{} workers need {} sessions but the package plans for {} (deploy.sessions); its tensor budget covers only that many",
                cfg.workers,
                cfg.workers,
                pkg.manifest().bounds.sessions
            ));
        }
        let pool = (0..cfg.workers)
            .map(|_| pkg.predictor(cfg.max_rows).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Arc::new(AppState {
            compute: Arc::new(Semaphore::new(cfg.workers)),
            admission: Semaphore::new(cfg.workers + cfg.queue),
            pool: Mutex::new(pool),
            pkg,
            cfg,
            shutting_down: AtomicBool::new(false),
            served: AtomicU64::new(0),
            rejected_overload: AtomicU64::new(0),
        }))
    }

    pub fn begin_shutdown(&self) {
        self.shutting_down.store(true, Ordering::SeqCst);
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    let timeout = state.cfg.timeout;
    let max_body = state.cfg.max_body_bytes;
    Router::new()
        .route("/healthz", get(health))
        .route("/readyz", get(ready))
        .route("/v1/model", get(model_info))
        .route(
            "/v1/predict",
            post(predict).route_layer(axum::middleware::from_fn_with_state(
                Arc::clone(&state),
                admit,
            )),
        )
        .layer(DefaultBodyLimit::max(max_body))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            timeout,
        ))
        .with_state(state)
}

fn error(status: StatusCode, code: &str, message: impl Into<String>) -> Response {
    let mut r = (
        status,
        Json(json!({"error": {"code": code, "message": message.into()}})),
    )
        .into_response();
    if status == StatusCode::SERVICE_UNAVAILABLE {
        r.headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
    }
    r
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn ready(State(s): State<Arc<AppState>>) -> Response {
    if s.shutting_down.load(Ordering::SeqCst) {
        return error(StatusCode::SERVICE_UNAVAILABLE, "BML-S003", "shutting down");
    }
    if s.admission.available_permits() == 0 {
        return error(StatusCode::SERVICE_UNAVAILABLE, "BML-S001", "at capacity");
    }
    Json(json!({"status": "ready"})).into_response()
}

async fn model_info(State(s): State<Arc<AppState>>) -> Json<Value> {
    let m = s.pkg.manifest();
    Json(json!({
        "name": m.name,
        "task": m.task,
        "features": m.schema.features,
        "outputs": m.output_names(),
        "missing": m.schema.missing,
        "max_rows_per_request": s.cfg.max_rows,
    }))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PredictRequest {
    rows: Vec<Row>,
}

/// Maximum fields accepted in one row object.
const MAX_FIELDS_PER_ROW: usize = 4096;

/// A JSON object kept as an ordered list of pairs so that duplicate keys reach
/// `Predictor::fill_named` (which rejects them) instead of being silently merged.
struct Row(Vec<(String, Value)>);

impl<'de> Deserialize<'de> for Row {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Row, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = Row;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("an object mapping feature names to numbers or null")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut m: A) -> Result<Row, A::Error> {
                let mut pairs = Vec::new();
                while let Some((k, v)) = m.next_entry::<String, Value>()? {
                    if pairs.len() >= MAX_FIELDS_PER_ROW {
                        return Err(serde::de::Error::custom(format!(
                            "more than {MAX_FIELDS_PER_ROW} fields in a row"
                        )));
                    }
                    pairs.push((k, v));
                }
                Ok(Row(pairs))
            }
        }
        d.deserialize_map(V)
    }
}

#[derive(Serialize)]
struct PredictResponse {
    model: String,
    output_names: Vec<String>,
    outputs: Vec<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    predicted: Option<Vec<String>>,
}

/// Admission control runs *before* the request body is read, so at most
/// `workers + queue` bodies (each <= `max_body_bytes`) are buffered at any time.
/// Excess requests get 503 without their bodies being read.
async fn admit(
    State(s): State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    if s.shutting_down.load(Ordering::SeqCst) {
        return error(StatusCode::SERVICE_UNAVAILABLE, "BML-S003", "shutting down");
    }
    let Ok(_admitted) = s.admission.try_acquire() else {
        s.rejected_overload.fetch_add(1, Ordering::Relaxed);
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "BML-S001",
            "server at capacity; retry later",
        );
    };
    next.run(req).await
}

async fn predict(State(s): State<Arc<AppState>>, body: axum::body::Bytes) -> Response {
    let req: PredictRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return error(
                StatusCode::BAD_REQUEST,
                "BML-S002",
                format!("invalid JSON request: {e}"),
            );
        }
    };
    let n = req.rows.len();
    if n == 0 || n > s.cfg.max_rows {
        return error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "BML-C004",
            format!("rows must contain 1..={} objects, got {n}", s.cfg.max_rows),
        );
    }
    // Waiting here is cancelled (no computation) if the request times out or the
    // client disconnects, because this future is dropped.
    let Ok(permit) = Arc::clone(&s.compute).acquire_owned().await else {
        return error(StatusCode::SERVICE_UNAVAILABLE, "BML-S003", "shutting down");
    };
    let st = Arc::clone(&s);
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut predictor = st
            .pool
            .lock()
            .map_err(|_| "predictor pool poisoned".to_string())?
            .pop()
            .ok_or_else(|| "no idle predictor".to_string())?;
        let out = run_rows(&st, &mut predictor, &req.rows);
        if let Ok(mut pool) = st.pool.lock() {
            pool.push(predictor);
        }
        Ok::<_, String>(out)
    })
    .await;
    match result {
        Ok(Ok(Ok(resp))) => {
            s.served.fetch_add(1, Ordering::Relaxed);
            Json(resp).into_response()
        }
        Ok(Ok(Err((status, code, msg)))) => error(status, &code, msg),
        Ok(Err(msg)) => error(StatusCode::INTERNAL_SERVER_ERROR, "BML-S004", msg),
        Err(_) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "BML-S004",
            "worker failed",
        ),
    }
}

type RowError = (StatusCode, String, String);

fn run_rows(s: &AppState, p: &mut Predictor, rows: &[Row]) -> Result<PredictResponse, RowError> {
    let nf = p.n_features();
    let no = p.output_cols();
    let n = rows.len();
    let mut raw = vec![None; n * nf];
    for (i, obj) in rows.iter().enumerate() {
        let mut pairs = Vec::with_capacity(obj.0.len());
        for (k, v) in &obj.0 {
            let val = match v {
                Value::Null => None,
                Value::Number(x) => x.as_f64(),
                _ => {
                    return Err((
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "BML-A008".into(),
                        format!(
                            "row {i}: field '{}' must be a number or null",
                            k.chars().take(64).collect::<String>()
                        ),
                    ));
                }
            };
            pairs.push((k.as_str(), val));
        }
        p.fill_named(pairs, &mut raw[i * nf..(i + 1) * nf], i)
            .map_err(|e| {
                (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    code_of(&e.to_string()),
                    e.to_string(),
                )
            })?;
    }
    if !s.cfg.simulate_compute_delay.is_zero() {
        std::thread::sleep(s.cfg.simulate_compute_delay);
    }
    let mut out = vec![0.0f32; n * no];
    p.predict_raw(&raw, n, &mut out).map_err(|e| {
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            code_of(&e.to_string()),
            e.to_string(),
        )
    })?;
    let m = s.pkg.manifest();
    let outputs: Vec<Vec<f32>> = out.chunks(no).map(<[f32]>::to_vec).collect();
    let predicted = m.schema.classes().map(|classes| {
        outputs
            .iter()
            .map(|o| classes[boundml_artifact::compare::decision(&m.task, o)].clone())
            .collect()
    });
    Ok(PredictResponse {
        model: m.name.clone(),
        output_names: m.output_names(),
        outputs,
        predicted,
    })
}

pub fn code_of(message: &str) -> String {
    message
        .split_whitespace()
        .next()
        .filter(|w| w.starts_with("BML-"))
        .unwrap_or("BML-E000")
        .to_string()
}

/// Run until `shutdown` resolves, then drain in-flight requests.
pub async fn run(
    state: Arc<AppState>,
    listener: tokio::net::TcpListener,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    let st = Arc::clone(&state);
    let app = router(state);
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown.await;
            st.begin_shutdown();
        })
        .await
}
