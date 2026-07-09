//! Phased Array Simulator — Axum HTTP API Server
//!
//! GPU-accelerated beamforming with CUDA + multi-core CPU fallback.

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

use phased_array_rs::{
    array::{generate_positions, ArrayConfig},
    beamforming::{self, apply_taper, find_sidelobe, steering_vector, BeamConfig},
    gpu,
    physics::{self, beam_squint_analysis, mutual_coupling_matrix, quantize_phase, simulate_failures},
    super_res::{self, esprit_ula, generate_snapshots, music_ula},
};

// ═══ App State ══════════════════════════════════════════

struct AppState {
    cuda_available: bool,
}

// ═══ Request Models ═════════════════════════════════════

#[derive(Deserialize)]
struct PatternRequest {
    array: ArrayConfig,
    beam: BeamConfig,
    #[serde(default = "default_resolution")]
    resolution: usize,
    #[serde(default)]
    pattern_type: String,
}

fn default_resolution() -> usize { 91 }

#[derive(Deserialize)]
struct DOARequest {
    n_elements: usize,
    n_snapshots: usize,
    source_angles_deg: Vec<f64>,
    snr_db: f64,
}

#[derive(Deserialize)]
struct RealisticRequest {
    array: ArrayConfig,
    beam: BeamConfig,
    #[serde(default = "default_resolution")]
    resolution: usize,
    #[serde(default)]
    element_q: f64,
    #[serde(default)]
    coupling_strength: f64,
    #[serde(default)]
    phase_bits: u32,
    #[serde(default)]
    failure_rate: f64,
}

#[derive(Deserialize)]
struct SquintRequest {
    n_elements: usize,
    spacing: f64,
    theta_steer_deg: f64,
    f0: f64,
    bandwidth_ghz: f64,
}

// ═══ Main ════════════════════════════════════════════════

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let cuda_available = gpu::init_cuda();
    tracing::info!("PTX kernels embedded: {}", gpu::PTX_BYTES.len());
    tracing::info!("CUDA: {}, Rayon threads: {}", cuda_available, rayon::current_num_threads());

    let state = Arc::new(AppState { cuda_available });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/health", get(health))
        .route("/api/geometries", get(list_geometries))
        .route("/api/compute-pattern", post(compute_pattern))
        .route("/api/element-positions", post(element_positions))
        .route("/api/music", post(music_doa))
        .route("/api/esprit", post(esprit_doa))
        .route("/api/mvdr", post(mvdr_beamform))
        .route("/api/pattern-realistic", post(pattern_realistic))
        .route("/api/beam-squint", post(beam_squint))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    tracing::info!("Phased Array Simulator (Rust) listening on :8000");
    axum::serve(listener, app).await.unwrap();
}

// ═══ Handlers ════════════════════════════════════════════

async fn health(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "status": "healthy",
        "version": "0.1.0",
        "backend": "rust",
        "cuda_available": state.cuda_available,
    }))
}

async fn list_geometries() -> Json<Value> {
    Json(json!({
        "geometries": [
            {"id": "rectangular", "name": "Rectangular Grid"},
            {"id": "circular", "name": "Circular Array"},
            {"id": "triangular", "name": "Triangular Grid"},
        ]
    }))
}

async fn compute_pattern(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PatternRequest>,
) -> Result<Json<Value>, AppError> {
    let (x, y) = generate_positions(&req.array);
    let n_elem = x.len();

    // Taper
    let taper_w = apply_taper(
        &x, &y, req.array.nx, req.array.ny,
        &req.beam.taper, req.beam.sidelobe_db,
        &format!("{:?}", req.array.geometry).to_lowercase(),
    );

    // Steering
    let steer = steering_vector(&x, &y, req.beam.theta_deg, req.beam.phi_deg);

    // Combined weights
    let weights: Vec<_> = taper_w.iter().zip(steer.iter())
        .map(|(&t, &s)| s * t)
        .collect();

    // Angular grid
    let n_theta = req.resolution * 2;
    let n_phi = req.resolution * 2;
    let theta: Vec<f64> = (0..n_theta).map(|i| std::f64::consts::PI * i as f64 / (n_theta - 1) as f64).collect();
    let phi: Vec<f64> = (0..n_phi).map(|i| 2.0 * std::f64::consts::PI * i as f64 / (n_phi - 1) as f64).collect();

    // Compute AF via GPU (with CPU fallback)
    let sin_theta: Vec<f64> = theta.iter().map(|t| t.sin()).collect();
    let cos_phi: Vec<f64> = phi.iter().map(|p| p.cos()).collect();
    let sin_phi: Vec<f64> = phi.iter().map(|p| p.sin()).collect();
    let w_real: Vec<f64> = weights.iter().map(|c| c.re).collect();
    let w_imag: Vec<f64> = weights.iter().map(|c| c.im).collect();

    let flat_pattern = gpu::compute_array_factor(
        x.as_slice().unwrap(), y.as_slice().unwrap(),
        &w_real, &w_imag,
        &sin_theta, &cos_phi, &sin_phi,
    );
    let max_db = flat_pattern.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let flat_norm: Vec<f64> = flat_pattern.iter().map(|v| v - max_db).collect();
    let pattern_db = ndarray::Array2::from_shape_vec((n_theta, n_phi), flat_norm).unwrap();

    // Beam cut via GPU
    let theta_cut: Vec<f64> = (-90..=90).map(|d| d as f64).collect();
    let cut_db = gpu::compute_beam_cut(
        x.as_slice().unwrap(), y.as_slice().unwrap(),
        &w_real, &w_imag,
        &theta_cut, req.beam.phi_deg,
    );
    let cut_max = cut_db.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let cut_db: Vec<f64> = cut_db.iter().map(|v| v - cut_max).collect();

    let beam_idx = (req.beam.theta_deg + 90.0) as usize;
    let sll = find_sidelobe(&cut_db, beam_idx);
    let bw = 51.0 / (req.array.nx as f64 * req.array.dx);

    // Convert pattern to 2D vec
    let pattern_2d: Vec<Vec<f64>> = (0..n_theta)
        .map(|i| pattern_db.row(i).to_vec())
        .collect();

    Ok(Json(json!({
        "type": req.pattern_type,
        "metadata": {
            "geometry": format!("{:?}", req.array.geometry).to_lowercase(),
            "n_elements": n_elem,
            "nx": req.array.nx, "ny": req.array.ny,
            "spacing": {"dx": req.array.dx, "dy": req.array.dy},
            "beam_direction": {"theta_deg": req.beam.theta_deg, "phi_deg": req.beam.phi_deg},
            "taper": format!("{:?}", req.beam.taper).to_lowercase(),
            "main_lobe_db": 0.0,
            "first_sidelobe_db": sll,
            "estimated_beamwidth_deg": bw,
            "backend": if state.cuda_available { "cuda" } else { "cpu_rayon" },
        },
        "theta_deg": theta.iter().map(|t| t.to_degrees()).collect::<Vec<_>>(),
        "phi_deg": phi.iter().map(|p| p.to_degrees()).collect::<Vec<_>>(),
        "pattern_db": pattern_2d,
        "cut_through_beam": {
            "theta_deg": theta_cut,
            "pattern_db": cut_db,
        },
    })))
}

async fn element_positions(Json(cfg): Json<ArrayConfig>) -> Json<Value> {
    let (x, y) = generate_positions(&cfg);
    let positions: Vec<Value> = (0..x.len())
        .map(|i| json!({"x": x[i], "y": y[i], "z": 0.0}))
        .collect();
    Json(json!({
        "n_elements": x.len(),
        "positions": positions,
    }))
}

async fn music_doa(Json(req): Json<DOARequest>) -> Result<Json<Value>, AppError> {
    let snapshots = generate_snapshots(
        req.n_elements, req.n_snapshots, &req.source_angles_deg, req.snr_db, 42,
    );
    let (theta_deg, spectrum_db) = music_ula(
        req.n_elements, &snapshots, req.source_angles_deg.len(), 360,
    );
    Ok(Json(json!({
        "theta_deg": theta_deg,
        "spectrum_db": spectrum_db,
        "true_angles": req.source_angles_deg,
        "n_elements": req.n_elements,
        "n_snapshots": req.n_snapshots,
        "snr_db": req.snr_db,
    })))
}

async fn esprit_doa(Json(req): Json<DOARequest>) -> Result<Json<Value>, AppError> {
    let snapshots = generate_snapshots(
        req.n_elements, req.n_snapshots, &req.source_angles_deg, req.snr_db, 42,
    );
    let estimated = esprit_ula(req.n_elements, &snapshots, req.source_angles_deg.len());
    Ok(Json(json!({
        "estimated_angles_deg": estimated,
        "true_angles_deg": req.source_angles_deg,
        "n_elements": req.n_elements,
        "n_snapshots": req.n_snapshots,
        "snr_db": req.snr_db,
    })))
}

async fn mvdr_beamform(Json(req): Json<DOARequest>) -> Result<Json<Value>, AppError> {
    let snapshots = generate_snapshots(
        req.n_elements, req.n_snapshots, &req.source_angles_deg, req.snr_db, 42,
    );
    let x: Vec<f64> = (0..req.n_elements).map(|i| (i as f64 - (req.n_elements - 1) as f64 / 2.0) * 0.5).collect();
    let y = vec![0.0; req.n_elements];
    let w = super_res::mvdr_weights(&x, &y, &snapshots, 0.0, 1e-4);
    Ok(Json(json!({
        "weights": w.iter().map(|c| [c.re, c.im]).collect::<Vec<_>>(),
        "n_elements": req.n_elements,
    })))
}

async fn pattern_realistic(Json(req): Json<RealisticRequest>) -> Result<Json<Value>, AppError> {
    let (x, y) = generate_positions(&req.array);
    let n_elem = x.len();

    let taper_w = apply_taper(
        &x, &y, req.array.nx, req.array.ny,
        &req.beam.taper, req.beam.sidelobe_db,
        &format!("{:?}", req.array.geometry).to_lowercase(),
    );
    let steer = steering_vector(&x, &y, req.beam.theta_deg, req.beam.phi_deg);
    let mut weights: Vec<_> = taper_w.iter().zip(steer.iter())
        .map(|(&t, &s)| s * t)
        .collect();

    let mut impairments = serde_json::Map::new();

    // Phase quantization
    if req.phase_bits > 0 {
        let rms_err = quantize_phase(&mut weights, req.phase_bits);
        impairments.insert("phase_quantization".into(), json!({
            "bits": req.phase_bits,
            "levels": 2_u32.pow(req.phase_bits),
            "rms_error_deg": rms_err,
        }));
    }

    // Mutual coupling
    if req.coupling_strength > 0.0 {
        let c_mat = mutual_coupling_matrix(n_elem, req.coupling_strength, 2.0);
        let mut w_new = vec![num_complex::Complex64::new(0.0, 0.0); n_elem];
        for i in 0..n_elem {
            for j in 0..n_elem {
                w_new[i] += c_mat[i][j] * weights[j];
            }
        }
        weights = w_new;
        impairments.insert("mutual_coupling".into(), json!({"strength": req.coupling_strength}));
    }

    // Element failures
    if req.failure_rate > 0.0 {
        let n_failed = simulate_failures(&mut weights, req.failure_rate, "open", 42);
        impairments.insert("element_failures".into(), json!({"rate": req.failure_rate, "n_failed": n_failed}));
    }

    let n_theta = req.resolution * 2;
    let n_phi = req.resolution * 2;
    let theta: Vec<f64> = (0..n_theta).map(|i| std::f64::consts::PI * i as f64 / (n_theta - 1) as f64).collect();
    let phi: Vec<f64> = (0..n_phi).map(|i| 2.0 * std::f64::consts::PI * i as f64 / (n_phi - 1) as f64).collect();

    let sin_theta: Vec<f64> = theta.iter().map(|t| t.sin()).collect();
    let cos_phi: Vec<f64> = phi.iter().map(|p| p.cos()).collect();
    let sin_phi: Vec<f64> = phi.iter().map(|p| p.sin()).collect();
    let w_real: Vec<f64> = weights.iter().map(|c| c.re).collect();
    let w_imag: Vec<f64> = weights.iter().map(|c| c.im).collect();

    let flat_pattern = gpu::compute_array_factor(
        x.as_slice().unwrap(), y.as_slice().unwrap(),
        &w_real, &w_imag, &sin_theta, &cos_phi, &sin_phi,
    );
    let max_db = flat_pattern.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let flat_norm: Vec<f64> = flat_pattern.iter().map(|v| v - max_db).collect();
    let pattern_db = ndarray::Array2::from_shape_vec((n_theta, n_phi), flat_norm).unwrap();

    // Element pattern
    let mut pattern_2d: Vec<Vec<f64>> = (0..n_theta).map(|i| pattern_db.row(i).to_vec()).collect();
    if req.element_q > 0.0 {
        physics::apply_element_pattern(&mut pattern_2d, &theta, &phi, req.element_q);
        impairments.insert("element_pattern".into(), json!({"q": req.element_q}));
    }

    let theta_cut: Vec<f64> = (-90..=90).map(|d| d as f64).collect();
    let cut_db = gpu::compute_beam_cut(
        x.as_slice().unwrap(), y.as_slice().unwrap(),
        &w_real, &w_imag, &theta_cut, req.beam.phi_deg,
    );
    let cut_max = cut_db.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let cut_db: Vec<f64> = cut_db.iter().map(|v| v - cut_max).collect();
    let beam_idx = (req.beam.theta_deg + 90.0) as usize;
    let sll = find_sidelobe(&cut_db, beam_idx);
    let bw = 51.0 / (req.array.nx as f64 * req.array.dx);

    Ok(Json(json!({
        "type": "3d",
        "metadata": {
            "geometry": format!("{:?}", req.array.geometry).to_lowercase(),
            "n_elements": n_elem,
            "nx": req.array.nx, "ny": req.array.ny,
            "spacing": {"dx": req.array.dx, "dy": req.array.dy},
            "beam_direction": {"theta_deg": req.beam.theta_deg, "phi_deg": req.beam.phi_deg},
            "taper": format!("{:?}", req.beam.taper).to_lowercase(),
            "first_sidelobe_db": sll,
            "estimated_beamwidth_deg": bw,
            "impairments": impairments,
        },
        "theta_deg": theta.iter().map(|t| t.to_degrees()).collect::<Vec<_>>(),
        "phi_deg": phi.iter().map(|p| p.to_degrees()).collect::<Vec<_>>(),
        "pattern_db": pattern_2d,
        "cut_through_beam": {"theta_deg": theta_cut, "pattern_db": cut_db},
    })))
}

async fn beam_squint(Json(req): Json<SquintRequest>) -> Json<Value> {
    let f_offsets: Vec<f64> = (0..51).map(|i| -req.bandwidth_ghz / 2.0 + req.bandwidth_ghz * i as f64 / 50.0).collect();
    let (theta_actual, squint) = beam_squint_analysis(req.theta_steer_deg, req.f0, &f_offsets);
    Json(json!({
        "frequency_ghz": f_offsets.iter().map(|&df| req.f0 + df).collect::<Vec<_>>(),
        "theta_actual_deg": theta_actual,
        "squint_error_deg": squint,
        "design_freq_ghz": req.f0,
        "design_theta_deg": req.theta_steer_deg,
    }))
}

// ═══ Error Handling ══════════════════════════════════════

struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"detail": format!("{}", self.0)}))).into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(err: E) -> Self {
        Self(err.into())
    }
}
