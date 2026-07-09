//! Beamforming computation — Array Factor, steering, tapering.
//!
//! CPU path uses Rayon for multi-core parallelism.
//! GPU path dispatches to CUDA kernels via the `gpu` module.

use ndarray::{Array1, Array2, s};
use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeamConfig {
    pub theta_deg: f64,
    pub phi_deg: f64,
    pub taper: TaperType,
    pub sidelobe_db: f64,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            theta_deg: 0.0,
            phi_deg: 0.0,
            taper: TaperType::Uniform,
            sidelobe_db: -30.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TaperType {
    Uniform,
    Taylor,
    Hamming,
    Hanning,
}

// ═══ Steering Vector ═════════════════════════════════════

/// Compute complex steering vector for beam direction (θ, φ).
pub fn steering_vector(
    x: &Array1<f64>, y: &Array1<f64>,
    theta_deg: f64, phi_deg: f64,
) -> Vec<Complex64> {
    let theta = theta_deg.to_radians();
    let phi = phi_deg.to_radians();
    let u = theta.sin() * phi.cos();
    let v = theta.sin() * phi.sin();
    let n = x.len();
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let phase = 2.0 * std::f64::consts::PI * (x[i] * u + y[i] * v);
        w.push(Complex64::new(phase.cos(), -phase.sin()));
    }
    w
}

// ═══ Tapering ════════════════════════════════════════════

/// Apply amplitude taper.
/// For rectangular grids: 2D separable. For others: radial distance taper.
pub fn apply_taper(
    x: &Array1<f64>, y: &Array1<f64>,
    nx: usize, ny: usize,
    taper: &TaperType,
    sidelobe_db: f64,
    geometry: &str,
) -> Vec<f64> {
    let n = x.len();
    if *taper == TaperType::Uniform {
        return vec![1.0; n];
    }

    if geometry != "rectangular" {
        return radial_taper(x, y, taper, sidelobe_db);
    }

    // 2D separable for rectangular
    match taper {
        TaperType::Taylor => {
            let tx = taylor_1d(nx, sidelobe_db);
            let ty = taylor_1d(ny, sidelobe_db);
            let mut w = Vec::with_capacity(ny * nx);
            for iy in 0..ny {
                for ix in 0..nx {
                    w.push(ty[iy] * tx[ix]);
                }
            }
            w
        }
        TaperType::Hamming => separable_window(nx, ny, |n| {
            (0..n).map(|i| 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()).collect()
        }),
        TaperType::Hanning => separable_window(nx, ny, |n| {
            (0..n).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()).collect()
        }),
        _ => vec![1.0; n],
    }
}

fn taylor_1d(n: usize, sll_db: f64) -> Vec<f64> {
    let nbar = (n / 2).min(5).max(1);
    let a = (10.0_f64.powf(sll_db.abs() / 20.0)).acosh() / std::f64::consts::PI;
    let a2 = a * a;
    let n2 = (2 * nbar) as f64;
    let n2_sq = n2 * n2;

    let mut w = vec![0.0_f64; n];
    for i in 0..n {
        let xi = (i as f64 - (n - 1) as f64 / 2.0) / (n as f64) * 2.0;
        let mut f = 0.0;
        for m in 1..nbar {
            let mf = m as f64;
            let fm = {
                let num = (1..nbar).fold(1.0, |acc, p| {
                    let pf = p as f64;
                    if p == m {
                        acc * (-1.0_f64).powi((m + 1) as i32) / (2.0 * pf)
                    } else {
                        acc * pf * pf / (a2 + (mf - 0.5).powi(2))
                    }
                });
                num * (a2 + (mf - 0.5).powi(2)).sqrt()
            };
            f += fm * (std::f64::consts::PI * mf * xi).cos();
        }
        w[i] = 1.0 + 2.0 * f;
    }
    // Normalize
    let max_w = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    w.iter_mut().for_each(|v| *v /= max_w);
    w
}

fn separable_window(nx: usize, ny: usize, win_fn: fn(usize) -> Vec<f64>) -> Vec<f64> {
    let wx = win_fn(nx);
    let wy = win_fn(ny);
    let mut w = Vec::with_capacity(nx * ny);
    for iy in 0..ny {
        for ix in 0..nx {
            w.push(wy[iy] * wx[ix]);
        }
    }
    w
}

fn radial_taper(x: &Array1<f64>, y: &Array1<f64>, taper: &TaperType, sll_db: f64) -> Vec<f64> {
    let n = x.len();
    let r: Vec<f64> = (0..n).map(|i| (x[i].powi(2) + y[i].powi(2)).sqrt()).collect();
    let r_max = r.iter().cloned().fold(0.0_f64, f64::max).max(1.0);
    let r_norm: Vec<f64> = r.iter().map(|v| v / r_max).collect();

    match taper {
        TaperType::Taylor => {
            let shape = (sll_db.abs() / 30.0).max(0.5);
            r_norm.iter().map(|&rn| {
                (std::f64::consts::FRAC_PI_2 * (1.0 - rn.min(0.99))).cos().powf(shape)
            }).collect()
        }
        TaperType::Hamming => r_norm.iter().map(|&rn| {
            0.54 - 0.46 * (2.0 * std::f64::consts::PI * rn).cos()
        }).collect(),
        TaperType::Hanning => r_norm.iter().map(|&rn| {
            0.5 - 0.5 * (2.0 * std::f64::consts::PI * rn).cos()
        }).collect(),
        _ => vec![1.0; n],
    }
}

// ═══ Array Factor ════════════════════════════════════════

/// Compute 2D array factor pattern (CPU, Rayon-parallelized).
///
/// AF(θ,φ) = Σ w_n * exp(j * 2π * (x_n * sinθ cosφ + y_n * sinθ sinφ))
///
/// Returns pattern_dB normalized to 0 dB max.
pub fn compute_array_factor(
    x: &Array1<f64>, y: &Array1<f64>,
    weights: &[Complex64],
    theta: &[f64],   // radians, len = n_theta
    phi: &[f64],     // radians, len = n_phi
) -> Array2<f64> {
    let n_elem = x.len();
    let n_theta = theta.len();
    let n_phi = phi.len();

    // Precompute sin(theta) and cos(theta)
    let sin_theta: Vec<f64> = theta.iter().map(|t| t.sin()).collect();
    let cos_phi: Vec<f64> = phi.iter().map(|p| p.cos()).collect();
    let sin_phi: Vec<f64> = phi.iter().map(|p| p.sin()).collect();

    // Parallel over theta
    let pattern: Vec<Vec<f64>> = (0..n_theta)
        .into_par_iter()
        .map(|it| {
            let mut row = vec![0.0_f64; n_phi];
            for ip in 0..n_phi {
                let u = sin_theta[it] * cos_phi[ip];
                let v = sin_theta[it] * sin_phi[ip];
                let mut af = Complex64::new(0.0, 0.0);
                for ie in 0..n_elem {
                    let phase = 2.0 * std::f64::consts::PI * (x[ie] * u + y[ie] * v);
                    let contrib = weights[ie] * Complex64::new(phase.cos(), phase.sin());
                    af += contrib;
                }
                let mag = af.norm();
                row[ip] = if mag > 1e-15 { 20.0 * mag.log10() } else { -150.0 };
            }
            row
        })
        .collect();

    // Normalize
    let max_db = pattern.iter().flatten().cloned().fold(f64::NEG_INFINITY, f64::max);
    let pattern_norm: Vec<Vec<f64>> = pattern
        .into_iter()
        .map(|row| row.into_iter().map(|v| v - max_db).collect())
        .collect();

    // Convert to ndarray
    let flat: Vec<f64> = pattern_norm.into_iter().flatten().collect();
    Array2::from_shape_vec((n_theta, n_phi), flat).unwrap()
}

/// Compute 1D beam cut (E-plane or H-plane).
pub fn compute_beam_cut(
    x: &Array1<f64>, y: &Array1<f64>,
    weights: &[Complex64],
    phi_deg: f64,
    theta_range: &[f64], // degrees
) -> Vec<f64> {
    let phi_rad = phi_deg.to_radians();
    let cos_phi = phi_rad.cos();
    let sin_phi = phi_rad.sin();
    let n_elem = x.len();

    let pattern: Vec<f64> = theta_range
        .par_iter()
        .map(|&th_deg| {
            let th = th_deg.to_radians();
            let u = th.sin() * cos_phi;
            let v = th.sin() * sin_phi;
            let mut af = Complex64::new(0.0, 0.0);
            for ie in 0..n_elem {
                let phase = 2.0 * std::f64::consts::PI * (x[ie] * u + y[ie] * v);
                af += weights[ie] * Complex64::new(phase.cos(), phase.sin());
            }
            let mag = af.norm();
            if mag > 1e-15 { 20.0 * mag.log10() } else { -150.0 }
        })
        .collect();

    let max_db = pattern.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    pattern.into_iter().map(|v| v - max_db).collect()
}

/// Find first sidelobe level from a 1D beam cut.
pub fn find_sidelobe(pattern: &[f64], beam_idx: usize) -> f64 {
    let n = pattern.len();

    // Find -3dB points around main beam
    let mut left_3db = beam_idx;
    for i in (0..beam_idx).rev() {
        if pattern[i] < -3.0 {
            left_3db = i;
            break;
        }
    }
    let mut right_3db = beam_idx;
    for i in beam_idx..n {
        if pattern[i] < -3.0 {
            right_3db = i;
            break;
        }
    }

    let half_bw = (right_3db - beam_idx).max(beam_idx - left_3db).max(2);
    let margin = (half_bw as f64 * 1.8) as usize;
    let left_edge = left_3db.saturating_sub(margin).max(1);
    let right_edge = (right_3db + margin).min(n - 2);

    // Find local peaks outside main lobe
    let mut best = -50.0_f64;
    // Right side
    for i in (right_edge + 5).min(n - 3)..n - 3 {
        if pattern[i] > pattern[i - 1] && pattern[i] > pattern[i + 1] && pattern[i] > -50.0 {
            best = best.max(pattern[i]);
            break;
        }
    }
    // Left side
    for i in (5..left_edge.saturating_sub(5)).rev() {
        if i > 0 && i + 1 < n && pattern[i] > pattern[i - 1] && pattern[i] > pattern[i + 1] && pattern[i] > -50.0 {
            best = best.max(pattern[i]);
            break;
        }
    }

    if best > -50.0 { best } else { -30.0 }
}
