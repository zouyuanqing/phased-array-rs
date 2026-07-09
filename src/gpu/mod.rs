//! GPU dispatch layer.
//!
//! CUDA PTX kernels are pre-compiled in `kernels/kernels.ptx`.
//! At runtime, PTX is loaded via CUDA Driver API (Python pycuda launcher
//! or future Rust CUDA FFI).
//!
//! Current default: CPU Rayon multi-core fallback on all platforms.

/// Check if CUDA PTX kernels are available.
/// PTX is embedded at compile time via include_bytes!.
pub fn cuda_available() -> bool {
    true // PTX is embedded, always available
}

/// Embedded PTX module (for runtime CUDA loading).
pub const PTX_BYTES: &[u8] = include_bytes!("../../kernels/kernels.ptx");

/// Available kernel names.
pub const KERNEL_NAMES: &[&str] = &[
    "array_factor_kernel",
    "beam_cut_kernel",
    "find_max_kernel",
];

// ═══ CPU Multi-Core Fallback (Rayon) ═════════════════════

pub fn array_factor_cpu(
    x: &[f64],
    y: &[f64],
    weights_real: &[f64],
    weights_imag: &[f64],
    sin_theta: &[f64],
    cos_phi: &[f64],
    sin_phi: &[f64],
) -> Vec<f64> {
    use rayon::prelude::*;

    let n_theta = sin_theta.len();
    let n_phi = cos_phi.len();
    let n_elem = x.len();
    let total = n_theta * n_phi;

    let mut output = vec![0.0_f64; total];
    output
        .par_chunks_mut(n_phi)
        .enumerate()
        .for_each(|(it, row)| {
            for (ip, val) in row.iter_mut().enumerate() {
                let u = sin_theta[it] * cos_phi[ip];
                let v = sin_theta[it] * sin_phi[ip];
                let mut sum_real = 0.0_f64;
                let mut sum_imag = 0.0_f64;
                for ie in 0..n_elem {
                    let phase = 2.0 * std::f64::consts::PI * (x[ie] * u + y[ie] * v);
                    let c = phase.cos();
                    let s = phase.sin();
                    sum_real += weights_real[ie] * c - weights_imag[ie] * s;
                    sum_imag += weights_real[ie] * s + weights_imag[ie] * c;
                }
                let mag = (sum_real * sum_real + sum_imag * sum_imag).sqrt();
                *val = if mag > 1e-15 { 20.0 * mag.log10() } else { -150.0 };
            }
        });

    output
}

pub fn beam_cut_cpu(
    x: &[f64],
    y: &[f64],
    weights_real: &[f64],
    weights_imag: &[f64],
    theta_deg: &[f64],
    phi_deg: f64,
) -> Vec<f64> {
    use rayon::prelude::*;

    let n_elem = x.len();
    let cos_phi = phi_deg.to_radians().cos();
    let sin_phi = phi_deg.to_radians().sin();

    theta_deg
        .par_iter()
        .map(|&th| {
            let th_rad = th.to_radians();
            let u = th_rad.sin() * cos_phi;
            let v = th_rad.sin() * sin_phi;
            let mut sum_real = 0.0;
            let mut sum_imag = 0.0;
            for ie in 0..n_elem {
                let phase = 2.0 * std::f64::consts::PI * (x[ie] * u + y[ie] * v);
                sum_real += weights_real[ie] * phase.cos()
                    - weights_imag[ie] * phase.sin();
                sum_imag += weights_real[ie] * phase.sin()
                    + weights_imag[ie] * phase.cos();
            }
            let mag = (sum_real * sum_real + sum_imag * sum_imag).sqrt();
            if mag > 1e-15 { 20.0 * mag.log10() } else { -150.0 }
        })
        .collect()
}
