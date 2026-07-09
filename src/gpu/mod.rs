//! GPU dispatch layer.
//!
//! CUDA: loads PTX at runtime via Driver API FFI (cuda_ffi.rs).
//! CPU: Rayon multi-core fallback on all platforms.
//!
//! Architecture:
//!   compute_array_factor() → try CUDA → fallback CPU Rayon

pub mod cuda_ffi;

/// Embedded PTX module binary (compiled by nvcc at build time).
pub const PTX_BYTES: &[u8] = include_bytes!("../../kernels/kernels.ptx");

/// Available kernel names.
pub const KERNEL_NAMES: &[&str] = &[
    "array_factor_kernel",
    "beam_cut_kernel",
    "find_max_kernel",
];

/// Initialize CUDA. Call once at startup.
pub fn init_cuda() -> bool {
    if cuda_ffi::available() {
        return true;
    }
    cuda_ffi::init()
}

/// Check if CUDA is ready.
pub fn cuda_available() -> bool {
    cuda_ffi::available()
}

// ═══ Public Dispatch API ═════════════════════════════════

use std::ffi::c_void;

/// Compute 2D array factor — CUDA or CPU fallback.
pub fn compute_array_factor(
    x: &[f64],
    y: &[f64],
    w_real: &[f64],
    w_imag: &[f64],
    sin_theta: &[f64],
    cos_phi: &[f64],
    sin_phi: &[f64],
) -> Vec<f64> {
    if cuda_ffi::available() {
        match array_factor_cuda(x, y, w_real, w_imag, sin_theta, cos_phi, sin_phi) {
            Ok(result) => return result,
            Err(e) => tracing::warn!("GPU AF failed, CPU fallback: {}", e),
        }
    }
    array_factor_cpu(x, y, w_real, w_imag, sin_theta, cos_phi, sin_phi)
}

/// Compute 1D beam cut — CUDA or CPU fallback.
pub fn compute_beam_cut(
    x: &[f64],
    y: &[f64],
    w_real: &[f64],
    w_imag: &[f64],
    theta_deg: &[f64],
    phi_deg: f64,
) -> Vec<f64> {
    if cuda_ffi::available() {
        match beam_cut_cuda(x, y, w_real, w_imag, theta_deg, phi_deg) {
            Ok(result) => return result,
            Err(e) => tracing::warn!("GPU beam_cut failed, CPU fallback: {}", e),
        }
    }
    beam_cut_cpu(x, y, w_real, w_imag, theta_deg, phi_deg)
}

// ═══ CUDA Implementations ════════════════════════════════

fn array_factor_cuda(
    x: &[f64],
    y: &[f64],
    w_real: &[f64],
    w_imag: &[f64],
    sin_theta: &[f64],
    cos_phi: &[f64],
    sin_phi: &[f64],
) -> Result<Vec<f64>, String> {
    let n_theta = sin_theta.len();
    let n_phi = cos_phi.len();
    let n_elem = x.len();
    let total = n_theta * n_phi;

    // Allocate device buffers
    let d_x = cuda_ffi::GpuBuffer::alloc(x.len(), 8)?;
    let d_y = cuda_ffi::GpuBuffer::alloc(y.len(), 8)?;
    let d_wr = cuda_ffi::GpuBuffer::alloc(w_real.len(), 8)?;
    let d_wi = cuda_ffi::GpuBuffer::alloc(w_imag.len(), 8)?;
    let d_st = cuda_ffi::GpuBuffer::alloc(sin_theta.len(), 8)?;
    let d_cp = cuda_ffi::GpuBuffer::alloc(cos_phi.len(), 8)?;
    let d_sp = cuda_ffi::GpuBuffer::alloc(sin_phi.len(), 8)?;
    let d_out = cuda_ffi::GpuBuffer::alloc(total, 8)?;

    // Copy input data to device
    d_x.copy_host_to_device(x)?;
    d_y.copy_host_to_device(y)?;
    d_wr.copy_host_to_device(w_real)?;
    d_wi.copy_host_to_device(w_imag)?;
    d_st.copy_host_to_device(sin_theta)?;
    d_cp.copy_host_to_device(cos_phi)?;
    d_sp.copy_host_to_device(sin_phi)?;

    // Build argument buffer for kernel launch
    let n_theta_i32 = n_theta as i32;
    let n_phi_i32 = n_phi as i32;
    let n_elem_i32 = n_elem as i32;

    let mut args: Vec<*mut c_void> = vec![
        &d_x as *const _ as *mut c_void,
        &d_y as *const _ as *mut c_void,
        &d_wr as *const _ as *mut c_void,
        &d_wi as *const _ as *mut c_void,
        &d_st as *const _ as *mut c_void,
        &d_cp as *const _ as *mut c_void,
        &d_sp as *const _ as *mut c_void,
        &d_out as *const _ as *mut c_void,
        &n_theta_i32 as *const i32 as *mut c_void,
        &n_phi_i32 as *const i32 as *mut c_void,
        &n_elem_i32 as *const i32 as *mut c_void,
    ];

    // Launch
    let grid = ((total as u32 + 255) / 256, 1, 1);
    let block = (256, 1, 1);
    unsafe { cuda_ffi::launch_kernel("array_factor_kernel", grid, block, &args)?; }
    cuda_ffi::synchronize()?;

    // Copy result back
    let mut output = vec![0.0_f64; total];
    d_out.copy_device_to_host(&mut output)?;

    // Cleanup
    d_x.free()?;
    d_y.free()?;
    d_wr.free()?;
    d_wi.free()?;
    d_st.free()?;
    d_cp.free()?;
    d_sp.free()?;
    d_out.free()?;

    Ok(output)
}

fn beam_cut_cuda(
    x: &[f64],
    y: &[f64],
    w_real: &[f64],
    w_imag: &[f64],
    theta_deg: &[f64],
    phi_deg: f64,
) -> Result<Vec<f64>, String> {
    let n_angles = theta_deg.len();
    let n_elem = x.len();
    let cos_phi = phi_deg.to_radians().cos();
    let sin_phi = phi_deg.to_radians().sin();

    let d_x = cuda_ffi::GpuBuffer::alloc(x.len(), 8)?;
    let d_y = cuda_ffi::GpuBuffer::alloc(y.len(), 8)?;
    let d_wr = cuda_ffi::GpuBuffer::alloc(w_real.len(), 8)?;
    let d_wi = cuda_ffi::GpuBuffer::alloc(w_imag.len(), 8)?;
    let d_th = cuda_ffi::GpuBuffer::alloc(theta_deg.len(), 8)?;
    let d_out = cuda_ffi::GpuBuffer::alloc(n_angles, 8)?;

    d_x.copy_host_to_device(x)?;
    d_y.copy_host_to_device(y)?;
    d_wr.copy_host_to_device(w_real)?;
    d_wi.copy_host_to_device(w_imag)?;
    d_th.copy_host_to_device(theta_deg)?;

    let n_angles_i32 = n_angles as i32;
    let n_elem_i32 = n_elem as i32;

    let args: Vec<*mut c_void> = vec![
        &d_x as *const _ as *mut c_void,
        &d_y as *const _ as *mut c_void,
        &d_wr as *const _ as *mut c_void,
        &d_wi as *const _ as *mut c_void,
        &d_th as *const _ as *mut c_void,
        &cos_phi as *const f64 as *mut c_void,
        &sin_phi as *const f64 as *mut c_void,
        &d_out as *const _ as *mut c_void,
        &n_angles_i32 as *const i32 as *mut c_void,
        &n_elem_i32 as *const i32 as *mut c_void,
    ];

    let grid = ((n_angles as u32 + 255) / 256, 1, 1);
    unsafe { cuda_ffi::launch_kernel("beam_cut_kernel", grid, (256, 1, 1), &args)?; }
    cuda_ffi::synchronize()?;

    let mut output = vec![0.0_f64; n_angles];
    d_out.copy_device_to_host(&mut output)?;

    d_x.free()?;
    d_y.free()?;
    d_wr.free()?;
    d_wi.free()?;
    d_th.free()?;
    d_out.free()?;

    Ok(output)
}

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
                let mut sr = 0.0_f64;
                let mut si = 0.0_f64;
                for ie in 0..n_elem {
                    let phase = 2.0 * std::f64::consts::PI * (x[ie] * u + y[ie] * v);
                    let c = phase.cos();
                    let s = phase.sin();
                    sr += weights_real[ie] * c - weights_imag[ie] * s;
                    si += weights_real[ie] * s + weights_imag[ie] * c;
                }
                let mag = (sr * sr + si * si).sqrt();
                *val = if mag > 1e-15 { 20.0 * mag.log10() } else { -150.0 };
            }
        });
    output
}

pub fn beam_cut_cpu(
    x: &[f64],
    y: &[f64],
    wr: &[f64],
    wi: &[f64],
    theta_deg: &[f64],
    phi_deg: f64,
) -> Vec<f64> {
    use rayon::prelude::*;
    let n_elem = x.len();
    let cp = phi_deg.to_radians().cos();
    let sp = phi_deg.to_radians().sin();
    theta_deg.par_iter().map(|&th| {
        let tr = th.to_radians();
        let u = tr.sin() * cp;
        let v = tr.sin() * sp;
        let mut sr = 0.0; let mut si = 0.0;
        for ie in 0..n_elem {
            let ph = 2.0 * std::f64::consts::PI * (x[ie] * u + y[ie] * v);
            sr += wr[ie] * ph.cos() - wi[ie] * ph.sin();
            si += wr[ie] * ph.sin() + wi[ie] * ph.cos();
        }
        let mag = (sr * sr + si * si).sqrt();
        if mag > 1e-15 { 20.0 * mag.log10() } else { -150.0 }
    }).collect()
}
