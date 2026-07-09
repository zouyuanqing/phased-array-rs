// phased-array-rs CUDA Kernels
// Target: sm_75 (Turing), sm_89 (Ada), sm_120 (Blackwell)
// Compile: nvcc -ptx -arch=sm_75 -o kernels.ptx kernels.cu

#define _USE_MATH_DEFINES
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---- Kernel 1: Array Factor (parallel over theta x phi) ----
extern "C" __global__ void array_factor_kernel(
    const double* x,
    const double* y,
    const double* w_real,
    const double* w_imag,
    const double* sin_theta,
    const double* cos_phi,
    const double* sin_phi,
    double* output_db,
    int n_theta,
    int n_phi,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_theta * n_phi;
    if (idx >= total) return;

    int it = idx / n_phi;
    int ip = idx % n_phi;

    double u = sin_theta[it] * cos_phi[ip];
    double v = sin_theta[it] * sin_phi[ip];

    double sum_real = 0.0;
    double sum_imag = 0.0;

    for (int ie = 0; ie < n_elements; ie++) {
        double phase = 2.0 * M_PI * (x[ie] * u + y[ie] * v);
        double c = cos(phase);
        double s = sin(phase);
        sum_real += w_real[ie] * c - w_imag[ie] * s;
        sum_imag += w_real[ie] * s + w_imag[ie] * c;
    }

    double mag = sqrt(sum_real * sum_real + sum_imag * sum_imag);
    output_db[idx] = (mag > 1e-15) ? 20.0 * log10(mag) : -150.0;
}

// ---- Kernel 2: Beam Cut (1D at fixed phi) ----
extern "C" __global__ void beam_cut_kernel(
    const double* x,
    const double* y,
    const double* w_real,
    const double* w_imag,
    const double* theta_deg,
    double cos_phi,
    double sin_phi,
    double* output,
    int n_angles,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_angles) return;

    double th = theta_deg[idx] * M_PI / 180.0;
    double u = sin(th) * cos_phi;
    double v = sin(th) * sin_phi;

    double sum_real = 0.0;
    double sum_imag = 0.0;

    for (int ie = 0; ie < n_elements; ie++) {
        double phase = 2.0 * M_PI * (x[ie] * u + y[ie] * v);
        sum_real += w_real[ie] * cos(phase) - w_imag[ie] * sin(phase);
        sum_imag += w_real[ie] * sin(phase) + w_imag[ie] * cos(phase);
    }

    double mag = sqrt(sum_real * sum_real + sum_imag * sum_imag);
    output[idx] = (mag > 1e-15) ? 20.0 * log10(mag) : -150.0;
}

// ---- Kernel 3: Block Reduction Max-Find ----
extern "C" __global__ void find_max_kernel(
    const double* data,
    double* result,
    int n
) {
    __shared__ double smem[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    double local_max = -1e30;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        local_max = fmax(local_max, data[i]);
    }
    smem[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmax(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) result[blockIdx.x] = smem[0];
}
