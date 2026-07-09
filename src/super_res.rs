//! Super-resolution DOA estimation: MUSIC, ESPRIT, MVDR.
use ndarray::Array2;
use num_complex::Complex64;
use rayon::prelude::*;

/// Generate synthetic array snapshots for DOA testing.
pub fn generate_snapshots(
    n_elements: usize,
    n_snapshots: usize,
    source_angles_deg: &[f64],
    snr_db: f64,
    seed: u64,
) -> Array2<Complex64> {
    use rand::Rng;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let d = 0.5;
    let positions: Vec<f64> = (0..n_elements)
        .map(|i| (i as f64 - (n_elements - 1) as f64 / 2.0) * d)
        .collect();

    // Steering matrix
    let n_src = source_angles_deg.len();
    let mut A = Array2::zeros((n_elements, n_src));
    for (j, &angle) in source_angles_deg.iter().enumerate() {
        let sin_th = angle.to_radians().sin();
        for i in 0..n_elements {
            let phase = -2.0 * std::f64::consts::PI * positions[i] * sin_th;
            A[[i, j]] = Complex64::new(phase.cos(), phase.sin());
        }
    }

    let signal_power = 1.0;
    let noise_power = signal_power / 10.0_f64.powf(snr_db / 10.0);

    let mut rng = StdRng::seed_from_u64(seed);

    // Generate signals
    let mut S = Array2::zeros((n_src, n_snapshots));
    for j in 0..n_src {
        for k in 0..n_snapshots {
            let re = rng.gen::<f64>();
            let im = rng.gen::<f64>();
            S[[j, k]] = Complex64::new(re, im) * signal_power.sqrt();
        }
    }

    // Generate noise
    let mut N = Array2::zeros((n_elements, n_snapshots));
    for i in 0..n_elements {
        for k in 0..n_snapshots {
            let re = rng.gen::<f64>();
            let im = rng.gen::<f64>();
            N[[i, k]] = Complex64::new(re, im) * (noise_power / 2.0).sqrt();
        }
    }

    // X = A * S + N
    let mut X = A.dot(&S);
    X += &N;
    X
}

// ═══ MUSIC ═══════════════════════════════════════════════

/// MUSIC pseudo-spectrum for ULA.
pub fn music_ula(
    n_elements: usize,
    snapshots: &Array2<Complex64>,
    n_sources: usize,
    n_angles: usize,
) -> (Vec<f64>, Vec<f64>) {
    let (_, eigenvectors) = covariance_eigendecomp(snapshots);

    // Noise subspace: columns n_sources..end
    let n_snap = snapshots.shape()[1];
    let mut E_n: Vec<Vec<Complex64>> = Vec::new();
    for i in n_sources..n_elements {
        let mut col = Vec::with_capacity(n_elements);
        for j in 0..n_elements {
            col.push(eigenvectors[[j, i]]);
        }
        E_n.push(col);
    }

    // Scan angles
    let theta_deg: Vec<f64> = (0..n_angles)
        .map(|i| -90.0 + 180.0 * i as f64 / (n_angles - 1) as f64)
        .collect();

    let d = 0.5;
    let positions: Vec<f64> = (0..n_elements)
        .map(|i| (i as f64 - (n_elements - 1) as f64 / 2.0) * d)
        .collect();

    let spectrum: Vec<f64> = theta_deg
        .par_iter()
        .map(|&th| {
            let sin_th = th.to_radians().sin();
            let a: Vec<Complex64> = positions
                .iter()
                .map(|&pos| {
                    let phase = -2.0 * std::f64::consts::PI * pos * sin_th;
                    Complex64::new(phase.cos(), phase.sin())
                })
                .collect();

            // denom = a^H * E_n * E_n^H * a
            let mut denom = 0.0_f64;
            for en_col in &E_n {
                let mut dot = Complex64::new(0.0, 0.0);
                for k in 0..n_elements {
                    dot += a[k].conj() * en_col[k];
                }
                denom += dot.norm_sqr();
            }

            1.0 / (denom + 1e-15)
        })
        .collect();

    // Normalize to dB
    let max_p = spectrum.iter().cloned().fold(0.0_f64, f64::max);
    let spectrum_db: Vec<f64> = spectrum
        .iter()
        .map(|&p| 10.0 * (p / max_p).log10())
        .collect();

    (theta_deg, spectrum_db)
}

// ═══ ESPRIT ══════════════════════════════════════════════

/// ESPRIT DOA estimation for ULA.
pub fn esprit_ula(
    n_elements: usize,
    snapshots: &Array2<Complex64>,
    n_sources: usize,
) -> Vec<f64> {
    let (_, eigenvectors) = covariance_eigendecomp(snapshots);

    // Signal subspace: first n_sources columns
    let mut E_s = Array2::zeros((n_elements, n_sources));
    for i in 0..n_elements {
        for j in 0..n_sources {
            E_s[[i, j]] = eigenvectors[[i, j]];
        }
    }

    // Subarrays
    let m = n_elements - 1;
    let mut E1 = Array2::zeros((m, n_sources));
    let mut E2 = Array2::zeros((m, n_sources));
    for i in 0..m {
        for j in 0..n_sources {
            E1[[i, j]] = E_s[[i, j]];
            E2[[i, j]] = E_s[[i + 1, j]];
        }
    }

    // Psi = (E1^H E1)^{-1} E1^H E2
    let e1h = conjugate_transpose(&E1);
    let e1h_e1 = e1h.dot(&E1);
    let e1h_e1_inv = invert_2d(&e1h_e1);
    let e1h_e2 = e1h.dot(&E2);
    let psi = e1h_e1_inv.dot(&e1h_e2);

    // Eigenvalues of Psi
    let evals = eigen_2x2(&psi, n_sources);

    // DOA = arcsin(-arg(λ) / (π))
    let d = 0.5;
    let mut angles: Vec<f64> = evals
        .iter()
        .map(|&z| {
            let phi = z.arg();
            (-phi / (2.0 * std::f64::consts::PI * d)).asin().to_degrees()
        })
        .collect();

    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    angles
}

// ═══ MVDR ════════════════════════════════════════════════

/// MVDR/Capon adaptive beamforming weights.
pub fn mvdr_weights(
    x: &[f64], y: &[f64],
    snapshots: &Array2<Complex64>,
    theta_look_deg: f64,
    diagonal_loading: f64,
) -> Vec<Complex64> {
    let n_elem = x.len();
    let n_snap = snapshots.shape()[1];

    // Covariance
    let mut R: Array2<Complex64> = Array2::zeros((n_elem, n_elem));
    for k in 0..n_snap {
        let col = snapshots.column(k);
        for i in 0..n_elem {
            for j in 0..n_elem {
                R[[i, j]] += col[i] * col[j].conj();
            }
        }
    }
    for i in 0..n_elem {
        R[[i, i]] = R[[i, i]] / n_snap as f64 + diagonal_loading;
        for j in (i + 1)..n_elem {
            R[[i, j]] /= n_snap as f64;
            R[[j, i]] = R[[i, j]].conj();
        }
    }

    // Steering vector for look direction
    let th = theta_look_deg.to_radians();
    let a: Vec<Complex64> = (0..n_elem)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * x[i] * th.sin();
            Complex64::new(phase.cos(), -phase.sin())
        })
        .collect();

    // w = R^{-1} a / (a^H R^{-1} a)
    let r_inv = invert_2d(&R);
    let mut w = vec![Complex64::new(0.0, 0.0); n_elem];
    for i in 0..n_elem {
        for j in 0..n_elem {
            w[i] += r_inv[[i, j]] * a[j];
        }
    }

    let mut denom = Complex64::new(0.0, 0.0);
    for i in 0..n_elem {
        denom += a[i].conj() * w[i];
    }

    for i in 0..n_elem {
        w[i] /= denom;
    }

    w
}

// ═══ Linear Algebra Helpers ══════════════════════════════

fn covariance_eigendecomp(snapshots: &Array2<Complex64>) -> (Vec<f64>, Array2<Complex64>) {
    let n_elem = snapshots.shape()[0];
    let n_snap = snapshots.shape()[1];

    // R = X X^H / K
    let mut R = Array2::zeros((n_elem, n_elem));
    for k in 0..n_snap {
        let col = snapshots.column(k);
        for i in 0..n_elem {
            for j in 0..n_elem {
                R[[i, j]] += col[i] * col[j].conj();
            }
        }
    }
    for i in 0..n_elem {
        for j in 0..n_elem {
            R[[i, j]] /= n_snap as f64;
        }
    }

    // For now, use a simple power iteration for Hermitian matrices
    // In production, use LAPACK (zheev) via ndarray-linalg or nalgebra-lapack
    eigen_hermitian(&R, n_elem)
}

/// Simple eigenvalue decomposition for small Hermitian matrices using Jacobi-like method.
fn eigen_hermitian(R: &Array2<Complex64>, n: usize) -> (Vec<f64>, Array2<Complex64>) {
    // Use power iteration + deflation for the largest eigenvalues
    let n_sources = (n / 2).min(8);
    let mut eigenvalues = vec![0.0_f64; n];
    let mut eigenvectors = Array2::zeros((n, n));

    // Initialize identity
    for i in 0..n {
        eigenvectors[[i, i]] = Complex64::new(1.0, 0.0);
    }

    // Jacobi eigenvalue algorithm for Hermitian matrices
    let mut A = R.clone();
    let max_iter = 50 * n;
    let tol = 1e-10;

    for _iter in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = A[[i, j]].norm();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break;
        }

        // Compute rotation
        let app = A[[p, p]].re;
        let aqq = A[[q, q]].re;
        let apq = A[[p, q]];

        let theta = 0.5 * (2.0 * apq.norm()).atan2(app - aqq);
        let c = theta.cos();
        let s = Complex64::new(
            theta.sin() * apq.re / apq.norm().max(1e-15),
            theta.sin() * apq.im / apq.norm().max(1e-15),
        );

        // Apply rotation to A
        for i in 0..n {
            if i != p && i != q {
                let aip = A[[i, p]];
                let aiq = A[[i, q]];
                A[[i, p]] = aip * c + aiq * s.conj();
                A[[p, i]] = A[[i, p]].conj();
                A[[i, q]] = -aip * s + aiq * c;
                A[[q, i]] = A[[i, q]].conj();
            }
        }
        A[[p, p]] = Complex64::new(app * c * c + aqq * s.norm_sqr() + 2.0 * apq.re * c * s.re, 0.0);
        A[[q, q]] = Complex64::new(app * s.norm_sqr() + aqq * c * c - 2.0 * apq.re * c * s.re, 0.0);
        A[[p, q]] = Complex64::new(0.0, 0.0);
        A[[q, p]] = Complex64::new(0.0, 0.0);

        // Update eigenvectors
        for i in 0..n {
            let eip = eigenvectors[[i, p]];
            let eiq = eigenvectors[[i, q]];
            eigenvectors[[i, p]] = eip * c + eiq * s.conj();
            eigenvectors[[i, q]] = -eip * s + eiq * c;
        }
    }

    // Extract eigenvalues
    for i in 0..n {
        eigenvalues[i] = A[[i, i]].re;
    }

    // Sort descending
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

    let sorted_evals: Vec<f64> = idx.iter().map(|&i| eigenvalues[i]).collect();
    let mut sorted_evecs = Array2::zeros((n, n));
    for j in 0..n {
        for i in 0..n {
            sorted_evecs[[i, j]] = eigenvectors[[i, idx[j]]];
        }
    }

    (sorted_evals, sorted_evecs)
}

fn conjugate_transpose(a: &Array2<Complex64>) -> Array2<Complex64> {
    let (rows, cols) = (a.shape()[0], a.shape()[1]);
    let mut result = Array2::zeros((cols, rows));
    for i in 0..rows {
        for j in 0..cols {
            result[[j, i]] = a[[i, j]].conj();
        }
    }
    result
}

fn invert_2d(a: &Array2<Complex64>) -> Array2<Complex64> {
    let n = a.shape()[0];
    // Copy to simple Vec for borrow-free Gauss-Jordan
    let mut data: Vec<Vec<Complex64>> = (0..n).map(|i| {
        let mut row: Vec<Complex64> = (0..n).map(|j| a[[i, j]]).collect();
        for j in 0..n {
            row.push(if i == j { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) });
        }
        row
    }).collect();

    for col in 0..n {
        let pivot = data[col][col];
        if pivot.norm() < 1e-12 { continue; }
        for j in 0..2 * n {
            data[col][j] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = data[row][col];
                // split_at_mut to avoid borrow conflict
                let (a, b) = if row < col {
                    let (left, right) = data.split_at_mut(col);
                    (&mut left[row], &right[0])
                } else {
                    let (left, right) = data.split_at_mut(row);
                    (&mut right[0], &left[col])
                };
                for j in 0..2 * n {
                    a[j] -= factor * b[j];
                }
            }
        }
    }

    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] = data[i][j + n];
        }
    }
    result
}

fn eigen_2x2(a: &Array2<Complex64>, _n: usize) -> Vec<Complex64> {
    // Closed-form eigenvalues for 2x2 matrix
    let trace = a[[0, 0]] + a[[1, 1]];
    let det = a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]];
    let disc = (trace * trace - 4.0 * det).sqrt();

    vec![
        (trace + disc) / 2.0,
        (trace - disc) / 2.0,
    ]
}
