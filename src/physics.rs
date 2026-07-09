//! Realistic physics: element patterns, mutual coupling, phase quantization,
//! beam squint, element failures.

use ndarray::Array1;
use num_complex::Complex64;
use rand::Rng;

/// Element pattern: E(θ) = cos^q(θ), cosθ clipped to [0,1].
pub fn element_pattern_cosq(theta: &[f64], q: f64) -> Vec<f64> {
    theta
        .iter()
        .map(|&t| {
            let cos_th = t.cos().max(0.0);
            if q == 0.0 {
                1.0
            } else {
                cos_th.powf(q)
            }
        })
        .collect()
}

/// Apply element pattern to array factor (in dB).
pub fn apply_element_pattern(
    pattern_db: &mut [Vec<f64>],
    theta: &[f64],
    phi: &[f64],
    q: f64,
) {
    let ep = element_pattern_cosq(theta, q);
    let ep_db: Vec<f64> = ep.iter().map(|&v| 20.0 * v.max(1e-15).log10()).collect();
    let n_theta = theta.len();
    let n_phi = phi.len();
    for i in 0..n_theta {
        for j in 0..n_phi {
            pattern_db[i][j] += ep_db[i];
        }
    }
}

/// Build mutual coupling matrix (Toeplitz).
pub fn mutual_coupling_matrix(
    n_elements: usize,
    strength: f64,
    decay: f64,
) -> Vec<Vec<Complex64>> {
    let mut c = vec![vec![Complex64::new(0.0, 0.0); n_elements]; n_elements];
    for i in 0..n_elements {
        c[i][i] = Complex64::new(1.0, 0.0);
        for j in (i + 1)..n_elements {
            let d = (j - i) as f64;
            let coupling = strength * (-d / decay).exp();
            let phase = std::f64::consts::PI * d * 0.1;
            let val = Complex64::new(coupling * phase.cos(), coupling * phase.sin());
            c[i][j] = val;
            c[j][i] = val.conj();
        }
    }
    c
}

/// Quantize complex weights to n_bits phase resolution.
pub fn quantize_phase(weights: &mut [Complex64], n_bits: u32) -> f64 {
    if n_bits == 0 {
        return 0.0;
    }
    let n_levels = 2_u32.pow(n_bits) as f64;
    let step = 2.0 * std::f64::consts::PI / n_levels;
    let mut sum_sq_err = 0.0_f64;

    for w in weights.iter_mut() {
        let amp = w.norm();
        let phase = w.arg();
        let quantized = (phase / step).round() * step;
        let err = phase - quantized;
        sum_sq_err += err * err;
        *w = Complex64::new(amp * quantized.cos(), amp * quantized.sin());
    }

    (sum_sq_err / weights.len() as f64).sqrt().to_degrees()
}

/// Simulate random element failures.
pub fn simulate_failures(
    weights: &mut [Complex64],
    failure_rate: f64,
    failure_type: &str,
    seed: u64,
) -> usize {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut n_failed = 0;

    for w in weights.iter_mut() {
        if rng.gen::<f64>() < failure_rate {
            n_failed += 1;
            match failure_type {
                "open" => *w = Complex64::new(0.0, 0.0),
                "short" => {
                    let phase = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
                    *w = Complex64::new(w.norm() * phase.cos(), w.norm() * phase.sin());
                }
                "drift" => {
                    let drift = rng.gen::<f64>() * std::f64::consts::PI / 2.0 - std::f64::consts::PI / 4.0;
                    *w *= Complex64::new(drift.cos(), drift.sin());
                }
                _ => {}
            }
        }
    }

    n_failed
}

/// Beam squint: actual steering angle at offset frequency.
pub fn beam_squint_analysis(
    theta_steer_deg: f64,
    f0: f64,
    f_offsets: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let sin_steer = theta_steer_deg.to_radians().sin();
    let mut theta_actual = Vec::with_capacity(f_offsets.len());
    let mut squint = Vec::with_capacity(f_offsets.len());

    for &df in f_offsets {
        let f = f0 + df;
        let sin_act = (f0 / f * sin_steer).clamp(-1.0, 1.0);
        let th = sin_act.asin().to_degrees();
        theta_actual.push(th);
        squint.push(th - theta_steer_deg);
    }

    (theta_actual, squint)
}
