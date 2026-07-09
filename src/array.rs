//! Array geometry and element position generation.
use ndarray::Array1;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArrayGeometry {
    Rectangular,
    Circular,
    Triangular,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayConfig {
    pub geometry: ArrayGeometry,
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
}

impl Default for ArrayConfig {
    fn default() -> Self {
        Self {
            geometry: ArrayGeometry::Rectangular,
            nx: 8,
            ny: 8,
            dx: 0.5,
            dy: 0.5,
        }
    }
}

/// Generate (x, y) element positions in wavelengths.
pub fn generate_positions(cfg: &ArrayConfig) -> (Array1<f64>, Array1<f64>) {
    match cfg.geometry {
        ArrayGeometry::Rectangular => rect_positions(cfg.nx, cfg.ny, cfg.dx, cfg.dy),
        ArrayGeometry::Circular => circular_positions(cfg.nx, cfg.ny, cfg.dx),
        ArrayGeometry::Triangular => triangular_positions(cfg.nx, cfg.ny, cfg.dx, cfg.dy),
    }
}

fn rect_positions(nx: usize, ny: usize, dx: f64, dy: f64) -> (Array1<f64>, Array1<f64>) {
    let n = nx * ny;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let x0 = (nx as f64 - 1.0) / 2.0;
    let y0 = (ny as f64 - 1.0) / 2.0;
    for i in 0..ny {
        for j in 0..nx {
            x.push((j as f64 - x0) * dx);
            y.push((i as f64 - y0) * dy);
        }
    }
    (Array1::from_vec(x), Array1::from_vec(y))
}

fn circular_positions(nx: usize, ny: usize, dr: f64) -> (Array1<f64>, Array1<f64>) {
    let n_rings = nx.min(ny);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for ring in 1..=n_rings {
        let r = ring as f64 * dr;
        let n_elems = (2.0 * std::f64::consts::PI * ring as f64 + 1.0) as usize;
        let n_elems = n_elems.max(6);
        for i in 0..n_elems {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n_elems as f64;
            x.push(r * angle.cos());
            y.push(r * angle.sin());
        }
    }
    (Array1::from_vec(x), Array1::from_vec(y))
}

fn triangular_positions(nx: usize, ny: usize, dx: f64, dy: f64) -> (Array1<f64>, Array1<f64>) {
    let mut x = Vec::with_capacity(nx * ny);
    let mut y = Vec::with_capacity(nx * ny);
    let x0 = (nx as f64 - 1.0) / 2.0;
    let y0 = (ny as f64 - 1.0) / 2.0;
    let sqrt3_2 = 3.0_f64.sqrt() / 2.0;
    for i in 0..ny {
        for j in 0..nx {
            let xp = (j as f64 - x0) * dx + (i % 2) as f64 * dx / 2.0;
            let yp = (i as f64 - y0) * dy * sqrt3_2;
            x.push(xp);
            y.push(yp);
        }
    }
    (Array1::from_vec(x), Array1::from_vec(y))
}
