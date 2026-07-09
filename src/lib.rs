//! Phased Array Beamforming — Core Physics Engine (Rust)
//!
//! GPU-accelerated via CUDA with CPU multi-core fallback via Rayon.

pub mod array;
pub mod beamforming;
pub mod gpu;
pub mod physics;
pub mod super_res;

// Re-exports
pub use array::{ArrayConfig, ArrayGeometry, generate_positions};
pub use beamforming::{
    apply_taper, compute_array_factor, compute_beam_cut,
    steering_vector, BeamConfig,
};
pub use physics::{
    apply_element_pattern, beam_squint_analysis,
    element_pattern_cosq, mutual_coupling_matrix,
    quantize_phase, simulate_failures,
};
pub use super_res::{esprit_ula, generate_snapshots, music_ula, mvdr_weights};
