use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Default seeds, so benchmarks and tests see the same data on every run.
pub const DATA_SEED: u64 = 42;
pub const QUERY_SEED: u64 = 4242;

/// `n` points with `dim` coordinates drawn uniformly from [0, 1), reproducible for a given seed.
pub fn generate_data_seeded(n: u64, dim: u64, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.random_range(0.0..1.0)).collect())
        .collect()
}

pub fn generate_data(n: u64, dim: u64) -> Vec<Vec<f32>> {
    generate_data_seeded(n, dim, DATA_SEED)
}

pub fn generate_query(n: u64, dim: u64) -> Vec<Vec<f32>> {
    generate_data_seeded(n, dim, QUERY_SEED)
}
