use rand::rng;
use rand::RngExt;

pub fn generate_data(n: u64, dim: u64) -> Vec<Vec<f32>> { 
    let points: Vec<Vec<f32>> = (0..n).map(|_|
        (0..dim).map(
            |_| rng().random_range(0.0..1.0)
        ).collect()
    ).collect();

    return points;
}

pub fn generate_query(n: u64, dim: u64) -> Vec<Vec<f32>> {
    let points: Vec<Vec<f32>> = (0..n).map(|_|
        (0..dim).map(
            |_| rng().random_range(0.0..1.0)
        ).collect()
    ).collect();

    return points;
}