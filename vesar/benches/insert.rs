use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion, BatchSize};
use vesar::core::ann_index::ANNIndex;
use vesar::core::hnsw_index::HNSWIndex;
use vesar::datasets::synthetic_data::generate_data;

fn parse_env_list<T>(var_name: &str, default_val: &str) -> Vec<T> 
where 
    T: std::str::FromStr,
{
    std::env::var(var_name)
        .unwrap_or(String::from(default_val))
        .split(',')
        .map(|s| s.replace('_', "").trim().parse::<T>())
        .flatten() // Silently skips errors/empty strings
        .collect()
}

fn bench_insert_ann(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_bench_ann");
    
    let n_set: Vec<u64> = parse_env_list("N", "1_000_000");
    let dim_set: Vec<u64> = parse_env_list("DIM", "16");
    let gd_set: Vec<usize> = parse_env_list("GD", "10");
    let m_set: Vec<usize> = parse_env_list("M", "5");

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim);
            group.throughput(criterion::Throughput::Elements(points.len() as u64));
            for &gd in &gd_set {
                for &m in &m_set {
                    group.bench_function(format!("insert_n_{}_dim_{}_gd_{}_m_{}", n, dim, gd, m),
                    |b| {
                        b.iter_batched(|| ANNIndex::new(),
                    |mut db| {
                        for point in &points {
                            db.insert(black_box(point), gd, m);
                        }}, 
                        BatchSize::LargeInput);
                    });
                }
            }
        }
    }

    group.finish();
}


fn bench_insert_knn(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_bench_knn");
    
    let n_set: Vec<u64> = parse_env_list("N", "1_000_000");
    let dim_set: Vec<u64> = parse_env_list("DIM", "16");
    let gd_set: Vec<usize> = parse_env_list("GD", "10");
    let m_set: Vec<usize> = parse_env_list("M", "5");

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim);
            group.throughput(criterion::Throughput::Elements(points.len() as u64));
            for &gd in &gd_set {
                for &m in &m_set {
                    group.bench_function(format!("insert_n_{}_dim_{}_gd_{}_m_{}", n, dim, gd, m),
                    |b| {
                        b.iter_batched(|| ANNIndex::new(),
                    |mut db| {
                        for point in &points {
                            db.k_insert(black_box(point), (3 * dim + 1) as usize, m);
                        }}, 
                        BatchSize::LargeInput);
                    });
                }
            }
        }
    }

    group.finish();
}

fn bench_insert_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_bench_hnsw");
    
    let n_set: Vec<u64> = parse_env_list("N", "1_000_000");
    let dim_set: Vec<u64> = parse_env_list("DIM", "16");
    let m_set: Vec<i32> = parse_env_list("GD", "32"); // number of connections i.e. m
    let ef_set: Vec<usize> = parse_env_list("M", "16"); // substitute for exploration factor

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim);
            group.throughput(criterion::Throughput::Elements(points.len() as u64));
            for &m in &m_set {
                for &ef in &ef_set {
                    group.bench_function(format!("insert_n_{}_dim_{}_gd_{}_m_{}", n, dim, m, ef),
                    |b| {
                        b.iter_batched(|| HNSWIndex::new(m),
                    |mut db| {
                        for point in &points {
                            db.insert(black_box(point), ef * 4);
                        }}, 
                        BatchSize::LargeInput);
                    });
                }
            }
        }
    }

    group.finish();
}

criterion_group!(benches, bench_insert_ann, bench_insert_knn, bench_insert_hnsw);
criterion_main!(benches);

