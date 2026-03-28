use std::cmp;
use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion, BatchSize};
use vesar::core::{hnsw_index::HNSWIndex, nsw_index::NSWIndex};
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
    let m_set: Vec<usize> = parse_env_list("M", "10");
    let efc_set: Vec<usize> = parse_env_list("EFC", "5");

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim);
            group.throughput(criterion::Throughput::Elements(points.len() as u64));
            for &m in &m_set {
                for &efc in &efc_set {
                    group.bench_function(format!("insert_n_{}_dim_{}_m_{}_efc_{}", n, dim, m, efc),
                    |b| {
                        b.iter_batched(|| NSWIndex::new(),
                    |mut db| {
                        for point in &points {
                            db.insert(black_box(point), m, efc);
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
    let m_set: Vec<usize> = parse_env_list("M", "10");
    let efc_set: Vec<usize> = parse_env_list("EFC", "5");

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim);
            group.throughput(criterion::Throughput::Elements(points.len() as u64));
            for &m in &m_set {
                let m_insert = cmp::max(3 * dim as usize + 1, m);
                for &efc in &efc_set {
                    group.bench_function(format!("insert_n_{}_dim_{}_m_{}_efc_{}", n, dim, m_insert, efc),
                    |b| {
                        b.iter_batched(|| NSWIndex::new(),
                    |mut db| {
                        for point in &points {
                            db.k_insert(black_box(point), m_insert as usize, efc);
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
    let dim_set: Vec<usize> = parse_env_list("DIM", "16");
    let m_set: Vec<usize> = parse_env_list("M", "16"); // number of connections i.e. m
    let efc_set: Vec<usize> = parse_env_list("EFC", "16"); // substitute for exploration factor

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim as u64);
            group.throughput(criterion::Throughput::Elements(points.len() as u64));
            for &m in &m_set {
                let m_insert = cmp::max(m, 2 * dim);
                for &efc in &efc_set {
                    let efc_insert = cmp::max(efc, 5 * dim);
                    group.bench_function(format!("insert_n_{}_dim_{}_m_{}_efc_{}", n, dim, m_insert, efc_insert),
                    |b| {
                        b.iter_batched(|| HNSWIndex::new(m_insert as i32),
                    |mut db| {
                        for point in &points {
                            db.insert(black_box(point), efc_insert);
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
// criterion_group!(benches, bench_insert_hnsw);
criterion_main!(benches);

