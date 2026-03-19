use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion, BatchSize};
use vesar::core::ann_index::ANNIndex;
use vesar::datasets::synthetic_data::generate_data;

fn parse_env_list<T>(var_name: &str) -> Vec<T> 
where 
    T: std::str::FromStr,
{
    std::env::var(var_name)
        .unwrap_or_default()
        .split(',')
        .map(|s| s.replace('_', "").trim().parse::<T>())
        .flatten() // Silently skips errors/empty strings
        .collect()
}

fn bench_insert_ann(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_bench_ann");
    
    let n_set: Vec<u64> = parse_env_list("N");
    let dim_set: Vec<u64> = parse_env_list("DIM");
    let gd_set: Vec<usize> = parse_env_list("GD");
    let m_set: Vec<usize> = parse_env_list("M");

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim);
            group.throughput(criterion::Throughput::Elements(points.len() as u64));
            for &gd in &gd_set {
                for &m in &m_set {
                    println!("insert bench (ann) set up: N={}, dim={}, graph_degree={}, multi_search={}", n, dim, gd, m);
                    
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
    
    let n_set: Vec<u64> = parse_env_list("N");
    let dim_set: Vec<u64> = parse_env_list("DIM");
    let gd_set: Vec<usize> = parse_env_list("GD");
    let m_set: Vec<usize> = parse_env_list("M");

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim);
            group.throughput(criterion::Throughput::Elements(points.len() as u64));
            for &gd in &gd_set {
                for &m in &m_set {

                    println!("insert bench (knn) set up: N={}, dim={}, graph_degree={}, multi_search={}", n, dim, gd, m);

                    group.bench_function(format!("insert_n_{}_dim_{}_gd_{}_m_{}", n, dim, gd, m),
                    |b| {
                        b.iter_batched(|| ANNIndex::new(),
                    |mut db| {
                        for point in &points {
                            db.k_insert(black_box(point), gd, m);
                        }}, 
                        BatchSize::LargeInput);
                    });
                }
            }
        }
    }

    group.finish();
}

criterion_group!(benches, bench_insert_ann, bench_insert_knn);
criterion_main!(benches);

