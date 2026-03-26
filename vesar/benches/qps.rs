use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use vesar::core::ann_index::ANNIndex;
use vesar::core::hnsw_index::HNSWIndex;
use vesar::datasets::synthetic_data::{generate_data, generate_query};

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

fn bench_query_ann(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_bench_ann");

    let n_set: Vec<u64> = parse_env_list("N");
    let dim_set: Vec<u64> = parse_env_list("DIM");
    let gd_set: Vec<usize> = parse_env_list("GD");
    let m_set: Vec<usize> = parse_env_list("M");

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim);
            let quries = generate_query((n as f64).sqrt() as u64, dim);

            group.throughput(criterion::Throughput::Elements(quries.len() as u64));
            for &gd in &gd_set {
                for &m in &m_set {
                    let mut db = ANNIndex::new();
                    for point in &points {
                        db.insert(point, gd, m);
                    }

                    group.bench_function(format!("query_n_{}_dim_{}_gd_{}_m_{}", n, dim, gd, m), 
                    |b| {
                        b.iter(|| {
                        for query in &quries {
                            black_box(db.multi_search(query, m));
                        }});
                    });
            }
        }
    }}

    group.finish();
}

fn bench_query_knn(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_bench_knn");

    let n_set: Vec<u64> = parse_env_list("N");
    let dim_set: Vec<u64> = parse_env_list("DIM");
    let gd_set: Vec<usize> = parse_env_list("GD");
    let m_set: Vec<usize> = parse_env_list("M");

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim);
            let quries = generate_query((n as f64).sqrt() as u64, dim);

            group.throughput(criterion::Throughput::Elements(quries.len() as u64));
            for &gd in &gd_set {
                for &m in &m_set {
                    let mut db = ANNIndex::new();
                    for point in &points {
                        db.k_insert(point, (3 * dim + 1) as usize, m);
                    }

                    group.bench_function(format!("query_n_{}_dim_{}_gd_{}_m_{}", n, dim, gd, m), 
                    |b| {
                        b.iter(|| {
                        for query in &quries {
                            black_box(db.k_multi_search(query, m, 10));
                        }});
                    });
            }
        }
    }}

    group.finish();
}

fn bench_query_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_bench_hnsw");

    let n_set: Vec<u64> = parse_env_list("N");
    let dim_set: Vec<u64> = parse_env_list("DIM");
    let m_set: Vec<usize> = parse_env_list("GD");
    let ef_set: Vec<usize> = parse_env_list("M");

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim);
            let quries = generate_query((n as f64).sqrt() as u64, dim);

            group.throughput(criterion::Throughput::Elements(quries.len() as u64));
            for &m in &m_set {
                for &ef in &ef_set {
                    let mut db = HNSWIndex::new(m as i32);
                    for point in &points {
                        db.insert(point, ef);
                    }

                    group.bench_function(format!("query_n_{}_dim_{}_gd_{}_m_{}", n, dim, m, ef), 
                    |b| {
                        b.iter(|| {
                        for query in &quries {
                            black_box(db.search(query, 10, ef));
                        }});
                    });
            }
        }
    }}

    group.finish();
}

// criterion_group!(benches, bench_query_ann, bench_query_knn);
criterion_group!(benches, bench_query_hnsw);
criterion_main!(benches);