use std::cmp;
use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use vesar::core::{hnsw_index::HNSWIndex, nsw_index::NSWIndex};
use vesar::datasets::synthetic_data::{generate_data, generate_query};

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

fn bench_query_ann(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_bench_ann");

    let n_set: Vec<u64> = parse_env_list("N", "1_000_000");
    let dim_set: Vec<u64> = parse_env_list("DIM", "16");
    let m_set: Vec<usize> = parse_env_list("M", "10");
    let efc_set: Vec<usize> = parse_env_list("EFC", "5");
    let efq_set: Vec<usize> = parse_env_list("EFQ", "5"); // don't need for nsw query actually, it's the same

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim);
            let quries = generate_query((n as f64).sqrt() as u64, dim);

            group.throughput(criterion::Throughput::Elements(quries.len() as u64));
            for &m in &m_set {
                for &efc in &efc_set {
                    let mut db = NSWIndex::new();
                    for point in &points {
                        db.insert(point, m, efc);
                    }

                    for &efq in &efq_set {
                        group.bench_function(format!("query_n_{}_dim_{}_m_{}_efc_{}_efq_{}_k_1", n, dim, m, efc, efq), // here k is 1
                        |b| {
                            b.iter(|| {
                            for query in &quries {
                                black_box(db.multi_search(query, efq));
                            }});
                        });
                    }
            }
        }
    }}

    group.finish();
}

fn bench_query_knn(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_bench_knn");

    let n_set: Vec<u64> = parse_env_list("N", "1_000_000");
    let dim_set: Vec<u64> = parse_env_list("DIM", "16");
    let m_set: Vec<usize> = parse_env_list("M", "10");
    let efc_set: Vec<usize> = parse_env_list("EFC", "5");
    let efq_set: Vec<usize> = parse_env_list("EFQ", "5");
    let k_set: Vec<usize> = parse_env_list("K", "10");

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim);
            let quries = generate_query((n as f64).sqrt() as u64, dim);

            group.throughput(criterion::Throughput::Elements(quries.len() as u64));
            for &m in &m_set {
                for &efc in &efc_set {
                    let mut db = NSWIndex::new();
                    for point in &points {
                        db.k_insert(point, m as usize, efc);
                    }

                    for &efq in &efq_set {
                        for &k in &k_set {
                            group.bench_function(format!("query_n_{}_dim_{}_m_{}_efc_{}_efq_{}_k_{}", n, dim, m, efc, efq, k), 
                            |b| {
                                b.iter(|| {
                                for query in &quries {
                                    black_box(db.k_multi_search(query, efq, k));
                                }});
                            });
                        }
                    }
                }
            }
        }
    }

    group.finish();
}

fn bench_query_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_bench_hnsw");

    let n_set: Vec<u64> = parse_env_list("N", "1_000_000");
    let dim_set: Vec<u64> = parse_env_list("DIM", "16");
    let m_set: Vec<i32> = parse_env_list("M", "16"); // number of connections i.e. m
    let efc_set: Vec<usize> = parse_env_list("EFC", "100"); // substitute for exploration factor
    let efq_set: Vec<usize> = parse_env_list("EFQ", "10");
    let k_set: Vec<usize> = parse_env_list("K", "10");

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim);
            let quries = generate_query((n as f64).sqrt() as u64, dim);

            group.throughput(criterion::Throughput::Elements(quries.len() as u64));
            for &m in &m_set {
                let m_insert = cmp::max(m, 2 * dim as i32);
                for &efc in &efc_set {
                    let efc_insert = cmp::max(efc, 5 * dim as usize);
                    let mut db = HNSWIndex::new(m_insert);
                    for point in &points {
                        db.insert(point, efc_insert);
                    }

                    for &efq in &efq_set {
                        let efq_query = cmp::max(efq, 2 * dim as usize);
                        for &k in &k_set {
                            // warm up
                            for query in quries.iter().take(50) {
                                db.search(query, k, efq_query);
                            }

                            group.bench_function(format!("query_n_{}_dim_{}_m_{}_efc_{}_efq_{}_k_{}", n, dim, m_insert, efc_insert, efq_query, k),
                            |b| {
                                b.iter(|| {
                                for query in &quries {
                                    black_box(db.search(query, k, efq_query));
                                }});
                            });
                        }
                    }
            }
        }
    }}

    group.finish();
}

criterion_group!(benches, bench_query_ann, bench_query_knn, bench_query_hnsw);
// criterion_group!(benches, bench_query_hnsw);
criterion_main!(benches);