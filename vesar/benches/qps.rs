use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use vesar::core::ann_index::ANNIndex;
use vesar::datasets::synthetic_data::{generate_data, generate_query};

fn bench_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_bench");

    let n_set = [1_000_000]; // start with 1M insertions, will benchmark for scale later
    let dim_set = [16];
    let k_set = [1, 5];
    let m_set = [5];

    for n in n_set {
        
        for dim in dim_set {

            let points = generate_data(n, dim);
            let quries = generate_query((n as f64).sqrt() as u64, dim);

            group.throughput(criterion::Throughput::Elements(quries.len() as u64));
            for k in k_set {
                for m in m_set {
                    
                    group.bench_function(format!("query_n_{}_dim_{}_k_{}_m_{}", n, dim, k, m), 
                    |b| {
                        b.iter_batched(|| {
                            let mut db = ANNIndex::new();
                            for point in &points {
                                db.insert(black_box(point), k, m);
                            }

                            db
                        },
                    |db| {
                        for query in &quries {
                            db.multi_search(query, m);
                        }}, 
                        BatchSize::LargeInput);
                    });
                }
            }
        }
    }

    group.finish();
}

criterion_group!(benches, bench_query);
criterion_main!(benches);