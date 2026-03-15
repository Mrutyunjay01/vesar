use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion, BatchSize};
use vesar::core::ann_index::ANNIndex;
use vesar::datasets::synthetic_data::generate_data;

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_bench");
    
    let n_set = [10_000, 100_000, 1_000_000]; // start with 1M insertions, will benchmark for scale later
    let dim_set = [2, 8, 16, 32];
    let k_set = [1, 5, 10];
    let m_set = [1, 3, 5];

    for n in n_set {
        for dim in dim_set {
            let points = generate_data(n, dim);
            group.throughput(criterion::Throughput::Elements(points.len() as u64));
            for k in k_set {
                for m in m_set {
                    
                    group.bench_function(format!("insert_n_{}_dim_{}_k_{}_m_{}", n, dim, k, m),
                    |b| {
                        b.iter_batched(|| ANNIndex::new(),
                    |mut db| {
                        for point in &points {
                            db.insert(black_box(point), k, m);
                        }}, 
                        BatchSize::LargeInput);
                    });
                }
            }
        }
    }

    group.finish();
}

criterion_group!(benches, bench_insert);
criterion_main!(benches);

