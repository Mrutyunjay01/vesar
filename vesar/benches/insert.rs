use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion, BatchSize};
use vesar::core::ann_index::ANNIndex;
use vesar::datasets::synthetic_data::generate_data;

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_bench");

    for _p in [100, 10_000, 100_000] {
        let n = _p; // 10u64.pow(_p);

        for _d in [2, 16, 32] {
            let dim = _d; // 2u64.pow(_d);

            let points = generate_data(n, dim);
            for k in [1, 5, 10] {
                for m in [1, 3, 5] {
                    
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

