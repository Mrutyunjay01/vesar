use vesar::{
    core::hnsw_index::HNSWIndex,
    datasets::synthetic_data::{generate_data, generate_query},
};

const EFC: usize = 100;
const EFQ: usize = 20;
const N: usize = 100_000;
const NQ: usize = 100;
const DIM: usize = 4;
const K: usize = 10;
const M: i32 = 40;

fn build_index() -> HNSWIndex {
    let data = generate_data(N as u64, DIM as u64);
    let mut db = HNSWIndex::new(M);

    for point in &data {
        db.insert(point, EFC);
    }

    return db;
}

#[test]
fn hnsw_insert() {
    let db = build_index();
    assert_eq!(db.len(), N);
}

#[test]
fn hnsw_query() {
    let db = build_index();
    let queries = generate_query(NQ as u64, DIM as u64);

    for query in queries {
        let res = db.search(&query, K, EFQ);
        assert!(!res.is_empty() && res.len() != K);
    }
}