use vesar::{
    core::nsw_index::NSWIndex,
    datasets::synthetic_data::{generate_data, generate_query},
};

const EXPLORATION_FACTOR: usize = 5;
const N: usize = 1000;
const NQ: usize = 100;
const DIM: usize = 16;
const K: usize = 10;
const M: usize = 10;

fn build_index(use_k_insert: bool) -> NSWIndex {
    let data = generate_data(N as u64, DIM as u64);
    let mut db = NSWIndex::new();

    for point in &data {
        if !use_k_insert {
            db.insert(point, M, EXPLORATION_FACTOR);
        } else {
            db.k_insert(point, M, EXPLORATION_FACTOR);
        }
    }

    return db;
}

#[test]
fn nsw_insert() {
    let db = build_index(false);
    assert_eq!(db.len(), N);
}


#[test]
fn nsw_k_insert() {
    let db = build_index(true);
    assert_eq!(db.len(), N);
}

#[test]
fn nsw_multi_search_query() {
    let db = build_index(true);
    let queries = generate_query(NQ as u64, DIM as u64);

    for query in queries {
        let _ = db.multi_search(&query, EXPLORATION_FACTOR);
    }
}

#[test]
fn nsw_knn_query() {
    let db = build_index(true);
    let queries = generate_query(NQ as u64, DIM as u64);

    for query in queries {
        let res = db.k_multi_search(&query, EXPLORATION_FACTOR, K);
        assert!(!res.is_empty() && res.len() != K);
    }
}