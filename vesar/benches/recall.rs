use std::cmp;
use std::collections::HashSet;
use std::time::{Duration, Instant};
use std::hint::black_box;
use vesar::core::hnsw_index::HNSWIndex;
use vesar::{core::nsw_index::{NSWIndex}, datasets::synthetic_data::{generate_data, generate_query}};

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

fn calculate_recall_k(gt: &[usize], results: &[usize]) -> f64 {
    let gt_set: HashSet<_> = gt.iter().collect();

    let mut correct_match = 0;
    for result in results {
        if gt_set.contains(result) {
            correct_match += 1;
        }
    }

    return correct_match as f64 / results.len() as f64;
}

fn recall_bench_ann() {
    let n_set: Vec<u64> = parse_env_list("N", "1000000");
    let dim_set: Vec<u64> = parse_env_list("DIM", "16");
    let m_set: Vec<usize> = parse_env_list("M", "10");
    let efc_set: Vec<usize> = parse_env_list("EFC", "5");
    let efq_set: Vec<usize> = parse_env_list("EFQ", "5");
    let k_set: Vec<u64> = parse_env_list("K", "10");
    let iters = std::env::var("ITERS").unwrap_or(String::from("10")).parse::<usize>().unwrap();

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim);
            let queries = generate_query((n as f64).sqrt() as u64, dim);

            for &m in &m_set {
                for &efc in &efc_set {
                    let mut db = NSWIndex::new();
                    let mut total_duration_insert = Duration::ZERO;
                    let start_insert = Instant::now();
                    for point in &points {
                        db.insert(point, m, efc as usize);
                    }
                    total_duration_insert += start_insert.elapsed();

                    for &k in &k_set {
                        let mut gt_top_k = Vec::with_capacity(queries.len());
                        for query in &queries {
                            gt_top_k.push(db.bruteforce_top_k(query, k as usize));
                        }

                        for &efq in &efq_set {
                            // warm up
                            for query in queries.iter().take(100) {
                                black_box(db.multi_search(query, efq));
                            }

                            let mut total_duration_ann = Duration::ZERO;
                            let mut total_recall_ann = 0.0;

                            for _ in 0..iters {
                                let mut iter_recall_ann: f64 = 0.0;
                                let start_ann = Instant::now();
                                for (query_idx, query) in queries.iter().enumerate() {
                                    let results = black_box(db.multi_search(query, efq));
                                    iter_recall_ann += calculate_recall_k(&gt_top_k[query_idx], &[results]);
                                }
                                total_duration_ann += start_ann.elapsed();
                                total_recall_ann += iter_recall_ann as f64 / queries.len() as f64;
                            }

                            let bench_name = format!("n_{}_dim_{}_m_{}_efc_{}_efq_{}_k_{}", n, dim, m, efc, efq, k);
                            let avg_recall_ann = total_recall_ann / iters as f64;
                            
                            let total_queries = (queries.len() * iters) as f64;
                            let qps_ann = total_queries / total_duration_ann.as_secs_f64();
                            let avg_time_ms_ann = (total_duration_ann.as_secs_f64() * 1000.0) / total_queries;
                            let avg_time_ms_insert = (total_duration_insert.as_secs_f64() * 1000.0) / points.len() as f64;

                            println!(
                                "{:<40} | {:<15} {:<6.4} | {:<11} {:<6.0} | {:<22} {:<6.4}ms | {:<22} {:<6.4}ms",
                                bench_name,
                                "RECALL (ANN)", avg_recall_ann,
                                "QPS (ANN)", qps_ann,
                                "QUERY_LATENCY (ANN)", avg_time_ms_ann,
                                "INSERT_LATENCY (ANN)", avg_time_ms_insert
                            );
                        }
                    }
                }
            }
        }
    }
}

fn recall_bench_knn() {
    let n_set: Vec<u64> = parse_env_list("N", "1_000_000");
    let dim_set: Vec<usize> = parse_env_list("DIM", "16");
    let m_set: Vec<usize> = parse_env_list("M", "10");
    let efc_set: Vec<usize> = parse_env_list("EFC", "5");
    let efq_set: Vec<usize> = parse_env_list("EFQ", "5");
    let k_set: Vec<u64> = parse_env_list("K", "10");
    let iters = std::env::var("ITERS").unwrap_or(String::from("10")).parse::<usize>().unwrap();

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim as u64);
            let queries = generate_query((n as f64).sqrt() as u64, dim as u64);

            for &m in &m_set {
                let m_insert = cmp::max(m, 3 * dim + 1);
                for &efc in &efc_set {
                    let mut db = NSWIndex::new();
                    for point in &points { db.k_insert(point, m_insert, efc); }

                    for &k in &k_set {
                        let mut gt_top_k = Vec::with_capacity(queries.len());
                        let mut total_duration_insert = Duration::ZERO;
                        let start_insert = Instant::now();
                        for query in &queries {
                            gt_top_k.push(db.bruteforce_top_k(query, k as usize));
                        }
                        total_duration_insert += start_insert.elapsed();

                        for &efq in &efq_set {
                            // warm up
                            for query in queries.iter().take(100) {
                                black_box(db.k_multi_search(query, efq, k as usize));
                            }

                            let mut total_duration_knn = Duration::ZERO;
                            let mut total_recall_knn = 0.0;

                            for _ in 0..iters {
                                let mut iter_recall_knn = 0.0;
                                let start_knn = Instant::now();
                                for (query_idx, query) in queries.iter().enumerate() {
                                    let results = black_box(db.k_multi_search(query, efq, k as usize));
                                    iter_recall_knn += calculate_recall_k(&gt_top_k[query_idx], &results);
                                }
                                total_duration_knn += start_knn.elapsed();
                                total_recall_knn += iter_recall_knn as f64 / queries.len() as f64;
                            }

                            let bench_name = format!("n_{}_dim_{}_m_{}_efc_{}_efq_{}_k_{}", n, dim, m_insert, efc, efq, k);
                            let avg_recall_knn = total_recall_knn / iters as f64;
                            
                            let total_queries = (queries.len() * iters) as f64;
                            let qps_knn = total_queries / total_duration_knn.as_secs_f64();
                            let avg_time_ms_knn = (total_duration_knn.as_secs_f64() * 1000.0) / total_queries;
                            let avg_time_ms_insert = (total_duration_insert.as_secs_f64() * 1000.0) / points.len() as f64;

                            println!(
                                "{:<40} | {:<15} {:<6.4} | {:<11} {:<6.0} | {:<22} {:<6.4}ms | {:<22} {:<6.4}ms",
                                bench_name,
                                "RECALL (KNN)", avg_recall_knn,
                                "QPS (KNN)", qps_knn,
                                "QUERY_LATENCY (KNN)", avg_time_ms_knn,
                                "INSERT_LATENCY (KNN)", avg_time_ms_insert
                            );
                        }
                    }
                }
            }
        }
    }
}


fn recall_bench_hnsw() {
    let n_set: Vec<u64> = parse_env_list("N", "1_000_000");
    let dim_set: Vec<u64> = parse_env_list("DIM", "16");
    let m_set: Vec<u64> = parse_env_list("M", "16"); // connections
    let efc_set: Vec<usize> = parse_env_list("EFC", "10"); // exploration factors
    let efq_set: Vec<usize> = parse_env_list("EFQ", "10"); // exploration factors
    let k_set: Vec<u64> = parse_env_list("K", "5");
    let iters = std::env::var("ITERS").unwrap_or(String::from("10")).parse::<usize>().unwrap();

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim);
            let queries = generate_query((n as f64).sqrt() as u64, dim);

            for &m in &m_set { // allowed connections per node
                let m_insert = cmp::max(m, 2 * dim as u64);
                for &efc in &efc_set { // exploration factor per layer
                    let mut db = HNSWIndex::new(m_insert as i32);
                    let mut total_duration_insert = Duration::ZERO;
                    let efc_insert = cmp::max(efc, 5 * dim as usize);
                    let start_insert = Instant::now();
                    for point in &points { 
                        db.insert(point, efc_insert); 
                    }
                    total_duration_insert += start_insert.elapsed();

                    for &k in &k_set {
                        let mut gt_top_k = Vec::with_capacity(queries.len());
                        for query in &queries {
                            gt_top_k.push(db.bruteforce_top_k(query, k as usize));
                        }

                        for &efq in &efq_set {
                            let efq_query = cmp::max(efq, 2 * dim as usize);
                            // warm up
                            for query in queries.iter().take(100) {
                                black_box(db.search(query, k as usize, efq_query));
                            }

                            let mut total_duration_knn = Duration::ZERO;
                            let mut total_recall_knn = 0.0;

                            for _ in 0..iters {
                                let mut iter_recall_knn = 0.0;
                                let start_knn = Instant::now();
                                let all_results: Vec<_> = queries.iter().map(
                                    |query| black_box(db.search(query,  k as usize, efq_query))).collect();
                                total_duration_knn += start_knn.elapsed();
                                
                                for (query_idx, res) in all_results.iter().enumerate() {
                                    assert!(res.len() == gt_top_k[query_idx].len(), "mismatching lengths between results and ground truth");
                                    iter_recall_knn += calculate_recall_k(&gt_top_k[query_idx], &res);
                                }
                                total_recall_knn += iter_recall_knn as f64 / queries.len() as f64;
                            }

                            let bench_name = format!("n_{}_dim_{}_m_{}_efc_{}_efq_{}_k_{}", n, dim, m_insert, efc_insert, efq_query, k);
                            let avg_recall_knn = total_recall_knn / iters as f64;
                            
                            let total_queries = (queries.len() * iters) as f64;
                            let qps_knn = total_queries / total_duration_knn.as_secs_f64();
                            let avg_time_ms_knn = (total_duration_knn.as_secs_f64() * 1000.0) / total_queries;
                            let avg_time_ms_insert = (total_duration_insert.as_secs_f64() * 1000.0) / points.len() as f64;

                            println!(
                                "{:<40} | {:<15} {:<6.4} | {:<11} {:<6.0} | {:<22} {:<6.4}ms | {:<22} {:<6.4}ms",
                                bench_name,
                                "RECALL (HNSW)", avg_recall_knn,
                                "QPS (HNSW)", qps_knn,
                                "QUERY_LATENCY (HNSW)", avg_time_ms_knn,
                                "INSERT_LATENCY (HNSW)", avg_time_ms_insert
                            );
                        }
                    }
                }
            }
        }
    }
}

fn main() {
    recall_bench_ann();
    recall_bench_knn();
    recall_bench_hnsw();
}