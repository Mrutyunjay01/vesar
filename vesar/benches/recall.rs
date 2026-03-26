use std::collections::HashSet;
use std::time::{Duration, Instant};
use std::hint::black_box;
use vesar::core::hnsw_index::HNSWIndex;
use vesar::{core::ann_index::{ANNIndex}, datasets::synthetic_data::{generate_data, generate_query}};

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
    let gd_set: Vec<u64> = parse_env_list("GD", "10");
    let m_set: Vec<u64> = parse_env_list("M", "5");
    let k_set: Vec<u64> = parse_env_list("K", "10");
    let iters = std::env::var("ITERS").unwrap_or(String::from("10")).parse::<usize>().unwrap();

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim);
            let queries = generate_query((n as f64).sqrt() as u64, dim);

            for &gd in &gd_set {
                for &m in &m_set {
                    let mut db = ANNIndex::new();
                    for point in &points { db.insert(point, gd as usize, m as usize); }

                    for &k in &k_set {
                        let mut gt_top_k = Vec::with_capacity(queries.len());
                        for query in &queries {
                            gt_top_k.push(db.bruteforce_top_k(query, k as usize));
                        }

                        // warm up
                        for query in queries.iter().take(100) {
                            black_box(db.multi_search(query, m as usize));
                        }

                        let mut total_duration_ann = Duration::ZERO;
                        let mut total_recall_ann = 0.0;

                        for _ in 0..iters {
                            let mut iter_recall_ann: f64 = 0.0;
                            let start_ann = Instant::now();
                            for (query_idx, query) in queries.iter().enumerate() {
                                let results = black_box(db.multi_search(query, m as usize));
                                iter_recall_ann += calculate_recall_k(&gt_top_k[query_idx], &[results]);
                            }
                            total_duration_ann += start_ann.elapsed();
                            total_recall_ann += iter_recall_ann as f64 / queries.len() as f64;
                        }

                        let bench_name = format!("n_{}_dim_{}_gd_{}_m_{}_k_{}", n, dim, gd, m, k);
                        let avg_recall_ann = total_recall_ann / iters as f64;
                        
                        let total_queries = (queries.len() * iters) as f64;
                        let qps_ann = total_queries / total_duration_ann.as_secs_f64();
                        let avg_time_ms_ann = (total_duration_ann.as_secs_f64() * 1000.0) / total_queries;

                        println!(
                            "{:<35} | RECALL_ANN: {:.4} | QPS_ANN: {:>10.0} | AVG LATENCY (ANN): {:.4}ms", 
                            bench_name, avg_recall_ann, qps_ann, avg_time_ms_ann
                        );
                    }
                }
            }
        }
    }
}

fn recall_bench_knn() {
    let n_set: Vec<u64> = parse_env_list("N", "1000000");
    let dim_set: Vec<u64> = parse_env_list("DIM", "16");
    let gd_set: Vec<u64> = parse_env_list("GD", "10");
    let m_set: Vec<u64> = parse_env_list("M", "5");
    let k_set: Vec<u64> = parse_env_list("K", "10");
    let iters = std::env::var("ITERS").unwrap_or(String::from("10")).parse::<usize>().unwrap();

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim);
            let queries = generate_query((n as f64).sqrt() as u64, dim);

            for &gd in &gd_set {
                for &m in &m_set {
                    let mut db = ANNIndex::new();
                    for point in &points { db.k_insert(point, (3 * dim + 1) as usize, m as usize); }

                    for &k in &k_set {
                        let mut gt_top_k = Vec::with_capacity(queries.len());
                        for query in &queries {
                            gt_top_k.push(db.bruteforce_top_k(query, k as usize));
                        }

                        // warm up
                        for query in queries.iter().take(100) {
                            black_box(db.k_multi_search(query, m as usize, k as usize));
                        }

                        let mut total_duration_knn = Duration::ZERO;
                        let mut total_recall_knn = 0.0;

                        for _ in 0..iters {
                            let mut iter_recall_knn = 0.0;
                            let start_knn = Instant::now();
                            for (query_idx, query) in queries.iter().enumerate() {
                                let results = black_box(db.k_multi_search(query, m as usize, k as usize));
                                iter_recall_knn += calculate_recall_k(&gt_top_k[query_idx], &results);
                            }
                            total_duration_knn += start_knn.elapsed();
                            total_recall_knn += iter_recall_knn as f64 / queries.len() as f64;
                        }

                        let bench_name = format!("n_{}_dim_{}_gd_{}_m_{}_k_{}", n, dim, gd, m, k);
                        let avg_recall_knn = total_recall_knn / iters as f64;
                        
                        let total_queries = (queries.len() * iters) as f64;
                        let qps_knn = total_queries / total_duration_knn.as_secs_f64();
                        let avg_time_ms_knn = (total_duration_knn.as_secs_f64() * 1000.0) / total_queries;

                        println!(
                            "{:<35} | RECALL_KNN: {:.4} | QPS_KNN: {:>10.0} | AVG LATENCY (KNN): {:.4}ms", 
                            bench_name, avg_recall_knn, qps_knn, avg_time_ms_knn
                        );
                    }
                }
            }
        }
    }
}


fn recall_bench_hnsw() {
    let n_set: Vec<u64> = parse_env_list("N", "1000000");
    let dim_set: Vec<u64> = parse_env_list("DIM", "16");
    let m_set: Vec<u64> = parse_env_list("GD", "10");
    let ef_set: Vec<u64> = parse_env_list("M", "5");
    let k_set: Vec<u64> = parse_env_list("K", "10");
    let iters = std::env::var("ITERS").unwrap_or(String::from("10")).parse::<usize>().unwrap();

    for &n in &n_set {
        for &dim in &dim_set {
            let points = generate_data(n, dim);
            let queries = generate_query((n as f64).sqrt() as u64, dim);

            for &m in &m_set {
                for &ef in &ef_set {
                    let mut db = HNSWIndex::new(m as i32);
                    for point in &points { db.insert(point, (ef*ef) as usize); }

                    for &k in &k_set {
                        let mut gt_top_k = Vec::with_capacity(queries.len());
                        for query in &queries {
                            gt_top_k.push(db.bruteforce_top_k(query, k as usize));
                        }

                        // warm up
                        for query in queries.iter().take(100) {
                            black_box(db.search(query, k as usize, ef as usize));
                        }

                        let mut total_duration_knn = Duration::ZERO;
                        let mut total_recall_knn = 0.0;

                        for _ in 0..iters {
                            let mut iter_recall_knn = 0.0;
                            let start_knn = Instant::now();
                            for (query_idx, query) in queries.iter().enumerate() {
                                let results = black_box(db.search(query, k as usize, ef as usize));
                                iter_recall_knn += calculate_recall_k(&gt_top_k[query_idx], &results);
                            }
                            total_duration_knn += start_knn.elapsed();
                            total_recall_knn += iter_recall_knn as f64 / queries.len() as f64;
                        }

                        let bench_name = format!("n_{}_dim_{}_gd_{}_m_{}_k_{}", n, dim, m, ef, k);
                        let avg_recall_knn = total_recall_knn / iters as f64;
                        
                        let total_queries = (queries.len() * iters) as f64;
                        let qps_knn = total_queries / total_duration_knn.as_secs_f64();
                        let avg_time_ms_knn = (total_duration_knn.as_secs_f64() * 1000.0) / total_queries;

                        println!(
                            "{:<35} | RECALL_KNN: {:.4} | QPS_KNN: {:>10.0} | AVG LATENCY (KNN): {:.4}ms", 
                            bench_name, avg_recall_knn, qps_knn, avg_time_ms_knn
                        );
                    }
                }
            }
        }
    }
}

fn main() {
    // recall_bench_ann();
    // recall_bench_knn();
    recall_bench_hnsw();
}