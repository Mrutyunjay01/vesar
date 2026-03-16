use std::time::{Duration, Instant};
use std::hint::black_box;
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

fn recall_bench() {
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

                        for query in queries.iter().take(100) {
                            black_box(db.multi_search(query, m as usize));
                        }

                        let mut total_duration = Duration::ZERO;
                        let mut total_recall = 0.0;

                        for _ in 0..iters {
                            let mut correct_match = 0;
                            
                            let start = Instant::now();
                            for (query_idx, query) in queries.iter().enumerate() {
                                let result = black_box(db.multi_search(query, m as usize));
                                
                                if gt_top_k[query_idx].contains(&result) {
                                    correct_match += 1;
                                }
                            }
                            total_duration += start.elapsed();
                            total_recall += correct_match as f64 / queries.len() as f64;
                        }

                        let bench_name = format!("n_{}_dim_{}_gd_{}_m_{}_k_{}", n, dim, gd, m, k);
                        let avg_recall = total_recall / iters as f64;
                        
                        let total_queries = (queries.len() * iters) as f64;
                        let qps = total_queries / total_duration.as_secs_f64();
                        let avg_time_ms = (total_duration.as_secs_f64() * 1000.0) / total_queries;

                        println!(
                            "{:<35} | RECALL: {:.4} | QPS: {:>10.0} | AVG LATENCY: {:.4}ms", 
                            bench_name, avg_recall, qps, avg_time_ms
                        );
                    }
                }
            }
        }
    }
}

fn main() {
    recall_bench();
}