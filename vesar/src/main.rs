use vesar::core::hnsw_index::HNSWIndex;
use vesar::datasets::synthetic_data;

fn main() {
    // let mut db = ANNIndex::new();
    let mut db = HNSWIndex::new(16);

    let k = 5;
    let m = 10; // exploration factor
    let dim = 4;
    let n_points = 100_000;
    let n_tests: u64 = 1000;

    println!("---- Inserting random points ----");

    let points: Vec<Vec<f32>> = synthetic_data::generate_data(n_points, dim);
    for point in points {
        // db.k_insert(&point, k, m);
        db.insert(&point, m);
    }

    // println!("Inserted {} nodes", db.nodes.len());
    println!("Inserted {} nodes", db.len());

    println!("---- Running search tests ----");
    let queries: Vec<Vec<f32>> = synthetic_data::generate_query(n_tests, dim);

    let mut total_recall: f64 = 0.0;

    for (test_id, query) in queries.iter().enumerate() {
        println!("\nTest {}: Query {:?}", test_id + 1, query);

        // let greedy_result = db.greedy_search(&query, 0);
        // let multi_result = db.multi_search(&query, m);
        // let k_results = db.k_multi_search(&query, m, k);
        let k_results = db.search(query, k, m);

        let a: Vec<String> = k_results.iter().map(|&ele| ele.to_string()).collect();

        // println!("Greedy result node: {}", greedy_result);
        // println!("Multi-search result node: {}", multi_result);
        println!("kNN Search Result Nodes: {}", a.join(","));

        // brute force nearest neighbour
        // let bruteforce_result = db.bruteforce_nn(query);
        let bruteforce_result = db.bruteforce_top_k(query, k);
        let b: Vec<String> = bruteforce_result.iter().map(|ele| ele.to_string()).collect();
        println!("Brute force nearest node: {}", b.join(","));

        // calculate recall
        let mut count = 0;
        for res in k_results {
            if bruteforce_result.contains(&res) {
                count += 1;
            }
        }
        let recall: f64 = count as f64 / bruteforce_result.len() as f64;
        total_recall += recall/queries.len() as f64;
        println!("recall: {}", recall);

        // if bruteforce_result == greedy_result {
        //     println!("Greedy search: EXACT");
        // } else {
        //     println!("Greedy search: APPROX");
        // }

        // if bruteforce_result == multi_result {
        //     println!("Multi-search: EXACT");
        // } else {
        //     println!("Multi-search: APPROX");
        // }
    }

    println!("total recall: {:.2}", total_recall);

    // println!("\n---- Graph statistics ----");

    // let mut total_edges = 0;

    // for node in &db.nodes {
    //     total_edges += node.neighbours.len();
    // }

    // let avg_degree = total_edges as f32 / db.nodes.len() as f32;

    // println!("Total nodes: {}", db.nodes.len());
    // println!("Total edges (directed): {}", total_edges);
    // println!("Average node degree: {:.2}", avg_degree);
}
