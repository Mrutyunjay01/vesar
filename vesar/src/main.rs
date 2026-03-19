use vesar::core::ann_index::ANNIndex;
use vesar::datasets::synthetic_data;

fn main() {
    let mut db = ANNIndex::new();

    let k = 5;
    let m = 5;
    let dim = 2;
    let n_points = 5;
    let n_tests: u64 = 2;

    println!("---- Inserting random points ----");

    let points: Vec<Vec<f32>> = synthetic_data::generate_data(n_points, dim);
    for point in points {
        db.k_insert(&point, k, m);
    }

    println!("Inserted {} nodes", db.nodes.len());

    println!("---- Running search tests ----");
    let queries: Vec<Vec<f32>> = synthetic_data::generate_query(n_tests, dim);

    for (test_id, query) in queries.iter().enumerate() {
        println!("\nTest {}: Query {:?}", test_id + 1, query);

        let greedy_result = db.greedy_search(&query, 0);
        let multi_result = db.multi_search(&query, m);
        let k_results = db.k_multi_search(&query, m, k);

        let a: Vec<String> = k_results.iter().map(|&ele| ele.to_string()).collect();

        println!("Greedy result node: {}", greedy_result);
        println!("Multi-search result node: {}", multi_result);
        println!("kNN Search Result Nodes: {}", a.join(","));

        // brute force nearest neighbour
        let bruteforce_result = db.bruteforce_nn(query);
        println!("Brute force nearest node: {}", bruteforce_result);

        if bruteforce_result == greedy_result {
            println!("Greedy search: EXACT");
        } else {
            println!("Greedy search: APPROX");
        }

        if bruteforce_result == multi_result {
            println!("Multi-search: EXACT");
        } else {
            println!("Multi-search: APPROX");
        }
    }

    println!("\n---- Graph statistics ----");

    let mut total_edges = 0;

    for node in &db.nodes {
        total_edges += node.neighbours.len();
    }

    let avg_degree = total_edges as f32 / db.nodes.len() as f32;

    println!("Total nodes: {}", db.nodes.len());
    println!("Total edges (directed): {}", total_edges);
    println!("Average node degree: {:.2}", avg_degree);
}
