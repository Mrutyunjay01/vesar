use rand::rng;
use rand::RngExt;
use vesar::core::ann_index::ANNIndex;
use vesar::metrics::l2::l2;

fn main() {
    let mut db = ANNIndex::new();

    let k = 5;
    let m = 5;
    let dim = 2;
    let n_points = 1000000;

    let mut rng = rng();

    println!("---- Inserting random points ----");

    for _ in 0..n_points {
        let point: Vec<f32> = (0..dim)
            .map(|_| rng.random_range(0.0..100.0))
            .collect();

        db.insert(&point, k, m);
    }

    println!("Inserted {} nodes", db.nodes.len());

    println!("---- Running search tests ----");

    for test_id in 0..10 {

        let query: Vec<f32> = (0..dim)
            .map(|_| rng.random_range(0.0..100.0))
            .collect();

        println!("\nTest {}: Query {:?}", test_id + 1, query);

        let greedy_result = db.greedy_search(&query, 0);
        let multi_result = db.multi_search(&query, m);

        println!("Greedy result node: {}", greedy_result);
        println!("Multi-search result node: {}", multi_result);

        // brute force nearest neighbour
        let mut best_node = 0;
        let mut best_dist = l2(&query, &db.nodes[0].value);

        for node in &db.nodes {
            let dist = l2(&query, &node.value);
            if dist < best_dist {
                best_dist = dist;
                best_node = node.id;
            }
        }

        println!("Brute force nearest node: {}", best_node);

        if best_node == greedy_result {
            println!("Greedy search: EXACT");
        } else {
            println!("Greedy search: APPROX");
        }

        if best_node == multi_result {
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
