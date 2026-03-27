use std::collections::HashMap;
use std::{cmp::Reverse, collections::HashSet};
use std::{cmp, collections::BinaryHeap};
use rand::{distr::{Distribution, Uniform}};
use crate::{container::heap::{HeapItem, NodeId}, metrics::l2::l2};

pub struct Node {
    id: NodeId,
    value: Vec<f32>,
    neighbours: HashMap<usize, Vec<usize>>
}

pub struct HNSWIndex {
    top_layer: u32, // number of layers (typically represents the top layer)
    m: i32, // maximum number of connections per node at each layer
    m0: i32, // maximum number of connections per node at layer 0
    ml: f64, // layer normalization factor
    entry_point: NodeId, // entry point
    nodes: Vec<Node>
}

fn compute_ml(m: i32) -> f64 {
    return 1.0 / (m as f64).ln();
}

impl HNSWIndex {
    // ctor
    pub fn new(m: i32) -> Self {
        return Self {
            top_layer: 0,
            m: m,
            m0: 2 * m,
            ml: compute_ml(m),
            entry_point: 0,
            nodes: Vec::new()
        };
    }

    pub fn len(&self) -> usize{
        return self.nodes.len();
    }

    fn is_empty(&self) -> bool {
        self.nodes.len() == 0
    }

    fn sample_layer(&self, ml: f64) -> usize {
        let mut rng = rand::rng();
        // floor(- ln(UniformSample(0, 1)) * ml)
        let u: f64 = Uniform::new(0.0, 1.0).unwrap().sample(&mut rng); // sample from Uniform Distribution
        return (- u.ln() * ml) as usize;
    }

    pub fn insert(
        &mut self, 
        candidate: &[f32], 
        exploration_factor: usize) {
        // create the new node instance
        let incoming_node = Node {
            id: self.nodes.len(),
            value: candidate.to_vec(),
            neighbours: HashMap::new()
        };

        // if the db is empty
        if self.is_empty() {
            // println!("no node in db, adding first node ...");
            // insert node
            self.nodes.push(Node { id: incoming_node.id, value: incoming_node.value.to_vec(), neighbours: incoming_node.neighbours });
        
            // update metadata
            self.entry_point = incoming_node.id;
            self.top_layer = 0;
            // println!("no node in db, adding first node > node_id: {}, layer: {}, ep: {}", incoming_node.id, self.top_layer, self.entry_point);
            return
        }

        // insert node
        self.nodes.push(Node { id: incoming_node.id, value: incoming_node.value.to_vec(), neighbours: incoming_node.neighbours });
        
        // generate layer for the incoming node
        let calculated_layer = self.sample_layer(self.ml);
        // println!("calculated layer: {}", calculated_layer);

        let mut entry_points = Vec::new();
        entry_points.push(self.entry_point);

        // otherwise, descend from L to calculated_layer entry point
        for current_layer in (calculated_layer+1..=self.top_layer as usize).rev() {
            // println!("finding closest in layer: {}", current_layer);
            // find single entry point per layer, which is the closest candidate to the incoming node
            let closest_in_current_layer = self.search_layer(&incoming_node.value, &entry_points, current_layer, 1);
            // println!("found {} closest neighoburs in layer: {}", closest_in_current_layer.len(), current_layer);
            assert!(closest_in_current_layer.len() == 1);
            entry_points = closest_in_current_layer; // assuming that the results are sorted, nearest element to incoming_node
        }

        // now that we have an entry point to go into the calculated_layer,
        // find closest neighbours -> connect -> shrink if grown beyond allowed
        // println!("descending from layers using found {} entry points", entry_point.len());
        let layer_cap = cmp::min(calculated_layer, self.top_layer as usize);
        for current_layer in (0..layer_cap).rev() {
            // println!("finding closest candidate in layer {} using {} entry points with ef: {}", current_layer, entry_point.len(), exploration_factor);
            let closest_candidates = self.search_layer(candidate, &entry_points, current_layer, exploration_factor);
            assert!(
                closest_candidates.len() <= exploration_factor,
                "closest candidates {} found aren't equal to exploration factor {}",
                closest_candidates.len(), exploration_factor
            );

            // println!("found {} closest neighoburs in layer: {}", closest_candidates.len(), current_layer);
            // find m closest neighbours
            // let neighbours = self.select_neighbours_naive(candidate, &closest_candidates, self.m as usize);
            let neighbours = self.select_neighbours_heuristic(candidate, &closest_candidates, current_layer, self.m as usize, true, true);
            // println!("found {} neighbourhood to connect to", neighbours.len());
            // bidirectional connection w/ neighbours
            for &neighbour in &neighbours {
                // connect candidate
                // println!("connecting {} with neighbour {}", incoming_node.id, neighbour);
                self.nodes[incoming_node.id].neighbours.entry(current_layer).or_insert_with(Vec::new).push(neighbour);
                // println!("connected neihbours to {}: {}", incoming_node.id, self.nodes[incoming_node.id].neighbours.len());
                // connect neighbours
                // println!("connecting neighobur {} with {}", neighbour, incoming_node.id);
                self.nodes[neighbour].neighbours.entry(current_layer).or_insert_with(Vec::new).push(incoming_node.id);
                // println!("neighbour {} has {} connections", neighbour, self.nodes[neighbour].neighbours.len());
            }

            for &neighbour in &neighbours {
                let allowed_connections_for_neighbour = if current_layer != 0 { self.m } else { self.m0 };
                let neighbourhood= self.nodes[neighbour].neighbours[&current_layer].clone();

                if neighbourhood.len() > (allowed_connections_for_neighbour as usize) {
                    // if neighbourhood is larger the allowed connections in that layer, shrink.
                    // let shrinked_neighbours = self.select_neighbours_naive(&self.nodes[neighbour].value, &neighbourhood, allowed_connections_for_neighbour as usize);
                    // don't extend candidates during shrinking
                    let shrinked_neighbours = self.select_neighbours_heuristic(&self.nodes[neighbour].value, &neighbourhood, current_layer, allowed_connections_for_neighbour as usize, false, true);
                    
                    // find the diff in sets neighoburhood - shrinked neighbour hood, unlink connections from those to current neighbours
                    for &old_neighbour in &neighbourhood {
                        if !shrinked_neighbours.contains(&old_neighbour) {
                            // prune the connection
                            if let Some(old_neighbours_neighbourhood) = self.nodes[old_neighbour].neighbours.get_mut(&current_layer) {
                                old_neighbours_neighbourhood.retain(|&ele| ele != neighbour);
                            }

                            if let Some(neighbours_neighbourhood) = self.nodes[neighbour].neighbours.get_mut(&current_layer) {
                                neighbours_neighbourhood.retain(|&ele| ele != neighbour);
                            }
                        }
                    }

                    // set as new neighbourhood
                    self.nodes[neighbour].neighbours.insert(current_layer, shrinked_neighbours);
                }

                assert!(
                    self.nodes[neighbour].neighbours[&current_layer].len() <= allowed_connections_for_neighbour as usize, 
                    "invalid number of neighbours for neighbour {}, allowed {}, current {}, layer: {}",
                     neighbour, allowed_connections_for_neighbour, self.nodes[neighbour].neighbours[&current_layer].len(), current_layer);
            }

            // update entry point with closest candidates
            // println!("updating entry point with {} closest candidates for layer {}", closest_candidates.len(), current_layer);
            entry_points = closest_candidates;
        }

        // calculated_layer can be greater or smaller than the top most layer
        if calculated_layer > self.top_layer as usize {
            // println!("setting top layer to {}, entry point for hnsw to {}", calculated_layer, incoming_node.id);
            // new layer is greater than max layer
            self.top_layer = calculated_layer as u32;
            self.entry_point = incoming_node.id
        }

        // println!(">>>> total node {}, top layer {}, entry point {}", self.nodes.len(), self.top_layer, self.entry_point);
    }

    fn select_neighbours_naive(
        &self, 
        query: &[f32], 
        candidates: &Vec<usize>, 
        m: usize) -> Vec<usize> {
        // naive neighbour selection -> select best m from W candidates
        let mut nearest_neighbours: BinaryHeap<HeapItem> = BinaryHeap::new(); // min heap: nearest first
        // println!("select_neighbours_naive: finding {} nns from {} closest candidates", m, candidates.len());
        for &candidate in candidates {
            // calculate distance with candidate
            let proximity = l2(query, &self.nodes[candidate].value);
            if nearest_neighbours.len() < m {
                nearest_neighbours.push(HeapItem { node: candidate, dist: proximity });
                continue;
            } else if let Some(farthest_neighbour) = nearest_neighbours.peek() {
                if proximity < farthest_neighbour.dist {
                    nearest_neighbours.pop();
                    nearest_neighbours.push(HeapItem { node: candidate, dist: proximity });
                }
            }
        }

        let results: Vec<usize> = nearest_neighbours.iter().map(|ele| ele.node).collect();
        // println!("select_neighbours_naive: found {} nns from {} closest candidates", results.len(), candidates.len());
        return results;
    }

    fn select_neighbours_heuristic(
        &self,
        query: &[f32],
        candidates: &Vec<usize>,
        current_layer: usize,
        m: usize,
        extend_candidates: bool,
        keep_pruned_connections: bool
    ) -> Vec<usize> {

        let mut neigbour_search_pool: HashSet<usize> = HashSet::new();

        // add all candidates to search pool
        for &candidate in candidates {
            neigbour_search_pool.insert(candidate);
        }

        // extend cadidates to include their neighbours
        if extend_candidates {
            for &candidate in candidates {
                if let Some(neighbourhood) = self.nodes[candidate].neighbours.get(&current_layer) {
                    for &neighbour in neighbourhood {
                        neigbour_search_pool.insert(neighbour);
                    }
                }
            }
        }

        // create a heap based on their dist
        let mut neigbour_search_list: Vec<HeapItem> = neigbour_search_pool.into_iter()
            .map(|node| {
                let proximity = l2(query, &self.nodes[node].value);
                HeapItem { node, dist: proximity }
            })
            .collect();

        neigbour_search_list.sort_by(|a, b| a.dist.total_cmp(&b.dist));

        let mut nearest_neighbours: Vec<usize> = Vec::new();
        let mut discarded_neighbours: Vec<usize> = Vec::new();

        // for every candidate in the search pool, compare their proximity with already selected candidates
        for nearest_candidate in neigbour_search_list {

            let mut found_closer = false;

            for &selected_node in &nearest_neighbours {
                let proximity = l2(
                    &self.nodes[nearest_candidate.node].value,
                    &self.nodes[selected_node].value,
                );

                // discard condition: if the closest candidate is nearer to any "already" selected candidate than the query,
                // discard that candidate.
                if proximity < nearest_candidate.dist {
                    found_closer = true;
                    break;
                }
            }

            if !found_closer {
                if nearest_neighbours.len() < m {
                    nearest_neighbours.push(nearest_candidate.node);
                }
            } else {
                discarded_neighbours.push(nearest_candidate.node);
            }
        }

        // if still space for top-m, pull the best from discard candidates, earlier sorting ensures that they are sorted
        if keep_pruned_connections {
            for node in discarded_neighbours {
                if nearest_neighbours.len() >= m {
                    break;
                }
                nearest_neighbours.push(node);
            }
        }

        return nearest_neighbours;
    }

    fn search_layer(
        &self, 
        query: &[f32], 
        entry_points: &Vec<usize>, 
        current_layer: usize, 
        exploration_factor: usize) -> Vec<NodeId> {
        // find nearest neighbours in the current layer
        // results for top-k, candidates - can be just a slice/vec, visitedSet - hash-set, tempRes - again heap
        let mut nearest_neighbours: BinaryHeap<HeapItem> = BinaryHeap::new(); // not more than ef neighbours
        let mut candidates: BinaryHeap<Reverse<HeapItem>> = BinaryHeap::new();
        let mut visited_set: HashSet<NodeId> = HashSet::new();

        for &entry_point in entry_points {
            let entry_proximity = l2(&query, &self.nodes[entry_point].value);
            // println!("search_layer: entry point {} with proximity to query {}", entry_point, entry_proximity);
            candidates.push(Reverse(HeapItem { node: entry_point, dist: entry_proximity}));
            visited_set.insert(entry_point);
        }

        // find the closest candidate from candidates, i.e. top of the min heap
        while let Some(Reverse(closest_candidate)) = candidates.pop() {
            // println!("search_layer: found closest candidate: {} with proximity {}", closest_candidate.node, closest_candidate.dist);
            
            // if closest_candidate is farther than the farthest result, break
            if nearest_neighbours.len() >= exploration_factor {
                if let Some(farthest_k) = nearest_neighbours.peek() {
                    if closest_candidate.dist > farthest_k.dist {
                        // println!("search_layer: found a candidate {} with proximity {} to query farther than the farthest neighobur {} with proximity to query {}", closest_candidate.node, closest_candidate.dist, farthest_k.node, farthest_k.dist);
                        break;
                    }
                }
            }
        

            // insert closest_candidate into top-k
            if nearest_neighbours.len() < exploration_factor {
                nearest_neighbours.push(HeapItem { node: closest_candidate.node, dist: closest_candidate.dist });
            } else if let Some(farthest_neighbour) = nearest_neighbours.peek() {
                if closest_candidate.dist < farthest_neighbour.dist {
                    nearest_neighbours.pop();
                    nearest_neighbours.push(HeapItem { node: closest_candidate.node, dist: closest_candidate.dist });
                }
            }

            // closest candidate might not have any neighbours in current_layer just yet.
            // println!(
            //     "current layers count in closest candidate: {}, current layer: {}",
            //     self.nodes[closest_candidate.node].neighbours.len(),
            //     current_layer
            // );

            if let Some(layer_neighbours) = self.nodes[closest_candidate.node]
                .neighbours
                .get(&current_layer)
            {
                for &neighbour in layer_neighbours {
                    if visited_set.insert(neighbour) {
                        let neighbour_proximity = l2(&query, &self.nodes[neighbour].value);

                        candidates.push(Reverse(HeapItem {
                            node: neighbour,
                            dist: neighbour_proximity,
                        }));
                    }
                }
            }
        }

        // sort neighbours as per their proximity to query
        let mut results: Vec<_> = nearest_neighbours.into_vec();
        results.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        let results = results.into_iter().map(|item| item.node).collect();

        return results;
    }

    pub fn search(
        &self, 
        query: &[f32], 
        k: usize, 
        exploration_factor: usize) -> Vec<NodeId> {
        // find an entry point till the 1st layer
        let mut entry_points = Vec::new();
        entry_points.push(self.entry_point);

        // find entry point till layer 1
        for current_layer in (1..=self.top_layer).rev(){
            let closest_candidates = self.search_layer(query, &entry_points, current_layer as usize, 1);
            assert!(closest_candidates.len() != 0);
            // ideally get the nearest from closest_candidates to query, but ok.
            entry_points = closest_candidates;
        }

        // use that entry point to search in the 0th layer
        let nearest_neighbours = self.search_layer(query, &entry_points, 0, exploration_factor);
        // sort and return top k
        let mut results: BinaryHeap<Reverse<HeapItem>> = BinaryHeap::new();
        for neighbour in nearest_neighbours {
            let proximity = l2(query, &self.nodes[neighbour].value);
            if results.len() < k {
                results.push(Reverse(HeapItem{node: neighbour, dist: proximity}));
            } else if let Some(Reverse(closest_neighbour)) = results.peek() {
                if proximity < closest_neighbour.dist {
                    results.pop();
                    results.push(Reverse(HeapItem { node: neighbour, dist: proximity }));
                }
            }
        }

        // todo: sort results before sending to ensure ordered results
        return results.iter().map(|Reverse(ele)| ele.node).collect();
    }

    pub fn bruteforce_top_k(&self, query: &[f32], k: usize) -> Vec<NodeId> {

        let mut max_heap_k: BinaryHeap<HeapItem> = BinaryHeap::new();
        for node in &self.nodes {
            let proximity = l2(&query, &node.value);

            if max_heap_k.len() < k {
                max_heap_k.push(HeapItem { node: (node.id), dist: (proximity) });
            } else if let Some(top) = max_heap_k.peek() { // if heap is full, remove the farthest, insert new closer
                if proximity < top.dist {
                    max_heap_k.pop();
                    max_heap_k.push(HeapItem { node: (node.id), dist: (proximity) });
                }
            }
        }
        
        let neighbours: Vec<NodeId> =
            max_heap_k.iter().map(|item| item.node).collect();

        return neighbours;
    }
}
