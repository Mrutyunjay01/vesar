use crate::metrics::l2::l2;
use rand::rng;
use rand::RngExt;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::collections::BinaryHeap;

pub type NodeId = usize;

#[derive(Debug)]
struct HeapItem {
    node: NodeId,
    dist: f32,
}

impl Eq for HeapItem {}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap()
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct Node {
    pub id: NodeId,
    pub value: Vec<f32>,
    pub neighbours: Vec<NodeId>,
}

pub struct ANNIndex {
    pub nodes: Vec<Node>, // adjacency list
}

impl ANNIndex {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
        }
    }

    pub fn insert(&mut self, candidate: &[f32], k: usize, m: usize) {
        /* 
        * find the m closest entry vertexes
        * create a neighbourhood consisting of all neighbours of the m vertices
        * find k closest neighbours
        * assign the candidate as a neighour to them and vice-versa
        * (approsimate delaunay graph instead of exact voronoi neighbours) 
        */

        let new_node_id = self.nodes.len();
        let new_node = Node {
            id: new_node_id,
            value: candidate.to_vec(),
            neighbours: Vec::new()
        };

        if self.nodes.is_empty() {
            self.nodes.push(new_node);
            return;
        }

        let local_neighbours = self._multi_search(&candidate, m);
        // set of entire neighbourhood derived from local neighbours, no duplicates
        let mut neighbourhood = HashSet::new(); 
        for &neighbour in &local_neighbours {
            // udpate neighbourhood set with neighbours of the neighbour
            neighbourhood.insert(neighbour);
            // calculate the proximity and update the heap
            for &neighbour in &self.nodes[neighbour].neighbours {
                neighbourhood.insert(neighbour);
            }
        }

        // create a max heap to contain the proximity metrics from the candidate to the neighbours of the neighbourhood
        // for top k elements in the heap, assign the candidate as the neighbour and vice versa

        let mut max_heap_k: BinaryHeap<HeapItem> = BinaryHeap::new();
        for node_id in neighbourhood {
            let proximity = l2(candidate, &self.nodes[node_id].value);

            if max_heap_k.len() < k {
                max_heap_k.push(HeapItem { node: (node_id), dist: (proximity) });
            } else if let Some(top) = max_heap_k.peek() { // if heap is full, remove the farthest, insert new closer
                if proximity < top.dist {
                    max_heap_k.pop();
                    max_heap_k.push(HeapItem { node: (node_id), dist: (proximity) });
                }
            }
        }
        
        let neighbours: Vec<NodeId> =
            max_heap_k.iter().map(|item| item.node).collect();

        self.nodes.push(new_node);

        for n in neighbours {
            self.nodes[new_node_id].neighbours.push(n);
            self.nodes[n].neighbours.push(new_node_id);
        }
    }

    pub fn greedy_search(&self, query: &[f32], entry_point: NodeId) -> NodeId {
        /*
        * start from the entry point, calculate the distance
        * look for closer nodes in entry point's neighbourhood, measured by proximity metrics
        * if a closer is found, make that the new closer, greedily search until no closer node is found.
        */ 
        
        let mut current_closest = entry_point;

        loop {
            let mut best_closest = current_closest;
            let mut best_proximity = l2(query, &self.nodes[current_closest].value);

            for &neighbour in &self.nodes[current_closest].neighbours {
                let proximity = l2(query, &self.nodes[neighbour].value);

                if proximity < best_proximity {
                    best_proximity = proximity;
                    best_closest = neighbour;
                }
            }

            if best_closest == current_closest {
                break;
            }

            current_closest = best_closest;
        }

        return current_closest;
    }

    pub fn multi_search(&self, query: &[f32], m: usize) -> NodeId {
        let results = self._multi_search(query, m);

        // among all present in the results set, find the closes to the query
        let mut closest_neighbour = results[0];
        let mut closest_proximity = l2(query, &self.nodes[closest_neighbour].value);

        for &ele in &results {
            let proximity = l2(query, &self.nodes[ele].value);
            if proximity < closest_proximity {
                closest_proximity = proximity;
                closest_neighbour = ele;
            }
        }

        return closest_neighbour;
    }

    pub fn bruteforce_nn(&self, query: &[f32]) -> NodeId {
        let mut best_node = 0;
        let mut best_dist = l2(&query, &self.nodes[best_node].value);

        for node in &self.nodes {
            let dist = l2(&query, &node.value);
            if dist < best_dist {
                best_dist = dist;
                best_node = node.id;
            }
        }

        return best_node;
    }

    fn _multi_search(&self, query: &[f32], m: usize) -> Vec<NodeId> {
        /*
        * instead of one entry vertext, start m searches
        * then take the closest element to the query
        */
        let mut rng = rng();
        let mut results = Vec::new();
        
        for _ in 0..m {
            let entry_vertex = rng.random_range(0..self.nodes.len());
            let local_node_id = self.greedy_search(query, entry_vertex);
            // if local_minima not in the set, add to the results set
            if !results.contains(&local_node_id) {
                results.push(local_node_id);
            }
        }

        return results;
    }
}
