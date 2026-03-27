use crate::metrics::l2::l2;
use rand::rng;
use rand::RngExt;
use std::cmp::Reverse;
use std::collections::HashSet;
use std::collections::BinaryHeap;
use crate::container::heap::{HeapItem, NodeId};

pub struct Node {
    pub id: NodeId,
    pub value: Vec<f32>,
    pub neighbours: Vec<NodeId>,
}

pub struct NSWIndex {
    pub nodes: Vec<Node>, // adjacency list
}

impl NSWIndex {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        return self.nodes.len();
    }

    pub fn insert(&mut self, candidate: &[f32], m: usize, ef: usize) {
        /* 
        * find the m closest entry vertexes
        * create a neighbourhood consisting of all neighbours of the m vertices
        * edit: renaming m to ef as exploration factor to be consistent with the hnsw impl
        * find k closest neighbours
        * assign the candidate as a neighour to them and vice-versa
        * note that neighbourhood may grow beyond k, it's not same as m which is max allowed connections per layer per node
        * but for consistency, renaming k to m.
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

        let local_neighbours = self._multi_search(&candidate, ef);
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

            if max_heap_k.len() < m {
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

    pub fn k_insert(&mut self, candidate: &[f32], m: usize, ef: usize) {
        // find m neighbours to connect with the candidate
        let new_node_id = self.nodes.len(); // this shouldn't be incremental counter
        let new_node = Node {
            id: new_node_id,
            value: candidate.to_vec(),
            neighbours: Vec::new()
        };

        self.nodes.push(new_node);
        if self.nodes.len() == 1 {
            return;
        }

        // find k nearest neighbours first
        let search_ef = ef * 2 + 10; // optimal choice as per paper: search_width = 2 * w + 10, w is width, i.e. ef
        // more like construction ef here, which is typically higher than the search ef (in search pipeline)
        let m_neighbours = self.k_multi_search(&candidate, search_ef, m);
        for neighbour in m_neighbours {
            self.nodes[new_node_id].neighbours.push(neighbour);
            self.nodes[neighbour].neighbours.push(new_node_id);
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

    pub fn k_multi_search(&self, query: &[f32], ef: usize, k: usize) -> Vec<NodeId> {
        
        // results for top-k, candidates - can be just a slice/vec, visitedSet - hash-set, tempRes - again heap
        let mut heap_k: BinaryHeap<HeapItem> = BinaryHeap::new();
        let mut rng = rng();

        for _ in 0..ef {
            // candidates placeholder
            let mut candidates: BinaryHeap<Reverse<HeapItem>> = BinaryHeap::new();
            let mut visited_set: HashSet<NodeId> = HashSet::new();

            // get random entry point
            let entry_vertex = rng.random_range(0..self.nodes.len());
            let entry_proximity = l2(&query, &self.nodes[entry_vertex].value);
            candidates.push(Reverse(HeapItem { node: entry_vertex, dist: entry_proximity}));
            visited_set.insert(entry_vertex);

            // find the closest candidate from candidates, i.e. top of the min heap
            while let Some(Reverse(closest_candidate)) = candidates.pop() {
                
                // if closest_candidate is farther than the farthest result, break
                if let Some(farthest_k) = heap_k.peek() {
                    if heap_k.len() >= k && closest_candidate.dist > farthest_k.dist {
                        break;
                    }
                }

                // insert closest_candidate into top-k
                if heap_k.len() < k {
                    heap_k.push(HeapItem { node: closest_candidate.node, dist: closest_candidate.dist });
                } else if let Some(top_heap) = heap_k.peek() {
                    if closest_candidate.dist < top_heap.dist {
                        heap_k.pop();
                        heap_k.push(HeapItem { node: closest_candidate.node, dist: closest_candidate.dist });
                    }
                }
                
                // if the closest candidate is accepted into top_k,
                // insert its neighbours into candidates if not visited already
                for &neighbour in &self.nodes[closest_candidate.node].neighbours {
                    if visited_set.insert(neighbour) {
                        let neighbour_proximity = l2(&query, &self.nodes[neighbour].value);

                        // add to candidates
                        candidates.push(Reverse(HeapItem { node: neighbour, dist: neighbour_proximity }));
                    }
                }
            }

        }

        let results = heap_k.iter().map(|item| item.node).collect();
        return results;
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

    fn _multi_search(&self, query: &[f32], ef: usize) -> Vec<NodeId> {
        /*
        * instead of one entry vertext, start m searches (m as in ef - exploration factor)
        * then take the closest element to the query
        */
        let mut rng = rng();
        let mut results = Vec::new();
        
        for _ in 0..ef {
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
