use std::cmp::Ordering;

pub type NodeId = usize;

#[derive(Debug)]
pub struct HeapItem {
    pub node: NodeId,
    pub dist: f32,
}

impl Eq for HeapItem {}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Ord for HeapItem {
    // total order on distances: a NaN distance no longer panics inside heap operations
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.total_cmp(&other.dist)
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}