# VEctor SeARch (VESAR)

Diving into the rabbit hole of vector search. Core idea is to explore vector search from first principles and implement, benchmark, experiment with known approaches. I've picked up 3 papers to begin with, that leads to HNSW and one accelerated implementation by NVIDIA. This repo would act as an accompanying code for a rather detailed blog I will publish soon on my portfolio [mrutyunjaybiswal.com](https://mrutyunjaybiswal.com/blog) (or substack). No immediate commitment on when I will able to wrap this up, I would instead treat this as a exploration, not an item to tick off.

I would appreciate more relevant paper suggestions for the reading list, I will try to implement so and benchmark.

> Note that, I am still quite afresh to Rust, so kindly bear with the ugly code and strucutre. I wish to improve as I write more in time. Also, I wouldn't prefer bringing in any coding agent to this repo just yet, as it's for learning and exploration (as of now).

## TODOs (High-level)

- [x] Approximate Nearest Neighbour (greedy + multi search, insertion) initial implementation
- [ ] Dataset 1: Simulated data points upto 5 * 10^7 elements with d upto 50; L2 as proximity metric.
- [ ] Dataste 2: CoPHiR dataset (208-d, k ~ 30 neighoburs, L1 metric).
- [ ] Benchmark against experimental set up of Malkov et. al. [2].
    - [ ] QPS.
    - [ ] Correctness.
    - [ ] Number of Proximity Metric Calculations as a fraction of Dataset size.
    - [ ] Scale with/against Dimensionality.

## References

1. [Approximate Nearest Neighbor Search Small World Approach](https://www.iiis.org/CDs2011/CD2011IDI/ICTA_2011/PapersPdf/CT175ON.pdf)
1. [Approximate Nearest Neighbor Algorithm Based on Navigable Small World Graphs](https://www.sciencedirect.com/science/article/abs/pii/S0306437913001300)
