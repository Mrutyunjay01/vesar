# VEctor SeARch (VESAR)

Diving into the rabbit hole of vector search. Core idea is to explore vector search from first principles and implement, benchmark, experiment with known approaches. I've picked up 3 papers to begin with, that leads to HNSW and one accelerated implementation by NVIDIA. This repo would act as an accompanying code for a rather detailed blog I will publish soon on my portfolio [mrutyunjaybiswal.com](https://mrutyunjaybiswal.com/blog) (or substack). No immediate commitment on when I will able to wrap this up, I would instead treat this as a exploration, not an item to tick off.

I would appreciate more relevant paper suggestions for the reading list, I will try to implement so and benchmark.

EDIT: I expected to wrap the blog first, then the code, but the script seems to have flipped. So here it is, Rust implementation of NSW and HNSW papers.

> Note that, I am still quite afresh to Rust, so kindly bear with the ugly code and strucutre. I wish to improve as I write more in time. Also, I wouldn't prefer bringing in any coding agent to this repo just yet, as it's for learning and exploration (as of now).

## Set up

### Prerequisites

- [x] Rust

Install [Rust](https://rust-lang.org/tools/install/) if you haven't already.

### Project structure

```sh
vesar/
    /vesar
        /benches        -- contains benchmarks
        /src            -- core libraries and implementation
            main.rs
            .......
        /tests          -- tests
    .gitignore
    README.md
```

### Benchmark

To run benchmarks on insertions, queries per second (qps) and recall with `1_00_000` 16-dimensional data points:

```sh
N=1_00_000 DIM=16 cargo bench --bench insert --bench qps --bench recall -- --sample-size 10
```

You can explore more about usage of environment variables for benchmarks in the respective files. To keep things consistent, here's an abstract nomencleture:

- N     : Number of data points in the dataset
- DIM   : Dimension of each datapoint in the dataset
- M     : Allowed connections per node per layer (per HNSW), or neighbours to connect to per insertion (NSW)
- EFC   : Exploration factor for Construction (which is typically higher than query in HNSW, but more or same in NSW with some tweaks)
- EFQ   : Exploration factor for Query (equivalant to beam width `W`, or multi-search params `m` in NSW papers)
- K     : top-K query (search) results.

### Tests

To run tests,

```sh
cargo test --test hnsw_tests  # for hnsw tests
cargo test --test nsw_tests   # for nsw tests
```

## Results

TODO: to add benchmarking snaps and results aggregation.

## TODOs (High-level)

### Algorithms & Papers

- [x] Approximate Nearest Neighbour (greedy + multi search, insertion) initial implementation
- [x] HNSW insertions + query

### Datasets

- [x] Dataset 1: Simulated data points upto 1M elements with d upto 16; L2 as proximity metric. (all configurable)
- [ ] Dataste 2: CoPHiR dataset (208-d, k ~ 30 neighoburs, L1 metric).

### Benchmarking

Benchmark against experimental set up of Malkov et. al. [2].

- [x] Insertion latency.
- [x] QPS.
- [x] Correctness (top-K Recall).
- [ ] Number of Proximity Metric Calculations as a fraction of Dataset size.
- [ ] Scale with/against Dimensionality.

## References

1. [Approximate Nearest Neighbor Search Small World Approach](https://www.iiis.org/CDs2011/CD2011IDI/ICTA_2011/PapersPdf/CT175ON.pdf) (NSW)
1. [Approximate Nearest Neighbor Algorithm Based on Navigable Small World Graphs](https://www.sciencedirect.com/science/article/abs/pii/S0306437913001300) (NSW)
1. [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) (HNSW paper)
