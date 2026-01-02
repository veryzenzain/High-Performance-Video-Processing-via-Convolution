# High-Performance 2D Convolution Engine (C / AVX2 / OpenMP / MPI)

A performance-focused C project that implements **2D discrete convolutions** (core primitive behind blur/sharpen filters) and progressively optimizes them from a correct baseline to a parallel/distributed implementation using:
- **AVX2 SIMD intrinsics** for data-level parallelism
- **OpenMP** for multi-threaded execution
- **MPI** for coordinating many convolution “tasks” across processes
---

## engineering highlights

- **Low-level performance optimization:** vectorized inner loops with AVX2 intrinsics via `__m256i` / `_mm256_*` operations :contentReference[oaicite:1]{index=1}  
- **Parallel scaling on a single node:** OpenMP work-sharing to spread convolution output computation across threads :contentReference[oaicite:2]{index=2}  
- **Distributed throughput:** MPI coordinator that distributes independent convolution tasks across multiple processes (coarse-grained parallelism)
- **Systems-style constraints:** binary matrix I/O, careful memory layout (row-major), tail handling, and correctness validation against reference outputs


## Code map

**Core implementations**
- `src/compute_naive.c`  
  Correct baseline convolution (easy to read / verify)
- `src/compute_optimized.c`  
  SIMD + OpenMP + other optimizations (cache/layout/loop structure)
- `src/compute_optimized_mpi.c`  
  Optimized compute used under the MPI coordinator
- `src/coordinator_mpi.c`  
  MPI task distribution + collection of results

