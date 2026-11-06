# GEMV with CUDA

CUDA implementation of a matrix-vector product with a cubic term,

```
out = sum_i( sum_k( A[i][k] * b[k]^3 ) )
```

on a **16384 x 16384** `unsigned char` matrix (268 MB) against a 16384-element
vector.

Course project for *Algoritmi e Architetture per il Calcolo ad Alte Prestazioni*
(AACAP), Challenge 3, Politecnico di Milano, 2024.

**Authors:** Marco Cioci (10694772), Francesco Rosnati (10562427), Luca Tramacere (10676881).

## Results

NVIDIA Tesla T4 (40 SMs, 320.06 GB/s peak memory bandwidth).

| Implementation | Time | Speedup |
|---|---|---|
| Naive serial CPU | 966.72 ms | 1.0x |
| CUDA, two-kernel reduction | **31.52 ms** | **30.7x** |

Result correctness is asserted against the CPU reference — both paths return
`396334905`. Full device query and raw output in
[`results/t4-colab-2024.txt`](results/t4-colab-2024.txt).

## What made it fast

- **Thread-local accumulation** — partial sums stay in registers.
- **`uchar4` vectorisation** — four elements per load instruction.
- **`#pragma unroll 8`** — fewer branches, more work in flight.
- **Shared-memory tree reduction** — one value per block, no atomics.
- **Hierarchical two-kernel reduction** — per-block partials collapsed in a
  single block, still no atomics.

Details and reasoning in [`docs/optimization-notes.md`](docs/optimization-notes.md).

## What is still wrong

The kernel maps **one thread per row**, so consecutive threads read addresses
16 KB apart and every warp load is uncoalesced. Measured effective bandwidth is
**8.5 GB/s against the T4's 320 GB/s peak — 2.7%**.

The 30.7x is real, but it is far from the hardware ceiling. The next steps are
warp-per-row loading with `__shfl_down_sync` reduction, and precomputing
`b[k]^3` once instead of per element. Both are written up in
[`docs/optimization-notes.md`](docs/optimization-notes.md) and neither is
implemented here — the submission was graded on the speedup.

## Build and run

Requires the CUDA toolkit (written against CUDA 12.2) and an NVIDIA GPU.

```bash
make                 # DIM_SIZE=16384 by default
make DIM_SIZE=4096   # smaller problem for GPUs with little memory
./gemv               # runs CPU reference, then 5 timed GPU runs after 2 warm-ups
```

`scripts/benchmark.sh` repeats the whole thing and prints each run.

Expected memory footprint at the default size: 268 MB host (pinned) plus
268 MB device, so roughly 550 MB of GPU memory.

## Repository layout

```
include/gemv.h              declarations and compile-time configuration
src/kernels.cu              CUDA kernels
src/reference.cpp           serial CPU reference
src/main.cu                 device query, benchmark harness, correctness check
scripts/benchmark.sh        repeated benchmark runner
results/t4-colab-2024.txt   the recorded measurement
docs/optimization-notes.md  what each optimisation does, and the known limit
notebook/                   original Colab notebook, kept for provenance
```

## Notes on the numbers

The CPU baseline is a naive single-threaded loop, not BLAS. The accurate claim
is **30.7x over the naive serial implementation**, not "30x faster than a CPU".

The `src/` tree was refactored from the original single-file notebook for
readability; the recorded measurement in `results/` comes from the notebook
run, not from this refactor, which has not been re-benchmarked. The refactored
version also adds `__syncthreads()` to the second reduction stage, which the
original omitted — it produced the correct answer on the recorded run but was
racy across warps.
