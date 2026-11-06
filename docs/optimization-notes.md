# Optimisation notes

What each change in `gemv_mul_reduce` does, and what is still wrong with it.

## The problem

```
out = sum_i( sum_k( A[i][k] * b[k]^3 ) )
```

`A` is 16384 x 16384 `unsigned char` — 268 MB, which does not fit in cache on a
T4 (6 MB L2). So the kernel is a streaming problem: every element of `A` is
touched once and never reused.

The interesting consequence is that this is **memory bound, not compute bound**.
The `b[k]^3` term looks expensive but `b` is only 16 KB and stays resident;
`A` is the whole cost.

## What was applied

### 1. Thread-local accumulation

```cpp
accum_t sum = 0;
for (...) { sum += ...; }
```

The partial sum lives in a register for the whole loop. The alternative — a
global-memory accumulator per thread — would add one read and one write per
element, which at this size would dominate everything else.

### 2. `uchar4` vectorisation

```cpp
const uchar4 a_vec = *reinterpret_cast<const uchar4*>(&a[i * n + k]);
```

Four `unsigned char` loads become one 32-bit load. This quarters the number of
memory *instructions*, which matters when the instruction issue rate — not
bandwidth — is the immediate limit. It does **not** change the number of bytes
moved.

### 3. Loop unrolling

`#pragma unroll 8` on the main loop and on the reduction. Fewer branch
instructions and more independent work in flight per thread.

### 4. Shared-memory tree reduction

Each thread's partial is written to `__shared__ accum_t temp[1024]`, then
reduced with a power-of-two tree over `__syncthreads()` barriers. One value
leaves each block.

This replaces what would otherwise be an atomic add per thread (16384 atomics)
with 8 rounds of shared-memory reads per block.

### 5. Hierarchical two-kernel reduction

`gemv_mul_reduce` writes one partial per block (256 values).
`reduce_block_sums` collapses those in a single block. Still no atomics.

A single kernel could do this with a grid-wide barrier or a last-block
counter, which would save one launch. Not worth it at this size.

## What is still wrong: coalescing

The mapping is **one thread per row**:

```
thread i  ->  row i  ->  A[i][0], A[i][1], A[i][2], ...
```

Consecutive threads are 16384 bytes apart. A warp of 32 threads issues 32
separate 32-byte transactions for what would ideally be one 128-byte
transaction. **Every load is uncoalesced.**

The measurement says exactly this: 8.5 GB/s achieved against 320 GB/s peak.
The 30.7x speedup is real, but it is ~2.7% of what the hardware can do. Faster
than naive C, nowhere near the roofline.

### The fixes, in the order I would try them

**a. Warp-per-row.** Assign a warp to a row instead of a thread. The 32 threads
of the warp read 32 consecutive elements, so each warp issues one coalesced
128-byte transaction. The warp then reduces with `__shfl_down_sync`, which is
faster than shared memory and needs no barrier.

Expected: the change most likely to move the bandwidth number substantially,
and it is a small edit.

**b. Transpose A.** Store A column-major (or transpose in a pre-pass) so the
reduction runs along a coalesced dimension. Costs one full pass over 268 MB,
which pays back if A is touched more than once — here it is touched exactly
once, so this is probably a wash unless the transpose is fused into a previous
kernel.

**c. Shared-memory tiling.** Load `[TILE][TILE]` blocks of A cooperatively into
shared memory, so each block-level load is coalesced and threads consume from
shared. Also lets a block reuse `b` from shared memory rather than reloading it.

**d. Precompute `b[k]^3`.** Currently every thread cubes every element of `b`
it reads — 16384 threads x 16384 cubes = 268 million multiply-pairs that are
identical across all rows. Cubing `b` once into a 16 KB buffer removes that
work entirely. Cheap, obviously correct, and it should have been in the first
version.

## Honest summary for an interview

> I got 30x over naive C with shared-memory reduction, vectorisation and
> unrolling. Then I measured achieved bandwidth and found I was using 2.7% of
> the T4's peak, because I mapped one thread per row and every warp load was
> strided. The fix is warp-per-row with shuffle reduction, plus cubing the
> vector once instead of per element. I did not implement it in the
> submission — the challenge was graded on the speedup — but it is the obvious
> next step.
