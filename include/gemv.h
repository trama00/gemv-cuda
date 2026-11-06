#ifndef GEMV_H
#define GEMV_H

#include <cstddef>

// Problem size. Override at compile time with -DDIM_SIZE=<n>.
// The recorded benchmark used 16384 (a 16384 x 16384 unsigned char matrix, 268 MB).
#ifndef DIM_SIZE
#define DIM_SIZE 16384
#endif

// Threads per block for the multiply-and-reduce kernel. Must be a power of two
// (the shared-memory reduction is a binary tree).
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 64
#endif

// Upper bound on threads per block; sizes the shared reduction buffer.
constexpr int kMaxThreadsPerBlock = 1024;

using elem_t = unsigned char;
using accum_t = long long;

// CPU reference: sum over all rows of sum_k( A[row][k] * b[k]^3 ).
accum_t cpu_reference(const elem_t* a, const elem_t* b, int n);

// Baseline nvcc build: one thread per row, no vectorisation, no shared memory.
__global__ void gemv_naive(const elem_t* a, const elem_t* b, accum_t* partials, int n);

// Optimised: thread-local accumulation, uchar4 vectorisation, unrolled loop,
// then a power-of-two shared-memory tree reduction to one value per block.
__global__ void gemv_mul_reduce(const elem_t* a, const elem_t* b, accum_t* block_sums, int n);

// Collapses one partial per block into a single value using a single block.
__global__ void reduce_block_sums(accum_t* vec, int count);

#endif  // GEMV_H
