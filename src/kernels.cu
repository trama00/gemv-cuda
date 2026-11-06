#include "gemv.h"

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Naive GPU baseline: one thread per row, scalar loads, no shared memory.
// Included so the optimisations in gemv_mul_reduce can be measured against
// something on the same hardware. Not part of the originally recorded run.
// ---------------------------------------------------------------------------
__global__ void gemv_naive(const elem_t* a, const elem_t* b, accum_t* partials, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    accum_t sum = 0;
    for (int k = 0; k < n; ++k) {
        accum_t bk = b[k];
        sum += static_cast<accum_t>(a[row * n + k]) * bk * bk * bk;
    }
    partials[row] = sum;
}

// ---------------------------------------------------------------------------
// Optimised kernel.
//
//   * thread-local accumulation  -> partial sum stays in a register
//   * uchar4 vectorisation       -> four elements per load instruction
//   * #pragma unroll 8           -> fewer branch instructions
//   * shared-memory tree reduce  -> one value leaves the block, no atomics
//
// Access pattern note: thread i owns row i, so consecutive threads read
// addresses DIM_SIZE bytes apart. Every warp therefore loads uncoalesced.
// That is the documented bottleneck in docs/optimization-notes.md.
// ---------------------------------------------------------------------------
__global__ void gemv_mul_reduce(const elem_t* a, const elem_t* b, accum_t* block_sums, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // one thread per row

    __shared__ accum_t temp[kMaxThreadsPerBlock];
    temp[threadIdx.x] = 0;

    accum_t sum = 0;
    if (i < n) {
#pragma unroll 8
        for (int k = 0; k < n; k += 4) {
            const uchar4 a_vec = *reinterpret_cast<const uchar4*>(&a[i * n + k]);
            const uchar4 b_vec = *reinterpret_cast<const uchar4*>(&b[k]);

            sum += a_vec.x * b_vec.x * b_vec.x * b_vec.x;
            sum += a_vec.y * b_vec.y * b_vec.y * b_vec.y;
            sum += a_vec.z * b_vec.z * b_vec.z * b_vec.z;
            sum += a_vec.w * b_vec.w * b_vec.w * b_vec.w;
        }
        temp[threadIdx.x] = sum;
    }

    __syncthreads();

    // Power-of-two tree reduction within the block.
#pragma unroll 8
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        const int index = 2 * stride * threadIdx.x;
        if (index < blockDim.x) {
            temp[index] += temp[index + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = temp[0];
    }
}

// ---------------------------------------------------------------------------
// Second stage of the hierarchical reduction: collapses the per-block partials
// into a single value in one block. Avoids atomics entirely.
// ---------------------------------------------------------------------------
__global__ void reduce_block_sums(accum_t* vec, int count) {
    const int thread = threadIdx.x;

    // All threads must reach the barrier, so the loop condition is inside the
    // body rather than in the for-header. The original single-file version
    // omitted __syncthreads() here; it happened to produce the right answer on
    // the T4 run but is racy across warps.
    for (int t = 1, k = count / 2; k >= 1; k /= 2, t *= 2) {
        if (thread < k) {
            vec[thread * 2 * t] = vec[thread * 2 * t] + vec[thread * 2 * t + t];
        }
        __syncthreads();
    }
}
