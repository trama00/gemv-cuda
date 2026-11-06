#include "gemv.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

namespace {

void print_device_info() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, i);
        const double peak_bw =
            2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
        std::printf("Device %d: %s\n", i, prop.name);
        std::printf("  SMs                          : %d\n", prop.multiProcessorCount);
        std::printf("  max blocks per SM            : %d\n", prop.maxBlocksPerMultiProcessor);
        std::printf("  max threads per block        : %d\n", prop.maxThreadsPerBlock);
        std::printf("  shared memory per block      : %zu bytes\n", prop.sharedMemPerBlock);
        std::printf("  memory clock                 : %d kHz\n", prop.memoryClockRate);
        std::printf("  memory bus width             : %d bits\n", prop.memoryBusWidth);
        std::printf("  peak memory bandwidth        : %.2f GB/s\n\n", peak_bw);
    }
}

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const int repeats = (argc > 1) ? std::atoi(argv[1]) : 5;
    const int n = DIM_SIZE;
    const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    if (n % 4 != 0) {
        std::fprintf(stderr, "DIM_SIZE must be a multiple of 4 (uchar4 vectorisation)\n");
        return EXIT_FAILURE;
    }

    print_device_info();

    // Pinned host memory so the device transfers are not the measurement.
    elem_t* h_matrix = nullptr;
    elem_t* h_vector = nullptr;
    check(cudaMallocHost(&h_matrix, static_cast<size_t>(n) * n * sizeof(elem_t)), "malloc host matrix");
    check(cudaMallocHost(&h_vector, static_cast<size_t>(n) * sizeof(elem_t)), "malloc host vector");

    elem_t* d_matrix = nullptr;
    elem_t* d_vector = nullptr;
    accum_t* d_block_sums = nullptr;
    check(cudaMalloc(&d_matrix, static_cast<size_t>(n) * n * sizeof(elem_t)), "malloc device matrix");
    check(cudaMalloc(&d_vector, static_cast<size_t>(n) * sizeof(elem_t)), "malloc device vector");
    check(cudaMalloc(&d_block_sums, static_cast<size_t>(blocks) * sizeof(accum_t)), "malloc device partials");

    // Deterministic data: same seed every run so results are reproducible.
    std::srand(42);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_matrix[static_cast<size_t>(i) * n + j] = static_cast<elem_t>(std::rand() % 2);
        }
    }
    for (int j = 0; j < n; ++j) {
        h_vector[j] = static_cast<elem_t>(std::rand() % 3);
    }

    // ---- CPU reference -----------------------------------------------------
    const clock_t cpu_begin = std::clock();
    const accum_t cpu_result = cpu_reference(h_matrix, h_vector, n);
    const clock_t cpu_end = std::clock();
    const double cpu_ms = 1000.0 * (cpu_end - cpu_begin) / CLOCKS_PER_SEC;
    std::printf("CPU reference (serial)          : %10.3f ms  (result %lld)\n", cpu_ms, cpu_result);

    // ---- GPU ---------------------------------------------------------------
    check(cudaMemcpy(d_matrix, h_matrix, static_cast<size_t>(n) * n * sizeof(elem_t),
                     cudaMemcpyHostToDevice), "copy matrix");
    check(cudaMemcpy(d_vector, h_vector, static_cast<size_t>(n) * sizeof(elem_t),
                     cudaMemcpyHostToDevice), "copy vector");

    const int warmup = 2;
    double best_ms = 1e9;
    std::vector<double> samples;
    accum_t gpu_result = 0;

    for (int rep = 0; rep < warmup + repeats; ++rep) {
        check(cudaMemset(d_block_sums, 0, static_cast<size_t>(blocks) * sizeof(accum_t)), "memset");

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        gemv_mul_reduce<<<blocks, THREADS_PER_BLOCK>>>(d_matrix, d_vector, d_block_sums, n);
        reduce_block_sums<<<1, blocks>>>(d_block_sums, blocks);

        check(cudaDeviceSynchronize(), "kernel run");
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        if (rep >= warmup) {
            samples.push_back(elapsed_ms);
            if (elapsed_ms < best_ms) best_ms = elapsed_ms;
        }
    }

    check(cudaMemcpy(&gpu_result, d_block_sums, sizeof(accum_t), cudaMemcpyDeviceToHost), "copy result");

    double sum_ms = 0.0;
    for (double v : samples) sum_ms += v;
    const double mean_ms = samples.empty() ? 0.0 : sum_ms / samples.size();

    std::printf("GPU kernel loop (%d timed runs) : %10.3f ms mean, %.3f ms best  (result %lld)\n",
                repeats, mean_ms, best_ms, gpu_result);
    std::printf("Speedup vs serial CPU           : %10.2fx\n", cpu_ms / best_ms);

    // Effective bandwidth assuming the matrix is read once.
    const double bytes = static_cast<double>(n) * n * sizeof(elem_t);
    std::printf("Effective bandwidth             : %10.2f GB/s\n", bytes / (best_ms * 1e6));

    std::printf("\n%s  (cpu=%lld, gpu=%lld)\n", (cpu_result == gpu_result) ? "OK" : "MISMATCH",
                cpu_result, gpu_result);

    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_block_sums);
    cudaFreeHost(h_matrix);
    cudaFreeHost(h_vector);
    return (cpu_result == gpu_result) ? EXIT_SUCCESS : EXIT_FAILURE;
}
