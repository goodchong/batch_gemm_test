#ifndef BENCHMARK_RUNNER_HPP
#define BENCHMARK_RUNNER_HPP

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept>
#include <type_traits> // For std::is_same
#include <cuda_runtime.h> // For cudaGetDeviceCount, etc.

#include "utils.hpp"
#include "cpu_benchmark.hpp"
#include "gpu_benchmark.hpp"

// --- Benchmark Runner ---
template<typename T>
void run_benchmarks(int batch_size, int M, int N, int K, int num_iterations) {
    std::cout << "Precision: " << (std::is_same<T, float>::value ? "float" : "double") << std::endl;
    std::cout << "Benchmarking Batch Matrix Multiplication (Batch=" << batch_size
              << ", M=" << M << ", N=" << N << ", K=" << K << ", Iterations=" << num_iterations << ")" << std::endl;
    std::cout << "Matrix A: " << batch_size << " x (" << M << " x " << K << ")" << std::endl;
    std::cout << "Matrix B: " << batch_size << " x (" << K << " x " << N << ")" << std::endl;
    std::cout << "Matrix C: " << batch_size << " x (" << M << " x " << N << ")" << std::endl;

    // --- Initialization ---
    std::vector<T> h_A, h_B;
    std::cout << "Initializing matrices..." << std::endl;
    initialize_matrices<T>(h_A, h_B, batch_size, M, N, K);
    std::vector<T> h_C_cpu(static_cast<size_t>(batch_size) * M * N);
    std::vector<T> h_C_gpu(static_cast<size_t>(batch_size) * M * N); // Separate result buffer for GPU

    // --- CPU Benchmark ---
    try {
        double cpu_avg_time = benchmark_cpu_mkl<T>(h_A, h_B, h_C_cpu, batch_size, M, N, K, num_iterations);
        std::cout << "CPU (MKL) Average Time: " << cpu_avg_time << " ms" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "CPU Benchmark Error: " << e.what() << std::endl;
    }

    // --- GPU Benchmark ---
    int device_count;
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err != cudaSuccess || device_count == 0) {
        std::cerr << "CUDA GPU not found or CUDA error: " << cudaGetErrorString(cuda_err) << ". Skipping GPU benchmark." << std::endl;
    } else {
         try {
            // Select GPU device (optional, defaults to device 0)
            CUDA_CHECK(cudaSetDevice(0));
            cudaDeviceProp deviceProp;
            CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
            std::cout << "Using GPU: " << deviceProp.name << std::endl;

            double gpu_avg_time = benchmark_gpu_cublas<T>(h_A, h_B, h_C_gpu, batch_size, M, N, K, num_iterations);
            std::cout << "GPU (cuBLAS) Average Time: " << gpu_avg_time << " ms" << std::endl;

            // Optional: Add verification step comparing h_C_cpu and h_C_gpu

        } catch (const std::exception& e) {
            std::cerr << "GPU Benchmark Error: " << e.what() << std::endl;
        }
    }
}

#endif // BENCHMARK_RUNNER_HPP
