#ifndef GPU_BENCHMARK_HPP
#define GPU_BENCHMARK_HPP

#include <vector>
#include <chrono>
#include <stdexcept>
#include <type_traits> // For std::is_same
#include <cublas_v2.h> // For cuBLAS
#include <cuda_runtime.h> // For CUDA Runtime API
#include "utils.hpp" // For CUDA_CHECK, CUBLAS_CHECK

// --- CUDA GPU Benchmark ---
template<typename T>
double benchmark_gpu_cublas(const std::vector<T>& h_A, const std::vector<T>& h_B, std::vector<T>& h_C,
                            int batch_size, int M, int N, int K, int num_iterations) {

    T *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t size_A = static_cast<size_t>(batch_size) * M * K * sizeof(T);
    size_t size_B = static_cast<size_t>(batch_size) * K * N * sizeof(T);
    size_t size_C = static_cast<size_t>(batch_size) * M * N * sizeof(T);

    // Allocate memory on GPU
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), size_A));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), size_B));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), size_C));

    // Copy data from Host to GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    // No need to copy C if beta is 0, but let's clear it just in case
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // cuBLAS parameters
    int lda = K; // Leading dimension of A (stride between rows)
    int ldb = N; // Leading dimension of B
    int ldc = N; // Leading dimension of C
    long long int strideA = static_cast<long long int>(M) * K; // Stride between matrices in batch A
    long long int strideB = static_cast<long long int>(K) * N; // Stride between matrices in batch B
    long long int strideC = static_cast<long long int>(M) * N; // Stride between matrices in batch C
    T alpha_val = static_cast<T>(1.0);
    T beta_val = static_cast<T>(0.0);
    const T* alpha = &alpha_val;
    const T* beta = &beta_val;


    // Pointers to device memory arrays
    std::vector<const T*> d_A_array(batch_size);
    std::vector<const T*> d_B_array(batch_size);
    std::vector<T*>       d_C_array(batch_size);
    for(int i = 0; i < batch_size; ++i) {
        d_A_array[i] = d_A + i * strideA;
        d_B_array[i] = d_B + i * strideB;
        d_C_array[i] = d_C + i * strideC;
    }

    // Allocate memory on the device for the arrays of pointers
    const T **d_A_ptrs = nullptr;
    const T **d_B_ptrs = nullptr;
    T **d_C_ptrs = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A_ptrs), batch_size * sizeof(const T*)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B_ptrs), batch_size * sizeof(const T*)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C_ptrs), batch_size * sizeof(T*)));

    // Copy the arrays of pointers from host to device
    CUDA_CHECK(cudaMemcpy(d_A_ptrs, d_A_array.data(), batch_size * sizeof(const T*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_ptrs, d_B_array.data(), batch_size * sizeof(const T*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_ptrs, d_C_array.data(), batch_size * sizeof(T*), cudaMemcpyHostToDevice));


    // Select cuBLAS function based on type T
    cublasStatus_t cublas_status;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end; // Declare start/end here
    std::chrono::duration<double, std::milli> elapsed;

    if constexpr (std::is_same<T, float>::value) {
        // Warm-up run
        cublas_status = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                           N, M, K,
                                           reinterpret_cast<const float*>(alpha),
                                           reinterpret_cast<const float**>(d_B_ptrs), ldb,
                                           reinterpret_cast<const float**>(d_A_ptrs), lda,
                                           reinterpret_cast<const float*>(beta),
                                           reinterpret_cast<float**>(d_C_ptrs), ldc,
                                           batch_size);
        CUBLAS_CHECK(cublas_status);
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync after warm-up

        start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < num_iterations; ++iter) {
             cublas_status = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                               N, M, K,
                                               reinterpret_cast<const float*>(alpha),
                                               reinterpret_cast<const float**>(d_B_ptrs), ldb,
                                               reinterpret_cast<const float**>(d_A_ptrs), lda,
                                               reinterpret_cast<const float*>(beta),
                                               reinterpret_cast<float**>(d_C_ptrs), ldc,
                                               batch_size);
             // CUBLAS_CHECK(cublas_status); // Optional: Check status inside loop
        }
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync before stopping timer
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;

    } else if constexpr (std::is_same<T, double>::value) {
        // Warm-up run
        cublas_status = cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                           N, M, K,
                                           reinterpret_cast<const double*>(alpha),
                                           reinterpret_cast<const double**>(d_B_ptrs), ldb,
                                           reinterpret_cast<const double**>(d_A_ptrs), lda,
                                           reinterpret_cast<const double*>(beta),
                                           reinterpret_cast<double**>(d_C_ptrs), ldc,
                                           batch_size);
        CUBLAS_CHECK(cublas_status);
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync after warm-up

        start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < num_iterations; ++iter) {
             cublas_status = cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                               N, M, K,
                                               reinterpret_cast<const double*>(alpha),
                                               reinterpret_cast<const double**>(d_B_ptrs), ldb,
                                               reinterpret_cast<const double**>(d_A_ptrs), lda,
                                               reinterpret_cast<const double*>(beta),
                                               reinterpret_cast<double**>(d_C_ptrs), ldc,
                                               batch_size);
             // CUBLAS_CHECK(cublas_status); // Optional: Check status inside loop
        }
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync before stopping timer
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;

    } else {
         // Cleanup before throwing
         cudaFree(d_A);
         cudaFree(d_B);
         cudaFree(d_C);
         cudaFree(d_A_ptrs);
         cudaFree(d_B_ptrs);
         cudaFree(d_C_ptrs);
         cublasDestroy(handle);
         throw std::runtime_error("Unsupported data type for cuBLAS benchmark");
    }

    // Copy result back to host (optional, for verification)
    // CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_A_ptrs));
    CUDA_CHECK(cudaFree(d_B_ptrs));
    CUDA_CHECK(cudaFree(d_C_ptrs));
    CUBLAS_CHECK(cublasDestroy(handle));

    return elapsed.count() / num_iterations;
}


#endif // GPU_BENCHMARK_HPP
