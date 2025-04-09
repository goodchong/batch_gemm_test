#include <iostream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <random>
#include <numeric>
#include <mkl.h>       // For Intel MKL
#include <cublas_v2.h> // For cuBLAS
#include <cuda_runtime.h> // For CUDA Runtime API

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                  \
do {                                                                      \
    cudaError_t err = call;                                               \
    if (err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));                                 \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)

// Helper macro for cuBLAS error checking
#define CUBLAS_CHECK(call)                                                \
do {                                                                      \
    cublasStatus_t status = call;                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                \
        fprintf(stderr, "cuBLAS Error at %s:%d\n", __FILE__, __LINE__);   \
        /* Note: cuBLAS doesn't have a dedicated error string function */ \
        /* like CUDA runtime. You might need to map status codes manually */ \
        /* if more detailed error reporting is needed. */                 \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)


// Function to initialize matrices with random values
void initialize_matrices(std::vector<float>& h_A, std::vector<float>& h_B, int batch_size, int M, int N, int K) {
    std::mt19937 rng(12345); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    size_t size_A = static_cast<size_t>(batch_size) * M * K;
    size_t size_B = static_cast<size_t>(batch_size) * K * N;

    h_A.resize(size_A);
    h_B.resize(size_B);

    for (size_t i = 0; i < size_A; ++i) {
        h_A[i] = dist(rng);
    }
    for (size_t i = 0; i < size_B; ++i) {
        h_B[i] = dist(rng);
    }
}

// --- MKL CPU Benchmark ---
double benchmark_cpu_mkl(const std::vector<float>& h_A, const std::vector<float>& h_B, std::vector<float>& h_C,
                         int batch_size, int M, int N, int K, int num_iterations) {

    // MKL expects arrays of pointers for batched operations
    std::vector<const float*> A_array(batch_size);
    std::vector<const float*> B_array(batch_size);
    std::vector<float*>       C_array(batch_size);

    size_t matrix_size_A = static_cast<size_t>(M) * K;
    size_t matrix_size_B = static_cast<size_t>(K) * N;
    size_t matrix_size_C = static_cast<size_t>(M) * N;

    for (int i = 0; i < batch_size; ++i) {
        A_array[i] = h_A.data() + i * matrix_size_A;
        B_array[i] = h_B.data() + i * matrix_size_B;
        C_array[i] = h_C.data() + i * matrix_size_C;
    }

    // MKL parameters
    MKL_INT m_mkl = M;
    MKL_INT n_mkl = N;
    MKL_INT k_mkl = K;
    MKL_INT lda_mkl = K; // Leading dimension of A
    MKL_INT ldb_mkl = N; // Leading dimension of B
    MKL_INT ldc_mkl = N; // Leading dimension of C
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;
    float alpha = 1.0f;
    float beta = 0.0f;
    MKL_INT group_size = batch_size; // Number of matrices in the batch

    // Warm-up run
    cblas_sgemm_batch(CblasRowMajor, &transA, &transB, &m_mkl, &n_mkl, &k_mkl,
                      &alpha, A_array.data(), &lda_mkl, B_array.data(), &ldb_mkl,
                      &beta, C_array.data(), &ldc_mkl, 1, &group_size);

    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < num_iterations; ++iter) {
        cblas_sgemm_batch(CblasRowMajor, &transA, &transB, &m_mkl, &n_mkl, &k_mkl,
                          &alpha, A_array.data(), &lda_mkl, B_array.data(), &ldb_mkl,
                          &beta, C_array.data(), &ldc_mkl, 1, &group_size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    return elapsed.count() / num_iterations;
}

// --- CUDA GPU Benchmark ---
double benchmark_gpu_cublas(const std::vector<float>& h_A, const std::vector<float>& h_B, std::vector<float>& h_C,
                            int batch_size, int M, int N, int K, int num_iterations) {

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t size_A = static_cast<size_t>(batch_size) * M * K * sizeof(float);
    size_t size_B = static_cast<size_t>(batch_size) * K * N * sizeof(float);
    size_t size_C = static_cast<size_t>(batch_size) * M * N * sizeof(float);

    // Allocate memory on GPU
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // Copy data from Host to GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    // No need to copy C if beta is 0, but let's clear it just in case
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // cuBLAS parameters (Note: cuBLAS uses column-major order by default,
    // but cublasSgemmBatched allows specifying strides, which makes row-major easier)
    int lda = K; // Leading dimension of A (stride between rows)
    int ldb = N; // Leading dimension of B
    int ldc = N; // Leading dimension of C
    long long int strideA = static_cast<long long int>(M) * K; // Stride between matrices in batch A
    long long int strideB = static_cast<long long int>(K) * N; // Stride between matrices in batch B
    long long int strideC = static_cast<long long int>(M) * N; // Stride between matrices in batch C
    float alpha = 1.0f;
    float beta = 0.0f;

    // Pointers to device memory arrays (needed for batched operation)
    // We can calculate these pointers on the host and pass them to cuBLAS,
    // or create arrays of pointers on the device. Calculating on host is simpler here.
    std::vector<const float*> d_A_array(batch_size);
    std::vector<const float*> d_B_array(batch_size);
    std::vector<float*>       d_C_array(batch_size);
    for(int i = 0; i < batch_size; ++i) {
        d_A_array[i] = d_A + i * strideA;
        d_B_array[i] = d_B + i * strideB;
        d_C_array[i] = d_C + i * strideC;
    }

    // Allocate memory on the device for the arrays of pointers
    const float **d_A_ptrs = nullptr;
    const float **d_B_ptrs = nullptr;
    float **d_C_ptrs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A_ptrs, batch_size * sizeof(const float*)));
    CUDA_CHECK(cudaMalloc(&d_B_ptrs, batch_size * sizeof(const float*)));
    CUDA_CHECK(cudaMalloc(&d_C_ptrs, batch_size * sizeof(float*)));

    // Copy the arrays of pointers from host to device
    CUDA_CHECK(cudaMemcpy(d_A_ptrs, d_A_array.data(), batch_size * sizeof(const float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_ptrs, d_B_array.data(), batch_size * sizeof(const float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_ptrs, d_C_array.data(), batch_size * sizeof(float*), cudaMemcpyHostToDevice));


    // Warm-up run
    // Note: Assuming row-major storage, A is M x K, B is K x N.
    // In cuBLAS column-major default: B^T * A^T = C^T.
    // To compute C = A * B with row-major data without transposing A and B explicitly:
    // Treat row-major A (M x K) as column-major A' (K x M) with lda=M.
    // Treat row-major B (K x N) as column-major B' (N x K) with ldb=N.
    // Compute C' = B' * A' (N x M) using cublasSgemm.
    // The result C' (N x M) stored column-major is equivalent to C (M x N) stored row-major.
    // However, cublasSgemmBatched is more flexible. We can directly use it with row-major layouts
    // by setting the leading dimensions correctly and NOT transposing.
    // Operation: C = alpha * A * B + beta * C
    CUBLAS_CHECK(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, // No Transpose
                                    N, M, K, // Note the order: N, M, K for row-major C=A*B -> C^T=B^T*A^T
                                    &alpha,
                                    d_B_ptrs, ldb, // B is KxN (row-major), treat as NxK (col-major) ldb=N
                                    d_A_ptrs, lda, // A is MxK (row-major), treat as KxM (col-major) lda=K
                                    &beta,
                                    d_C_ptrs, ldc, // C is MxN (row-major), treat as NxM (col-major) ldc=N
                                    batch_size));


    // Synchronize after warm-up
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < num_iterations; ++iter) {
         CUBLAS_CHECK(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    N, M, K,
                                    &alpha,
                                    d_B_ptrs, ldb,
                                    d_A_ptrs, lda,
                                    &beta,
                                    d_C_ptrs, ldc,
                                    batch_size));
    }
    // Synchronize device to ensure all GPU operations are finished before stopping timer
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;


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


int main(int argc, char** argv) {
    // --- Configuration ---
    int batch_size = 64;
    int M = 128;
    int N = 128;
    int K = 128;
    int num_iterations = 100; // Number of iterations for averaging

    // --- Initialization ---
    std::vector<float> h_A, h_B;
    initialize_matrices(h_A, h_B, batch_size, M, N, K);
    std::vector<float> h_C_cpu(static_cast<size_t>(batch_size) * M * N);
    std::vector<float> h_C_gpu(static_cast<size_t>(batch_size) * M * N); // Separate result buffer for GPU if needed

    std::cout << "Benchmarking Batch Matrix Multiplication (Batch=" << batch_size
              << ", M=" << M << ", N=" << N << ", K=" << K << ", Iterations=" << num_iterations << ")" << std::endl;
    std::cout << "Matrix A: " << batch_size << " x (" << M << " x " << K << ")" << std::endl;
    std::cout << "Matrix B: " << batch_size << " x (" << K << " x " << N << ")" << std::endl;
    std::cout << "Matrix C: " << batch_size << " x (" << M << " x " << N << ")" << std::endl;


    // --- CPU Benchmark ---
    try {
        double cpu_avg_time = benchmark_cpu_mkl(h_A, h_B, h_C_cpu, batch_size, M, N, K, num_iterations);
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

            double gpu_avg_time = benchmark_gpu_cublas(h_A, h_B, h_C_gpu, batch_size, M, N, K, num_iterations);
            std::cout << "GPU (cuBLAS) Average Time: " << gpu_avg_time << " ms" << std::endl;

            // Optional: Add verification step comparing h_C_cpu and h_C_gpu

        } catch (const std::exception& e) {
            std::cerr << "GPU Benchmark Error: " << e.what() << std::endl;
        }
    }


    return 0;
}
