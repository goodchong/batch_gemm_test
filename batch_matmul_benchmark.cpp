#include <iostream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <random>
#include <numeric>
#include <string>      // For std::string
#include <type_traits> // For std::is_same
#include <cstdlib>     // For atoi, atof
#include <limits>      // For numeric_limits
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
    }
} while (0)


// --- Templated Functions ---

// Function to initialize matrices with random values
template<typename T>
void initialize_matrices(std::vector<T>& h_A, std::vector<T>& h_B, int batch_size, int M, int N, int K) {
    std::mt19937 rng(12345); // Fixed seed for reproducibility
    std::uniform_real_distribution<T> dist(static_cast<T>(0.0), static_cast<T>(1.0));

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
template<typename T>
double benchmark_cpu_mkl(const std::vector<T>& h_A, const std::vector<T>& h_B, std::vector<T>& h_C,
                         int batch_size, int M, int N, int K, int num_iterations) {

    // MKL expects arrays of pointers for batched operations
    std::vector<const T*> A_array(batch_size);
    std::vector<const T*> B_array(batch_size);
    std::vector<T*>       C_array(batch_size);

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
    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);
    MKL_INT group_size = batch_size; // Number of matrices in the batch

    // Select MKL function based on type T
    if constexpr (std::is_same<T, float>::value) {
        // Warm-up run
        cblas_sgemm_batch(CblasRowMajor, &transA, &transB, &m_mkl, &n_mkl, &k_mkl,
                          (float*)&alpha, (const float**)A_array.data(), &lda_mkl, (const float**)B_array.data(), &ldb_mkl,
                          (float*)&beta, (float**)C_array.data(), &ldc_mkl, 1, &group_size);

        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < num_iterations; ++iter) {
            cblas_sgemm_batch(CblasRowMajor, &transA, &transB, &m_mkl, &n_mkl, &k_mkl,
                              (float*)&alpha, (const float**)A_array.data(), &lda_mkl, (const float**)B_array.data(), &ldb_mkl,
                              (float*)&beta, (float**)C_array.data(), &ldc_mkl, 1, &group_size);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count() / num_iterations;
    } else if constexpr (std::is_same<T, double>::value) {
         // Warm-up run
        cblas_dgemm_batch(CblasRowMajor, &transA, &transB, &m_mkl, &n_mkl, &k_mkl,
                          (double*)&alpha, (const double**)A_array.data(), &lda_mkl, (const double**)B_array.data(), &ldb_mkl,
                          (double*)&beta, (double**)C_array.data(), &ldc_mkl, 1, &group_size);

        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < num_iterations; ++iter) {
             cblas_dgemm_batch(CblasRowMajor, &transA, &transB, &m_mkl, &n_mkl, &k_mkl,
                              (double*)&alpha, (const double**)A_array.data(), &lda_mkl, (const double**)B_array.data(), &ldb_mkl,
                              (double*)&beta, (double**)C_array.data(), &ldc_mkl, 1, &group_size);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count() / num_iterations;
    } else {
        throw std::runtime_error("Unsupported data type for MKL benchmark");
    }
}


// --- CUDA GPU Benchmark ---
template<typename T>
double benchmark_gpu_cublas(const std::vector<T>& h_A, const std::vector<T>& h_B, std::vector<T>& h_C,
                            int batch_size, int M, int N, int K, int num_iterations) {

    T *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t size_A = static_cast<size_t>(batch_size) * M * K * sizeof(T);
    size_t size_B = static_cast<size_t>(batch_size) * K * N * sizeof(T);
    size_t size_C = static_cast<size_t>(batch_size) * M * N * sizeof(T);
    // Note: 'elapsed' is now calculated inside the if/else blocks below

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

    // cuBLAS parameters (Note: cuBLAS uses column-major order by default,
    // but cublasSgemmBatched allows specifying strides, which makes row-major easier)
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


    // Pointers to device memory arrays (needed for batched operation)
    // We can calculate these pointers on the host and pass them to cuBLAS,
    // or create arrays of pointers on the device. Calculating on host is simpler here.
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
    // Select cuBLAS function based on type T
    cublasStatus_t cublas_status;
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

        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < num_iterations; ++iter) {
             cublas_status = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                               N, M, K,
                                               reinterpret_cast<const float*>(alpha),
                                               reinterpret_cast<const float**>(d_B_ptrs), ldb,
                                               reinterpret_cast<const float**>(d_A_ptrs), lda,
                                               reinterpret_cast<const float*>(beta),
                                               reinterpret_cast<float**>(d_C_ptrs), ldc,
                                               batch_size);
             CUBLAS_CHECK(cublas_status); // Check status inside loop if desired, though often omitted for performance measurement
        }
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync before stopping timer
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        // Cleanup and return... (handled below)

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

        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < num_iterations; ++iter) {
             cublas_status = cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                               N, M, K,
                                               reinterpret_cast<const double*>(alpha),
                                               reinterpret_cast<const double**>(d_B_ptrs), ldb,
                                               reinterpret_cast<const double**>(d_A_ptrs), lda,
                                               reinterpret_cast<const double*>(beta),
                                               reinterpret_cast<double**>(d_C_ptrs), ldc,
                                               batch_size);
             CUBLAS_CHECK(cublas_status);
        }
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync before stopping timer
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
         // Cleanup and return... (handled below)

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

    std::chrono::duration<double, std::milli> elapsed = end - start; // This line is now inside the if/else blocks

    // Copy result back to host (optional, for verification)
    // CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    // Note: 'elapsed' calculation moved inside the if/else blocks above


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

    // Note: elapsed calculation moved inside the if/else blocks above
    return elapsed.count() / num_iterations;
}


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


// --- Argument Parsing Helper ---
// Simple argument parser
int get_arg(int argc, char** argv, const std::string& flag, int default_val) {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == flag) {
            try {
                return std::stoi(argv[i + 1]);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing integer value for " << flag << ": " << argv[i + 1] << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
    return default_val;
}

std::string get_arg_str(int argc, char** argv, const std::string& flag, const std::string& default_val) {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == flag) {
            return std::string(argv[i + 1]);
        }
    }
    return default_val;
}

void print_usage(const char* prog_name) {
     std::cerr << "Usage: " << prog_name << " [options]\n"
               << "Options:\n"
               << "  --batch_size <int>   Number of matrices in the batch (default: 64)\n"
               << "  --m <int>            Number of rows of matrix A and C (default: 128)\n"
               << "  --n <int>            Number of columns of matrix B and C (default: 128)\n"
               << "  --k <int>            Number of columns of A / rows of B (default: 128)\n"
               << "  --iterations <int>   Number of benchmark iterations (default: 100)\n"
               << "  --precision <string> Data type: float or double (default: float)\n"
               << "  --help               Show this help message\n";
}


int main(int argc, char** argv) {

    // Check for help flag
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // --- Configuration from Arguments ---
    int batch_size     = get_arg(argc, argv, "--batch_size", 64);
    int M              = get_arg(argc, argv, "--m", 128);
    int N              = get_arg(argc, argv, "--n", 128);
    int K              = get_arg(argc, argv, "--k", 128);
    int num_iterations = get_arg(argc, argv, "--iterations", 100);
    std::string precision = get_arg_str(argc, argv, "--precision", "float");

    // Validate inputs
    if (batch_size <= 0 || M <= 0 || N <= 0 || K <= 0 || num_iterations <= 0) {
        std::cerr << "Error: Dimensions (batch_size, m, n, k) and iterations must be positive." << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    // --- Run Benchmarks based on precision ---
    try {
        if (precision == "float") {
            run_benchmarks<float>(batch_size, M, N, K, num_iterations);
        } else if (precision == "double") {
            run_benchmarks<double>(batch_size, M, N, K, num_iterations);
        } else {
            std::cerr << "Error: Invalid precision specified. Use 'float' or 'double'." << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
         std::cerr << "Benchmark execution failed: " << e.what() << std::endl;
         return 1;
    }

    return 0;
}
