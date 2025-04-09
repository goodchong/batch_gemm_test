#ifndef CPU_BENCHMARK_HPP
#define CPU_BENCHMARK_HPP

#include <vector>
#include <chrono>
#include <stdexcept>
#include <type_traits> // For std::is_same
#include <mkl.h>       // For Intel MKL

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

#endif // CPU_BENCHMARK_HPP
