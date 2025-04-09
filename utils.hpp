#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include <cstdio> // For fprintf
#include <cstdlib> // For exit
#include <cuda_runtime.h> // For cudaError_t, cudaGetErrorString
#include <cublas_v2.h> // For cublasStatus_t

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
template<typename T>
void initialize_matrices(std::vector<T>& h_A, std::vector<T>& h_B, int batch_size, int M, int N, int K);

// Argument Parsing Helpers
int get_arg(int argc, char** argv, const std::string& flag, int default_val);
std::string get_arg_str(int argc, char** argv, const std::string& flag, const std::string& default_val);
void print_usage(const char* prog_name);


#endif // UTILS_HPP
