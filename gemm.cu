/*
 * Simple CUDA GEMM: C = alpha * A * B + beta * C
 * Row-major: A is MxK, B is KxN, C is MxN
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

static constexpr int kWarmupRuns = 10;
static constexpr int kTimedRuns = 10;

template <typename T>
__global__ void gemm_naive(
    int M, int N, int K,
    T alpha,
    const T* __restrict__ A,
    const T* __restrict__ B,
    T beta,
    T* __restrict__ C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N)
        return;

    T acc = static_cast<T>(0);
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];

    int idx = row * N + col;
    C[idx] = alpha * acc + beta * C[idx];
}

static void check_cuda(cudaError_t e, const char* msg)
{
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

static void check_cublas(cublasStatus_t s, const char* msg)
{
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: cublas status %d\n", msg, static_cast<int>(s));
        std::exit(1);
    }
}

static void print_cuda_device_props(int device = 0)
{
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    std::printf("GPU device %d: %s\n", device, prop.name);
    std::printf("SM count: %d\n", prop.multiProcessorCount);
    std::printf("Shared mem per block: %.1f KB\n", prop.sharedMemPerBlock / 1024.0);
    std::printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    std::printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    std::printf("Max warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / 32);
}

int main()
{
    print_cuda_device_props(0);

    using T = float;
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    const T alpha = 1.0f;
    const T beta = 0.0f;

    const size_t szA = static_cast<size_t>(M) * K * sizeof(T);
    const size_t szB = static_cast<size_t>(K) * N * sizeof(T);
    const size_t szC = static_cast<size_t>(M) * N * sizeof(T);

    T *hA = static_cast<T*>(std::malloc(szA));
    T *hB = static_cast<T*>(std::malloc(szB));
    T *hC_gpu = static_cast<T*>(std::malloc(szC));
    T *hC_blas = static_cast<T*>(std::malloc(szC));
    if (!hA || !hB || !hC_gpu || !hC_blas) {
        std::fprintf(stderr, "host malloc failed\n");
        return 1;
    }

    for (int i = 0; i < M * K; ++i)
        hA[i] = static_cast<T>((i % 17) * 0.01f);
    for (int i = 0; i < K * N; ++i)
        hB[i] = static_cast<T>((i % 23) * 0.01f);
    for (int i = 0; i < M * N; ++i)
        hC_gpu[i] = hC_blas[i] = static_cast<T>(0);

    T *dA = nullptr, *dB = nullptr, *dC = nullptr;
    check_cuda(cudaMalloc(&dA, szA), "cudaMalloc A");
    check_cuda(cudaMalloc(&dB, szB), "cudaMalloc B");
    check_cuda(cudaMalloc(&dC, szC), "cudaMalloc C");

    check_cuda(cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice), "H2D A");
    check_cuda(cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice), "H2D B");
    check_cuda(cudaMemcpy(dC, hC_gpu, szC, cudaMemcpyHostToDevice), "H2D C");

    // dim3 block(8, 32);
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    cudaEvent_t t0, t1;
    check_cuda(cudaEventCreate(&t0), "event0");
    check_cuda(cudaEventCreate(&t1), "event1");

    for (int r = 0; r < kWarmupRuns; ++r) {
        gemm_naive<T><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
        check_cuda(cudaGetLastError(), "warmup kernel");
    }
    check_cuda(cudaDeviceSynchronize(), "warmup sync");

    float total_ms = 0.f;
    for (int r = 0; r < kTimedRuns; ++r) {
        check_cuda(cudaEventRecord(t0), "record t0");
        gemm_naive<T><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
        check_cuda(cudaGetLastError(), "kernel launch");
        check_cuda(cudaEventRecord(t1), "record t1");
        check_cuda(cudaEventSynchronize(t1), "sync t1");
        float ms_iter = 0.f;
        check_cuda(cudaEventElapsedTime(&ms_iter, t0, t1), "elapsed");
        total_ms += ms_iter;
    }
    const float ms = total_ms / static_cast<float>(kTimedRuns);

    check_cuda(cudaMemcpy(hC_gpu, dC, szC, cudaMemcpyDeviceToHost), "D2H C");

    double flops = 2.0 * static_cast<double>(M) * N * K;
    double gflops_naive = (flops / 1e9) / (static_cast<double>(ms) / 1000.0);

    std::printf("GEMM %dx%dx%d  row-major  naive kernel\n", M, N, K);
    std::printf(
        "GPU time (avg of %d timed runs, after %d warmup): %.3f ms  (~%.2f GFLOP/s)\n",
        kTimedRuns,
        kWarmupRuns,
        ms,
        gflops_naive);
    cublasHandle_t blas_handle{};
    check_cublas(cublasCreate(&blas_handle), "cublasCreate");
    // Row-major C(MxN) = A(MxK)*B(KxN): use SGEMM with NN and leading dims = row strides (K,N,N).
    for (int r = 0; r < kWarmupRuns; ++r) {
        check_cublas(
            cublasSgemm(
                blas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                M,
                K,
                &alpha,
                dB,
                N,
                dA,
                K,
                &beta,
                dC,
                N),
            "cublasSgemm warmup");
    }
    check_cuda(cudaDeviceSynchronize(), "cublas warmup sync");

    float total_ms_blas = 0.f;
    for (int r = 0; r < kTimedRuns; ++r) {
        check_cuda(cudaEventRecord(t0), "cublas record t0");
        check_cublas(
            cublasSgemm(
                blas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                M,
                K,
                &alpha,
                dB,
                N,
                dA,
                K,
                &beta,
                dC,
                N),
            "cublasSgemm timed");
        check_cuda(cudaEventRecord(t1), "cublas record t1");
        check_cuda(cudaEventSynchronize(t1), "cublas sync t1");
        float ms_iter = 0.f;
        check_cuda(cudaEventElapsedTime(&ms_iter, t0, t1), "cublas elapsed");
        total_ms_blas += ms_iter;
    }
    const float ms_blas = total_ms_blas / static_cast<float>(kTimedRuns);

    check_cuda(cudaMemcpy(hC_blas, dC, szC, cudaMemcpyDeviceToHost), "D2H C (cublas)");

    double max_diff_naive_blas = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double e = std::fabs(static_cast<double>(hC_gpu[i] - hC_blas[i]));
        if (e > max_diff_naive_blas)
            max_diff_naive_blas = e;
    }

    double gflops_blas = (flops / 1e9) / (static_cast<double>(ms_blas) / 1000.0);
    std::printf("cuBLAS SGEMM (row-major data; NN, m=%d n=%d k=%d, lda=%d ldb=%d ldc=%d)\n",
        N,
        M,
        K,
        N,
        K,
        N);
    std::printf(
        "GPU time (avg of %d timed runs, after %d warmup): %.3f ms  (~%.2f GFLOP/s)\n",
        kTimedRuns,
        kWarmupRuns,
        ms_blas,
        gflops_blas);
    std::printf("max |naive - cuBLAS| (host): %.6e\n", max_diff_naive_blas);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    check_cublas(cublasDestroy(blas_handle), "cublasDestroy");
    std::free(hA);
    std::free(hB);
    std::free(hC_gpu);
    std::free(hC_blas);
    const bool ok = max_diff_naive_blas < 1e-2;
    return ok ? 0 : 1;
}
