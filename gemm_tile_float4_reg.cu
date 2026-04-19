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

#define TILE_SIZE 32
template <typename T>
__global__ void gemm_block_float4_reg_32_32(
    int M, int N, int K,
    T alpha,
    const T* __restrict__ A,
    const T* __restrict__ B,
    T beta,
    T* __restrict__ C)
{
    int row = blockIdx.y * TILE_SIZE;
    int col = blockIdx.x * TILE_SIZE;

    __shared__ T shared_A[TILE_SIZE * TILE_SIZE];
    __shared__ T shared_B[TILE_SIZE * TILE_SIZE];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads_per_sm_block_row = TILE_SIZE / 4;

    T acc[4] = {0, 0, 0, 0};

    for (int kt = 0; kt < (K + TILE_SIZE - 1) / TILE_SIZE; ++kt) {
        const int a_row = row + tid / threads_per_sm_block_row;
        const int a_col = kt * TILE_SIZE + tid % threads_per_sm_block_row * 4;
        const int b_row = kt * TILE_SIZE + tid / threads_per_sm_block_row;
        const int b_col = col + tid % threads_per_sm_block_row * 4;
        
        const float4* A4 = reinterpret_cast<const float4*>(A + a_row * K + a_col);
        const float4* B4 = reinterpret_cast<const float4*>(B + b_row * N + b_col);
        shared_A[tid * 4] = A4->x;
        shared_A[tid * 4 + 1] = A4->y;
        shared_A[tid * 4 + 2] = A4->z;
        shared_A[tid * 4 + 3] = A4->w;
        shared_B[tid * 4] = B4->x;
        shared_B[tid * 4 + 1] = B4->y;
        shared_B[tid * 4 + 2] = B4->z;
        shared_B[tid * 4 + 3] = B4->w;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            acc[0] += shared_A[tid / threads_per_sm_block_row * TILE_SIZE + i] * shared_B[i * TILE_SIZE + tid % threads_per_sm_block_row * 4];
            acc[1] += shared_A[tid / threads_per_sm_block_row * TILE_SIZE + i] * shared_B[i * TILE_SIZE + tid % threads_per_sm_block_row * 4 + 1];
            acc[2] += shared_A[tid / threads_per_sm_block_row * TILE_SIZE + i] * shared_B[i * TILE_SIZE + tid % threads_per_sm_block_row * 4 + 2];
            acc[3] += shared_A[tid / threads_per_sm_block_row * TILE_SIZE + i] * shared_B[i * TILE_SIZE + tid % threads_per_sm_block_row * 4 + 3];
        }

        __syncthreads();
    }

    const int idx = (row + tid / threads_per_sm_block_row) * N + col + tid % threads_per_sm_block_row * 4;
    C[idx] = alpha * acc[0] + beta * C[idx];
    C[idx + 1] = alpha * acc[1] + beta * C[idx + 1];
    C[idx + 2] = alpha * acc[2] + beta * C[idx + 2];
    C[idx + 3] = alpha * acc[3] + beta * C[idx + 3];
}

/*
 * Tiled GEMM: BM=128, BN=128, BK=8, TM=TN=8, blockDim (16,16).
 * Row-major C(M×N) = alpha*A(M×K)*B(K×N) + beta*C.
 * Grid must be ((N+BN-1)/BN, (M+BM-1)/BM). Requires M,N multiple of 128, K multiple of 8, N multiple of 4 for float4 stores.
 */
__global__ void gemm_block_128_8(
    int M,
    int N,
    int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C)
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j)
            r_c[i][j] = 0.f;
    }

    const int load_a_smem_m = tid >> 1;
    const int load_a_smem_k = (tid & 1) << 2;
    const int load_b_smem_k = tid >> 5;
    const int load_b_smem_n = (tid & 31) << 2;

    const int load_a_gmem_m = by * BM + load_a_smem_m;
    const int load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
        const int load_a_gmem_k = bk * BK + load_a_smem_k;
        *reinterpret_cast<float4*>(&s_a[load_a_smem_m][load_a_smem_k]) =
            *reinterpret_cast<const float4*>(A + load_a_gmem_m * K + load_a_gmem_k);

        const int load_b_gmem_k = bk * BK + load_b_smem_k;
        *reinterpret_cast<float4*>(&s_b[load_b_smem_k][load_b_smem_n]) =
            *reinterpret_cast<const float4*>(B + load_b_gmem_k * N + load_b_gmem_n);

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    const int comp_a_smem_m = ty * TM + m;
                    const int comp_b_smem_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
            const int sm = by * BM + ty * TM + i;
            const int sn = bx * BN + tx * TN + j;
            if (sm < M && sn + 3 < N) {
                const int addr = sm * N + sn;
                float4 out{
                    alpha * r_c[i][j + 0] + beta * C[addr + 0],
                    alpha * r_c[i][j + 1] + beta * C[addr + 1],
                    alpha * r_c[i][j + 2] + beta * C[addr + 2],
                    alpha * r_c[i][j + 3] + beta * C[addr + 3]};
                *reinterpret_cast<float4*>(C + addr) = out;
            }
        }
    }
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

    constexpr int BM128 = 128;
    constexpr int BN128 = 128;
    dim3 block(16, 16);
    dim3 grid((N + BN128 - 1) / BN128, (M + BM128 - 1) / BM128);

    cudaEvent_t t0, t1;
    check_cuda(cudaEventCreate(&t0), "event0");
    check_cuda(cudaEventCreate(&t1), "event1");

    for (int r = 0; r < kWarmupRuns; ++r) {
        gemm_block_128_8<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
        check_cuda(cudaGetLastError(), "warmup kernel");
    }
    check_cuda(cudaDeviceSynchronize(), "warmup sync");

    float total_ms = 0.f;
    for (int r = 0; r < kTimedRuns; ++r) {
        check_cuda(cudaEventRecord(t0), "record t0");
        gemm_block_128_8<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
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

    std::printf("GEMM %dx%dx%d  row-major  gemm_block_128_8 (BM=BN=128, BK=8, TM=TN=8)\n", M, N, K);
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
