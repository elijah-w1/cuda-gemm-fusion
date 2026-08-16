// softmax_fusion.cu — Day 3 升级: 归约类融合 softmax(A*B + bias)
// 演示 relu(elementwise) 能简单折叠 vs softmax(归约) 需要特殊布局
//
// 关键设计: 每个 block 覆盖"完整行"(N=256 列) → 行归约在 block 内完成
// block = (32, 32): 32 行 × 256 列, 每线程算 1 行 × 8 列
// warp 0 = ty=0 的 32 个线程 = 同一行 → 行归约用 __shfl_xor_sync
//
// 编译:  nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 softmax_fusion.cu -o softmax_fusion.exe
// 运行:  ./softmax_fusion.exe [M] [K]   默认 M=1024 K=512 (N 固定 256)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32      // K 方向 tiling
#define NCOLS 256    // N 固定 = 256 (每 block 覆盖整行)
#define ROWS 32      // 每 block 处理 32 行
#define CPT 8        // 每线程 8 列 (32 线程 x 方向 * 8 = 256 列)

// ================= 计时工具 =================
template <typename F>
float time_ms(F launch, int iters) {
    launch();  // warmup
    cudaDeviceSynchronize();
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / iters;
}

// ============ 融合版: softmax(A*B + bias) 一个 kernel ============
// 每线程算 1 行(row) x 8 列(列起点 tx*8)
__global__ void gemm_softmax_fused(const float* A, const float* B,
                                   const float* bias, float* C, int M, int K) {
    
    const int tx = threadIdx.x;   // 0..31  列方向：本线程管哪 8 列
    const int ty = threadIdx.y;   // 0..31  行方向：本线程管哪一行
    const int row = blockIdx.y * ROWS + ty;   // 全局行号（ROWS=32）

    __shared__ float As[TILE][TILE];          // A tile 32x32 = 4KB
    __shared__ float Bs[TILE][NCOLS];         // B tile 32x256 = 32KB (共36KB<48KB)

    float c[CPT];
#pragma unroll
    for (int i = 0; i < CPT; ++i) c[i] = 0.f;

    // ---------- GEMM (tiling over K) ----------
    for (int k0 = 0; k0 < K; k0 += TILE) {
        // 加载 As: 每线程 1 个元素
        As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.f;
        // 加载 Bs: 每线程 8 个连续元素 (32 行 x 256 列)
        if (k0 + ty < K) {
#pragma unroll
            for (int i = 0; i < CPT; ++i)
                Bs[ty][tx * CPT + i] = B[(k0 + ty) * NCOLS + tx * CPT + i];
        } else {
#pragma unroll
            for (int i = 0; i < CPT; ++i) Bs[ty][tx * CPT + i] = 0.f;
        }
        __syncthreads();
        for (int k = 0; k < TILE; ++k) {
            const float a = As[ty][k];
#pragma unroll
            for (int i = 0; i < CPT; ++i)
                c[i] += a * Bs[k][tx * CPT + i];
        }
        __syncthreads();
    }
    if (row >= M) return;

    // ---------- softmax (每行归约: 同一行的 32 线程 = 一个 warp) ----------
    // 1) 加 bias, 得到 v[i] = A*B + bias
    float v[CPT];
#pragma unroll
    for (int i = 0; i < CPT; ++i) v[i] = c[i] + bias[tx * CPT + i];

    // 2) 每线程局部 max (8 个值) -> warp 归约得行 max
    float local_max = v[0];
#pragma unroll
    for (int i = 1; i < CPT; ++i) local_max = fmaxf(local_max, v[i]);
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));

    // 3) exp(v - max), 每线程局部 sum -> warp 归约得行 sum
    float local_sum = 0.f;
#pragma unroll
    for (int i = 0; i < CPT; ++i) {
        v[i] = __expf(v[i] - local_max);   // 数值稳定 softmax
        local_sum += v[i];
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, off);

    // 4) 归一写回
#pragma unroll
    for (int i = 0; i < CPT; ++i)
        C[row * NCOLS + tx * CPT + i] = v[i] / local_sum;
}

// ============ 分离版 1/3: GEMM (Day 3 的 tiled 版本) ============
__global__ void sgemm_tiled(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        As[threadIdx.y][threadIdx.x] =
            (row < M && k0 + threadIdx.x < K) ? A[row * K + k0 + threadIdx.x] : 0.f;
        Bs[threadIdx.y][threadIdx.x] =
            (k0 + threadIdx.y < K && col < N) ? B[(k0 + threadIdx.y) * N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ============ 分离版 2/3: 加 bias ============
__global__ void bias_kernel(float* C, const float* bias, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
        C[row * N + col] += bias[col];
}

// ============ 分离版 3/3: softmax (独立 kernel, 每行 warp 归约) ============
__global__ void softmax_kernel(float* C, int M) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * ROWS + ty;
    if (row >= M) return;

    float v[CPT];
#pragma unroll
    for (int i = 0; i < CPT; ++i) v[i] = C[row * NCOLS + tx * CPT + i];

    float local_max = v[0];
#pragma unroll
    for (int i = 1; i < CPT; ++i) local_max = fmaxf(local_max, v[i]);
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));

    float local_sum = 0.f;
#pragma unroll
    for (int i = 0; i < CPT; ++i) {
        v[i] = __expf(v[i] - local_max);
        local_sum += v[i];
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, off);

#pragma unroll
    for (int i = 0; i < CPT; ++i)
        C[row * NCOLS + tx * CPT + i] = v[i] / local_sum;
}

// ================= 校验 =================
double max_rel_err(const float* a, const float* b, int n, float scale) {
    double err = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > err) err = d;
    }
    return err / (scale > 0.f ? scale : 1.0);
}

// ================= 主流程 =================
void run_case(int M, int K, int iters) {
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * NCOLS * sizeof(float);
    size_t bytesC = (size_t)M * NCOLS * sizeof(float);

    float *h_A = (float*)malloc(bytesA);
    float *h_B = (float*)malloc(bytesB);
    float *h_bias = (float*)malloc((size_t)NCOLS * sizeof(float));
    float *h_C1 = (float*)malloc(bytesC);
    float *h_C2 = (float*)malloc(bytesC);
    if (!h_A || !h_B || !h_bias || !h_C1 || !h_C2) { printf("malloc failed\n"); return; }

    for (int i = 0; i < M * K; ++i) h_A[i] = ((float)(i % 7) - 3.f) * 0.1f;
    for (int i = 0; i < K * NCOLS; ++i) h_B[i] = ((float)(i % 11) - 5.f) * 0.1f;
    for (int i = 0; i < NCOLS; ++i) h_bias[i] = ((float)(i % 5) - 2.f) * 0.2f;

    float *d_A, *d_B, *d_bias, *d_C1, *d_C2;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_bias, (size_t)NCOLS * sizeof(float));
    cudaMalloc(&d_C1, bytesC);
    cudaMalloc(&d_C2, bytesC);
    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, (size_t)NCOLS * sizeof(float), cudaMemcpyHostToDevice);

    // 启动配置
    dim3 blockF(32, 32);                       // 融合版 & softmax: 32x32
    dim3 gridF(1, (M + ROWS - 1) / ROWS);
    dim3 blockG(TILE, TILE);                   // GEMM & bias: 32x32
    dim3 gridG((NCOLS + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    auto launch_separate = [&]() {
        sgemm_tiled<<<gridG, blockG>>>(d_A, d_B, d_C1, M, NCOLS, K);
        bias_kernel<<<gridG, blockG>>>(d_C1, d_bias, M, NCOLS);
        softmax_kernel<<<gridF, blockF>>>(d_C1, M);
    };
    auto launch_fused = [&]() {
        gemm_softmax_fused<<<gridF, blockF>>>(d_A, d_B, d_bias, d_C2, M, K);
    };

    // 正确性: 分离版为标准答案
    launch_separate();
    launch_fused();
    cudaDeviceSynchronize();
    cudaMemcpy(h_C1, d_C1, bytesC, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2, d_C2, bytesC, cudaMemcpyDeviceToHost);
    float scale = 0.f;
    for (int i = 0; i < M * NCOLS; ++i) scale = fmax(scale, fabsf(h_C1[i]));
    double err = max_rel_err(h_C1, h_C2, M * NCOLS, scale);

    // 计时
    float t_sep = time_ms(launch_separate, iters);
    float t_fus = time_ms(launch_fused, iters);

    printf("M=%4d K=%4d | separate(gemm+bias+softmax): %9.4f ms | fused(1 kernel): %9.4f ms | saved %6.1f%% | rel_err=%g\n",
           M, K, t_sep, t_fus, (1.0 - t_fus / t_sep) * 100.0, err);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_C1); cudaFree(d_C2);
    free(h_A); free(h_B); free(h_bias); free(h_C1); free(h_C2);
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int K = (argc > 2) ? atoi(argv[2]) : 512;
    printf("y = softmax(A x B + bias),  N fixed = %d\n", NCOLS);
    printf("separate = 3 kernels (gemm, bias, softmax) | fused = 1 kernel (block covers full row)\n");
    printf("------------------------------------------------------------------------\n");
    run_case(M, K, (M >= 4096) ? 5 : 20);
    return 0;
}


