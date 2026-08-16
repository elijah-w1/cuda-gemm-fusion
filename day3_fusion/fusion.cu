// fusion.cu — Day 3 算子融合: y = relu(A x B + bias)
// 手写 CUDA，不依赖 cuBLAS/Triton
//
// 编译:  nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 fusion.cu -o fusion.exe
// 运行:  ./fusion.exe [N]     默认跑 128 和 2048 两档对比
//
// 分离版: gemm -> bias -> relu (3 次 launch, C 写 3 次)
// 融合版: 1 个 kernel 完成全部 (1 次 launch, C 写 1 次)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32

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

// ============ 分离版 1/3: GEMM (tiled, Day 2 的 kernel) ============
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

// ============ 分离版 2/3: 加 bias  C[i][j] += bias[j] ============
__global__ void bias_kernel(float* C, const float* bias, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
        C[row * N + col] += bias[col];
}

// ============ 分离版 3/3: ReLU  C = max(0, C) ============
__global__ void relu_kernel(float* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
        C[row * N + col] = fmaxf(0.f, C[row * N + col]);
}

// ============ 融合版: relu(A x B + bias) 一个 kernel ============
// 与 sgemm_tiled 的唯一区别: 写 C 时直接 fmaxf(0, sum + bias[col])
__global__ void sgemm_fused(const float* A, const float* B, const float* bias,
                            float* C, int M, int N, int K) {
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
        C[row * N + col] = fmaxf(0.f, sum + bias[col]);  // 融合点!
}

// ================= 正确性校验 =================
double max_rel_err(const float* a, const float* b, int n, float scale) {
    double err = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > err) err = d;
    }
    return err / (scale > 0.f ? scale : 1.0);
}

// ================= 主流程 =================
void run_case(int N, int iters) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_bias = (float*)malloc((size_t)N * sizeof(float));
    float *h_C1 = (float*)malloc(bytes);
    float *h_C2 = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_bias || !h_C1 || !h_C2) { printf("malloc failed\n"); return; }

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = ((float)(i % 7) - 3.f) * 0.1f;
        h_B[i] = ((float)(i % 11) - 5.f) * 0.1f;
    }
    for (int i = 0; i < N; ++i) h_bias[i] = ((float)(i % 5) - 2.f) * 0.2f;

    float *d_A, *d_B, *d_bias, *d_C1, *d_C2;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_bias, (size_t)N * sizeof(float));
    cudaMalloc(&d_C1, bytes);
    cudaMalloc(&d_C2, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, (size_t)N * sizeof(float), cudaMemcpyHostToDevice);

    // 启动配置 (block 32x32)
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    auto launch_separate = [&]() {
        sgemm_tiled<<<grid, block>>>(d_A, d_B, d_C1, N, N, N);
        bias_kernel<<<grid, block>>>(d_C1, d_bias, N, N);
        relu_kernel<<<grid, block>>>(d_C1, N, N);
    };
    auto launch_fused = [&]() {
        sgemm_fused<<<grid, block>>>(d_A, d_B, d_bias, d_C2, N, N, N);
    };

    // 正确性: 两版结果对比
    launch_separate();
    launch_fused();
    cudaDeviceSynchronize();
    cudaMemcpy(h_C1, d_C1, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2, d_C2, bytes, cudaMemcpyDeviceToHost);
    float scale = 0.f;
    for (int i = 0; i < N * N; ++i) scale = fmax(scale, fabsf(h_C1[i]));
    double err = max_rel_err(h_C1, h_C2, N * N, scale);

    // 计时
    float t_sep = time_ms(launch_separate, iters);
    float t_fus = time_ms(launch_fused, iters);

    printf("N=%4d  separate(gemm+bias+relu): %9.4f ms | fused(1 kernel): %9.4f ms | saved %6.1f%% | rel_err=%g\n",
           N, t_sep, t_fus, (1.0 - t_fus / t_sep) * 100.0, err);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_C1); cudaFree(d_C2);
    free(h_A); free(h_B); free(h_bias); free(h_C1); free(h_C2);
}

int main(int argc, char** argv) {
    printf("y = relu(A x B + bias),  row-major fp32\n");
    printf("separate = 3 kernels (gemm, bias, relu) | fused = 1 kernel\n");
    printf("------------------------------------------------------------------------\n");
    if (argc > 1) {
        int N = atoi(argv[1]);
        int iters = (N >= 2048) ? 5 : 50;
        run_case(N, iters);
    } else {
        run_case(128, 50);    // 小矩阵: launch 开销主导
        run_case(2048, 5);    // 大矩阵: 访存主导
    }
    return 0;
}

