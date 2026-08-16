// gemm.cu — Day 2 手写 SGEMM: naive -> shared-memory tiling -> float4
// C = A x B, 行主序, 单精度
//
// 编译:
//   nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 gemm.cu -o gemm.exe -lcublas
// 运行:
//   ./gemm.exe [M] [N] [K]     (默认 1024 1024 1024)
//
// 说明: 需要链接 cuBLAS 库 (-lcublas), 作为性能天花板参考。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE 32  // 共享内存 tile 边长

// ================= 计时工具 =================
template <typename F>
float time_ms(F launch, int iters) {
    launch();  // warmup: 让 cuBLAS heuristics / 上下文稳定
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

// ============ Kernel 0: naive 三重循环 ============
// 每个线程算 C 的一个元素。B 按列访问 -> 不合并。
__global__ void sgemm_naive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.f;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ============ Kernel 1: 共享内存 tiling ============
// block = TILE x TILE, 先协同加载 A/B 的 tile 到 __shared__, 再算。
// 每个线程算 C 的一个元素, 访存全部变成合并访问。
__global__ void sgemm_tiled(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        // 协同加载: 每线程 1 个元素, 边界外补 0
        As[threadIdx.y][threadIdx.x] =
            (row < M && k0 + threadIdx.x < K) ? A[row * K + k0 + threadIdx.x] : 0.f;
        Bs[threadIdx.y][threadIdx.x] =
            (k0 + threadIdx.y < K && col < N) ? B[(k0 + threadIdx.y) * N + col] : 0.f;
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();  // 防止下一个 tile 覆盖当前正在读的数据
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ============ Kernel 2: tiling + float4 向量化 ============
// block = (8, 32) 共 256 线程: 32 行 x 32 列, 每个线程算 1 行 x 4 列。
// B tile 以 float4 加载, C 以 float4 写回 -> 访存指令数减到 1/4。
__global__ void sgemm_vec4(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    __shared__ float  As[TILE][TILE];          // 32x32
    __shared__ float4 Bs[TILE][TILE / 4];      // 32 行 x 8 个 float4 (=32 列)

    const int tx = threadIdx.x;   // 0..7  (每线程 4 列)
    const int ty = threadIdx.y;   // 0..31

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx * 4;      // 本线程负责的 4 列起点

    float4 c = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int k0 = 0; k0 < K; k0 += TILE) {
        // --- 加载 As: 256 线程填 1024 个元素, 每线程 4 个 ---
        for (int idx = ty * 8 + tx; idx < TILE * TILE; idx += 256) {
            int ar = idx / TILE, ac = idx % TILE;
            int gr = blockIdx.y * TILE + ar, gc = k0 + ac;
            As[ar][ac] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
        // --- 加载 Bs: 每线程一个 float4 (32 行 x 8 组) ---
        {
            int br = k0 + ty;                       // B 行号
            int bc = blockIdx.x * TILE + tx * 4;    // B 列起点
            if (br < K && bc + 3 < N)
                Bs[ty][tx] = *reinterpret_cast<const float4*>(B + br * N + bc);
            else
                Bs[ty][tx] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            float a = As[ty][k];
            float4 b = Bs[k][tx];
            c.x += a * b.x; c.y += a * b.y; c.z += a * b.z; c.w += a * b.w;
        }
        __syncthreads();
    }
    if (row < M && col + 3 < N)
        *reinterpret_cast<float4*>(C + row * N + col) = c;
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
int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float *h_A = (float*)malloc(bytesA);
    float *h_B = (float*)malloc(bytesB);
    float *h_C = (float*)malloc(bytesC);
    float *h_ref = (float*)malloc(bytesC);
    if (!h_A || !h_B || !h_C || !h_ref) { printf("malloc failed\n"); return 1; }

    for (int i = 0; i < M * K; ++i) h_A[i] = ((float)(i % 7) - 3.f) * 0.1f;
    for (int i = 0; i < K * N; ++i) h_B[i] = ((float)(i % 11) - 5.f) * 0.1f;

    float *d_A, *d_B, *d_C1, *d_C2, *d_C3;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C1, bytesC);
    cudaMalloc(&d_C2, bytesC);
    cudaMalloc(&d_C3, bytesC);
    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

    double flops = 2.0 * (double)M * N * K;  // 一次乘 + 一次加

    // 启动配置
    dim3 block0(16, 16);
    dim3 grid0((N + 15) / 16, (M + 15) / 16);
    dim3 block1(TILE, TILE);
    dim3 grid1((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    dim3 block2(8, 32);                      // vec4: 8x32 线程 = 32 列 x 32 行
    dim3 grid2((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    auto l0 = [&]() { sgemm_naive<<<grid0, block0>>>(d_A, d_B, d_C1, M, N, K); };
    auto l1 = [&]() { sgemm_tiled<<<grid1, block1>>>(d_A, d_B, d_C2, M, N, K); };
    auto l2 = [&]() { sgemm_vec4 <<<grid2, block2>>>(d_A, d_B, d_C3, M, N, K); };

    // ---- 正确性: 以 naive 输出为参考 ----
    l0(); cudaDeviceSynchronize();
    cudaMemcpy(h_ref, d_C1, bytesC, cudaMemcpyDeviceToHost);
    float cscale = 0.f;
    for (int i = 0; i < M * N; ++i) cscale = fmax(cscale, fabsf(h_ref[i]));

    l1(); l2(); cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C2, bytesC, cudaMemcpyDeviceToHost);
    double err_tiled = max_rel_err(h_ref, h_C, M * N, cscale);
    cudaMemcpy(h_C, d_C3, bytesC, cudaMemcpyDeviceToHost);
    double err_vec4 = max_rel_err(h_ref, h_C, M * N, cscale);

    // ---- 计时 ----
    int iters_naive = 3, iters_opt = 10;
    float t0 = time_ms(l0, iters_naive);
    float t1 = time_ms(l1, iters_opt);
    float t2 = time_ms(l2, iters_opt);

    // ---- cuBLAS 参考 ----
    cublasHandle_t hc;
    cublasCreate(&hc);
    float alpha = 1.f, beta = 0.f;
    auto lc = [&]() {
        // 行主序 C = A*B 等价于列主序 C' = B'*A'(B'=B^T), 直接用 OP_N
        cublasSgemm(hc, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, d_B, N, d_A, K, &beta, d_C1, N);
    };
    float tc = time_ms(lc, iters_opt);

    // ---- 报告 ----
    printf("M=%d N=%d K=%d  (GFLOPS = 2*M*N*K / t)\n", M, N, K);
    printf("--------------------------------------------------------------\n");
    printf("%-16s %10s %12s %8s\n", "version", "ms", "GFLOPS", "speedup");
    printf("%-16s %10.4f %12.1f %8s\n", "naive", t0,
           flops / (t0 / 1e3) / 1e9, "1x");
    printf("%-16s %10.4f %12.1f %8.2f\n", "tiled", t1,
           flops / (t1 / 1e3) / 1e9, t0 / t1);
    printf("%-16s %10.4f %12.1f %8.2f\n", "tiled+vec4", t2,
           flops / (t2 / 1e3) / 1e9, t0 / t2);
    printf("%-16s %10.4f %12.1f %8.2f\n", "cuBLAS", tc,
           flops / (tc / 1e3) / 1e9, t0 / tc);
    printf("--------------------------------------------------------------\n");
    printf("correctness (rel.err vs naive): tiled=%g vec4=%g (PASS if <1e-3)\n",
           err_tiled, err_vec4);
    printf("RTX4070 refs: FP32 ~15 TFLOPS, cuBLAS usually reaches 7-12 TFLOPS.\n");

    cublasDestroy(hc);
    cudaFree(d_A); cudaFree(d_B);
    cudaFree(d_C1); cudaFree(d_C2); cudaFree(d_C3);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}

