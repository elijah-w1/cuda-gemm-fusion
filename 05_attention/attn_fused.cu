// attn_fused.cu — 综合收官：简化版 Flash Attention 前向融合
// O = softmax(Q K^T / sqrt(d)) V
//
// 朴素版(3 kernel): S=QK^T写回显存 -> softmax -> P·V   (S 占 M*M 显存)
// 融合版(1 kernel): 在线 softmax(running max/sum)，S 矩阵不落显存
//
// 综合运用:
//   阶段 1: 线程模型 / CUDA events 计时 / warmup
//   阶段 2: 共享内存 tiling / 向量化 / 合并访问
//   阶段 3: 算子融合 / 在线 softmax / block 归约
//   阶段 4: 朴素 vs 融合公平对比（控制变量：同一 GEMM 实现）
//
// 编译: nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 attn_fused.cu -o attn_fused.exe
// 运行: ./attn_fused.exe [M] [d]    默认 M=1024 d=64

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32

// ================= 计时 =================
template <typename F>
float time_ms(F launch, int iters) {
    launch(); cudaDeviceSynchronize();
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

// ================= 朴素版 1/3: S = Q·K^T * scale (tiled GEMM, 阶段 2 知识) =====
__global__ void sgemm_tiled(const float* A, const float* B, float* C,
                            int M, int N, int K, float scale) {
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
        C[row * N + col] = sum * scale;
}

// ================= 朴素版 2/3: softmax 每行 (阶段 3.5 的 warp 归约知识) =====
__global__ void softmax_kernel(float* S, int M) {
    __shared__ float red[32];
    const int tx = threadIdx.x;      // 0..31, 每线程处理 M/32 列（grid-stride）
    const int row = blockIdx.x;
    float* srow = S + (size_t)row * M;

    float local_max = -INFINITY;
    for (int c = tx; c < M; c += 32) local_max = fmaxf(local_max, srow[c]);
    // warp 归约 max
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));
    float rowmax = local_max;

    float local_sum = 0.f;
    for (int c = tx; c < M; c += 32) {
        srow[c] = __expf(srow[c] - rowmax);
        local_sum += srow[c];
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, off);
    float rowsum = local_sum;

    for (int c = tx; c < M; c += 32) srow[c] /= rowsum;
}

// ============ 融合版: 在线 softmax attention（S 不写回显存）============
// 每个 block 处理 Q 的一行（block = d 线程），grid = M
// 扫 K/V 的 d 列块，在线更新 running max/sum —— Flash Attention 核心思想
// 正确性依据: softmax 的 max/sum 可以增量更新（数值等价）
__global__ void attn_fused(const float* Q, const float* K, const float* V,
                           float* O, int M, int d, float scale) {
    const int tid = threadIdx.x;       // 0..d-1
    const int row = blockIdx.x;        // 本 block 处理的行
    if (tid >= d || row >= M) return;

    __shared__ float Qs[64];
    __shared__ float Ks[64][64];       // 当前 K 块 (d x d)
    __shared__ float Vs[64][64];       // 当前 V 块 (d x d)
    __shared__ float Ps[64];           // 当前块的 P = exp(S - m_new)
    __shared__ float red[64];          // 归约缓冲

    const float* qrow = Q + (size_t)row * d;
    Qs[tid] = qrow[tid];
    __syncthreads();

    float m = -INFINITY;   // running max
    float l = 0.f;         // running sum
    float o = 0.f;         // O[row][tid] 的累积值（在寄存器）

    for (int jb = 0; jb < M; jb += d) {
        // ① 协同加载 K/V 块（每线程 d 个元素，合并访问）
        for (int idx = tid; idx < d * d; idx += d) {
            Ks[idx / d][idx % d] = K[(size_t)(jb + idx / d) * d + idx % d];
            Vs[idx / d][idx % d] = V[(size_t)(jb + idx / d) * d + idx % d];
        }
        __syncthreads();

        // ② 每线程算 S[jb+tid] = scale * Σ_dd Qs[dd] * Ks[tid][dd]
        float s = 0.f;
        for (int dd = 0; dd < d; ++dd) s += Qs[dd] * Ks[tid][dd];
        s *= scale;

        // ③ 在线 softmax: 新块行 max → 更新 running max
        red[tid] = s; __syncthreads();
        for (int off = d / 2; off > 0; off >>= 1) {
            if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
            __syncthreads();
        }
        float m_new = fmaxf(m, red[0]);
        float alpha = __expf(m - m_new);   // 旧累计的修正因子

        // ④ 本块 P 值 + 行 sum
        float p = __expf(s - m_new);
        Ps[tid] = p;
        red[tid] = p; __syncthreads();
        for (int off = d / 2; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        l = l * alpha + red[0];            // 增量更新 running sum

        // ⑤ O 增量更新: O[tid] += Σ_jj Ps[jj] * Vs[jj][tid]
        float contrib = 0.f;
        for (int jj = 0; jj < d; ++jj) contrib += Ps[jj] * Vs[jj][tid];
        o = o * alpha + contrib;

        m = m_new;
        __syncthreads();   // 防止下一轮覆盖 Ps
    }

    O[(size_t)row * d + tid] = o / l;
}

// ============ 融合版 v2: 并行 Flash Attention（每 block 处理多行）============
// 相比 v1（每行一 block）的改进:
//   1. 每 block 处理 BR=8 行 → K/V 加载一次被 8 行复用（全局访存 ÷8）
//   2. block = 256 线程（8 个 warp）→ 线程利用率 4 倍于 v1
//   3. 每行一个 warp（x 方向 32 线程）→ 行归约仍用 __shfl_xor_sync
// 布局: block(32, 8) = 256 线程, grid = M/8, 每线程算 1 行 × 2 列
#define BR 8
__global__ void attn_fused_v2(const float* Q, const float* K, const float* V,
                              float* O, int M, int d, float scale) {
    const int tx = threadIdx.x;   // 0..31 (一个 warp = 一行)
    const int ty = threadIdx.y;   // 0..BR-1
    const int row = blockIdx.x * BR + ty;
    if (row >= M) return;

    __shared__ float Qs[BR][64];
    __shared__ float Ks[64][64];   // K 块 (d x d)
    __shared__ float Vs[64][64];   // V 块 (d x d)
    __shared__ float Ps[BR][64];   // P 值 (BR 行 × 当前块)

    // 加载 Q 块 (BR×d), 每线程 2 列
    Qs[ty][tx] = Q[(size_t)row * d + tx];
    Qs[ty][tx + 32] = Q[(size_t)row * d + tx + 32];
    __syncthreads();

    const int dcol = tx * 2;   // 本线程负责 O 的 2 列
    float m = -INFINITY, l = 0.f;
    float o0 = 0.f, o1 = 0.f;

    for (int jb = 0; jb < M; jb += d) {
        // ① 协同加载 K/V 块（每线程 16 个，合并访问）
        for (int idx = ty * 32 + tx; idx < d * d; idx += 256) {
            Ks[idx / d][idx % d] = K[(size_t)(jb + idx / d) * d + idx % d];
            Vs[idx / d][idx % d] = V[(size_t)(jb + idx / d) * d + idx % d];
        }
        __syncthreads();

        // ② 每线程算 S[ty][j] 和 S[ty][j+1] (j = tx*2)
        float s0 = 0.f, s1 = 0.f;
#pragma unroll
        for (int dd = 0; dd < d; ++dd) {
            const float q = Qs[ty][dd];
            s0 += q * Ks[tx * 2][dd];
            s1 += q * Ks[tx * 2 + 1][dd];
        }
        s0 *= scale; s1 *= scale;

        // ③ 在线 softmax（每行 = 一个 warp，shfl 归约）
        float local_max = fmaxf(s0, s1);
        for (int off = 16; off > 0; off >>= 1)
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));
        float m_new = fmaxf(m, local_max);
        float alpha = __expf(m - m_new);   // 旧累计修正因子

        float p0 = __expf(s0 - m_new), p1 = __expf(s1 - m_new);
        Ps[ty][tx * 2] = p0;
        Ps[ty][tx * 2 + 1] = p1;
        float local_sum = p0 + p1;
        for (int off = 16; off > 0; off >>= 1)
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, off);
        l = l * alpha + local_sum;

        // ④ O 增量更新: O[row][dcol..dcol+1] += Σ_j Ps[ty][j] * Vs[j][dcol..]
        float c0 = 0.f, c1 = 0.f;
#pragma unroll
        for (int j = 0; j < d; ++j) {
            const float pj = Ps[ty][j];
            c0 += pj * Vs[j][dcol];
            c1 += pj * Vs[j][dcol + 1];
        }
        o0 = o0 * alpha + c0;
        o1 = o1 * alpha + c1;
        m = m_new;
        __syncthreads();   // 防下一轮覆盖 Ps
    }

    O[(size_t)row * d + dcol] = o0 / l;
    O[(size_t)row * d + dcol + 1] = o1 / l;
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
int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int d = (argc > 2) ? atoi(argv[2]) : 64;
    float scale = 1.f / sqrtf((float)d);
    if (M % d != 0) { printf("M must be multiple of d\n"); return 1; }

    size_t bytesQ = (size_t)M * d * sizeof(float);
    size_t bytesS = (size_t)M * M * sizeof(float);

    float *h_Q = (float*)malloc(bytesQ);
    float *h_K = (float*)malloc(bytesQ);
    float *h_V = (float*)malloc(bytesQ);
    float *h_KT = (float*)malloc(bytesQ);   // K 的转置 (d×M)
    float *h_O1 = (float*)malloc(bytesQ);
    float *h_O2 = (float*)malloc(bytesQ);
    float *h_O3 = (float*)malloc(bytesQ);
    if (!h_Q || !h_K || !h_V || !h_KT || !h_O1 || !h_O2 || !h_O3) { printf("malloc failed\n"); return 1; }

    for (int i = 0; i < M * d; ++i) {
        h_Q[i] = ((float)(i % 7) - 3.f) * 0.05f;
        h_K[i] = ((float)(i % 11) - 5.f) * 0.05f;
        h_V[i] = ((float)(i % 5) - 2.f) * 0.05f;
    }
    for (int i = 0; i < M; ++i)               // KT[j][i] = K[i][j] (d×M)
        for (int j = 0; j < d; ++j)
            h_KT[j * M + i] = h_K[i * d + j];

    float *d_Q, *d_K, *d_V, *d_KT, *d_S, *d_O1, *d_O2, *d_O3;
    cudaMalloc(&d_Q, bytesQ);
    cudaMalloc(&d_K, bytesQ);
    cudaMalloc(&d_V, bytesQ);
    cudaMalloc(&d_KT, bytesQ);
    cudaMalloc(&d_S, bytesS);
    cudaMalloc(&d_O1, bytesQ);
    cudaMalloc(&d_O2, bytesQ);
    cudaMalloc(&d_O3, bytesQ);
    cudaMemcpy(d_Q, h_Q, bytesQ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytesQ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytesQ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_KT, h_KT, bytesQ, cudaMemcpyHostToDevice);

    // 朴素版: S=QK^T·scale -> softmax(S) -> O=S·V   (3 kernel, S 写回显存)
    dim3 blk(TILE, TILE);
    dim3 grdS((M + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    dim3 grdO((d + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    auto launch_naive = [&]() {
        sgemm_tiled<<<grdS, blk>>>(d_Q, d_KT, d_S, M, M, d, scale);
        softmax_kernel<<<M, 32>>>(d_S, M);
        sgemm_tiled<<<grdO, blk>>>(d_S, d_V, d_O1, M, d, M, 1.f);
    };

    // 融合版 v1: 每行一个 block（教学简化）
    auto launch_fused = [&]() {
        attn_fused<<<M, d>>>(d_Q, d_K, d_V, d_O2, M, d, scale);
    };
    // 融合版 v2: 每 block 处理多行（并行 Flash Attention）
    dim3 blockV2(32, BR);
    dim3 gridV2((M + BR - 1) / BR);
    auto launch_fused2 = [&]() {
        attn_fused_v2<<<gridV2, blockV2>>>(d_Q, d_K, d_V, d_O3, M, d, scale);
    };

    // 正确性: 以朴素版（写回 S 的标准算法）为参考
    launch_naive();
    launch_fused();
    launch_fused2();
    cudaDeviceSynchronize();
    cudaMemcpy(h_O1, d_O1, bytesQ, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_O2, d_O2, bytesQ, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_O3, d_O3, bytesQ, cudaMemcpyDeviceToHost);
    float cscale = 0.f;
    for (int i = 0; i < M * d; ++i) cscale = fmax(cscale, fabsf(h_O1[i]));
    double err1 = max_rel_err(h_O1, h_O2, M * d, cscale);
    double err2 = max_rel_err(h_O1, h_O3, M * d, cscale);

    // 计时
    int iters = (M >= 2048) ? 3 : 10;
    float t_naive = time_ms(launch_naive, iters);
    float t_fused = time_ms(launch_fused, iters);
    float t_fused2 = time_ms(launch_fused2, iters);

    printf("O = softmax(Q K^T / sqrt(%d)) V   M=%d d=%d\n", d, M, d);
    printf("--------------------------------------------------------------\n");
    printf("naive (3 kernel, 写回S) : %9.4f ms\n", t_naive);
    printf("fused v1 (在线)         : %9.4f ms\n", t_fused);
    printf("fused v2 (并行分块)     : %9.4f ms\n", t_fused2);
    printf("S 矩阵显存: 朴素 %6.1f MB | 融合 0 MB (不写回)\n",
           (double)bytesS / 1024 / 1024);
    printf("kernel 数 : 朴素 3 | 融合 1\n");
    printf("rel_err: v1=%g v2=%g (PASS if <1e-3)\n", err1, err2);
    printf("v2 vs v1 加速: %.2fx\n", t_fused / t_fused2);
    printf("--------------------------------------------------------------\n");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_KT);
    cudaFree(d_S); cudaFree(d_O1); cudaFree(d_O2); cudaFree(d_O3);
    free(h_Q); free(h_K); free(h_V); free(h_KT); free(h_O1); free(h_O2); free(h_O3);
    return 0;
}


