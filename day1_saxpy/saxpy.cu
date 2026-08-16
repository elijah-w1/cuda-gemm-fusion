// saxpy.cu — Day 1 入门 kernel
// y = a * x + y, N 个 float 元素
//
// 编译（Windows, 在装有 nvcc 的终端）:
//   nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 saxpy.cu -o saxpy.exe
// 运行:
//   ./saxpy.exe 16777216           (N = 16M 元素, 默认)
//   ./saxpy.exe 16777216 128       (kernel1 threads=128)
//   ./saxpy.exe 16777216 256 4096  (kernel1 threads=256, kernel2 blocks_gs=4096)
//
// 注意: 中文注释要求 -Xcompiler /utf-8, 否则 MSVC 在 GBK 代码页下会解析错乱。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

// ---------------------------------------------------------------
// Kernel 1: 每个线程处理 1 个元素（最简单的写法）
// 内置变量:
//   threadIdx.x : 线程在 block 内的编号
//   blockIdx.x  : block 在整个 grid 中的编号
//   blockDim.x  : 每个 block 的线程数
//   gridDim.x   : grid 中 block 的数量
// 全局索引公式:  i = blockIdx.x * blockDim.x + threadIdx.x
// ---------------------------------------------------------------
__global__ void saxpy_one_per_thread(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

// ---------------------------------------------------------------
// Kernel 2: grid-stride loop（推荐写法，可处理任意 n，且循环分块利于缓存）
// 每个线程处理多个元素，步长为 total_threads
// ---------------------------------------------------------------
__global__ void saxpy_grid_stride(float a, const float* x, float* y, int n) {
    int total = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += total) {
        y[i] = a * x[i] + y[i];
    }
}

// ---------------------------------------------------------------
// CPU 参考实现（用于校验正确性）
// ---------------------------------------------------------------
void saxpy_cpu(float a, const float* x, float* y, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

// ---------------------------------------------------------------
// CUDA 事件计时器：测量 GPU kernel 的真实耗时
// 模板函数，可接受任意可调用对象（含带捕获的 lambda）
// ---------------------------------------------------------------
template <typename F>
float time_kernel(F launch, int iters) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        launch();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

// ---------------------------------------------------------------
// 正确性校验: 比较 GPU 结果与 CPU 参考, 返回最大绝对误差
// ---------------------------------------------------------------
double max_error(const float* ref, const float* got, int n) {
    double err = 0.0;
    for (int i = 0; i < n; ++i) {
        err = fmax(err, fabs((double)ref[i] - (double)got[i]));
    }
    return err;
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 24);  // 默认 16M 元素 (64MB)
    float a = 2.0f;
    int bytes = n * sizeof(float);

    // ---- 分配显存与内存 ----
    float *d_x, *d_y1, *d_y2;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y1, bytes);
    cudaMalloc(&d_y2, bytes);

    float* h_x = (float*)malloc(bytes);
    float* h_y = (float*)malloc(bytes);
    float* h_y_cpu = (float*)malloc(bytes);
    float* h_y_gpu = (float*)malloc(bytes);

    for (int i = 0; i < n; ++i) {
        h_x[i] = (float)(i % 7) * 0.5f;
        h_y[i] = (float)(i % 3);
    }
    memcpy(h_y_cpu, h_y, bytes);

    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y1, h_y, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y2, h_y, bytes, cudaMemcpyHostToDevice);

    // ---- CPU 基线 ----
    auto t0 = std::chrono::high_resolution_clock::now();
    saxpy_cpu(a, h_x, h_y_cpu, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- GPU: kernel 1 (每线程一个元素) ----
    int threads = (argc > 2) ? atoi(argv[2]) : 256;  // 第 3 个参数可指定
    int blocks = (n + threads - 1) / threads;
    auto launch1 = [&]() { saxpy_one_per_thread<<<blocks, threads>>>(a, d_x, d_y1, n); };
    launch1();  // 第一次运行, y 初值 = h_y, 结果 = a*x + h_y, 用于正确性校验
    cudaMemcpy(h_y_gpu, d_y1, bytes, cudaMemcpyDeviceToHost);
    double err1 = max_error(h_y_cpu, h_y_gpu, n);
    // 注意: y = a*x + y 是累积运算, 之后每次 launch1 都会叠加到 d_y1 上,
    // 所以正确性校验必须在计时循环之前完成!
    float gpu1_ms = time_kernel(launch1, 50);

    // ---- GPU: kernel 2 (grid-stride, 固定 block 数) ----
    int blocks_gs = (argc > 3) ? atoi(argv[3]) : 1024;  // 第 4 个参数可指定
    auto launch2 = [&]() { saxpy_grid_stride<<<blocks_gs, threads>>>(a, d_x, d_y2, n); };
    launch2();
    cudaMemcpy(h_y_gpu, d_y2, bytes, cudaMemcpyDeviceToHost);
    double err2 = max_error(h_y_cpu, h_y_gpu, n);
    float gpu2_ms = time_kernel(launch2, 50);

    // ---- 带宽: 读 x (n) + 读 y (n) + 写 y (n) = 3*n*4 字节 ----
    double total_bytes = 3.0 * bytes;
    double bdw1 = total_bytes / (gpu1_ms / 1000.0) / 1e9;
    double bdw2 = total_bytes / (gpu2_ms / 1000.0) / 1e9;

    printf("N = %d (%d MB), a = %.1f\n", n, bytes / 1024 / 1024, a);
    printf("kernel1 grid: blocks=%d threads=%d | kernel2: blocks=%d threads=%d\n",
           blocks, threads, blocks_gs, threads);
    printf("----------------------------------------------------------\n");
    printf("CPU               : %8.3f ms\n", cpu_ms);
    printf("GPU kernel1       : %8.3f ms  (%6.1fx vs CPU)\n", gpu1_ms, cpu_ms / gpu1_ms);
    printf("GPU kernel2 grid  : %8.3f ms  (%6.1fx vs CPU)\n", gpu2_ms, cpu_ms / gpu2_ms);
    printf("kernel1 bandwidth : %8.1f GB/s (RTX4070 theory ~256 GB/s)\n", bdw1);
    printf("kernel2 bandwidth : %8.1f GB/s\n", bdw2);
    printf("max_err kernel1   : %g   (0 = PASS)\n", err1);
    printf("max_err kernel2   : %g   (0 = PASS)\n", err2);
    printf("----------------------------------------------------------\n");
    printf("Takeaway: SAXPY is memory-bound. Effective bandwidth near the\n");
    printf("theoretical limit means memory is the bottleneck.\n");
    printf("Record these numbers into docs/notes.md.\n");

    cudaFree(d_x); cudaFree(d_y1); cudaFree(d_y2);
    free(h_x); free(h_y); free(h_y_cpu); free(h_y_gpu);
    return 0;
}

