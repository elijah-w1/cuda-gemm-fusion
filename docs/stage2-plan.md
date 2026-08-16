# 阶段 2 具体计划：手写 SGEMM（naive → 共享内存 → 向量化）

> 硬件：RTX 4070 Laptop（sm_89，理论带宽 256 GB/s，FP32 ~15 TFLOPS）
> 预计用时：4~6 小时

---

## 一、目标

手写一个 SGEMM（单精度矩阵乘 C = A×B，行主序），
跑出 naive → 共享内存 tiling → float4 向量化 三个版本，
并用 cuBLAS 作为性能天花板对照，量化每一步优化带来的提升。

**为什么是 GEMM？** 它是神经网络全连接层/卷积的核心算子，也是
AI 编译器的基准测试（BLAS Benchmark 的 GEBP 内核），简历上最有代表性。

---

## 二、任务卡

### Task 1：naive 版（1 小时）
- 三重循环：`for i / for j / for k`，`C[i][j] += A[i][k]*B[k][j]`
- 每个线程算 C 中一个元素（`row = blockIdx.y*blockDim.y+threadIdx.y`）
- 先求正确（相对误差 < 1e-4），再测 GFLOPS
- 预期：极慢（远低于算力上限），因为完全没有利用缓存/合并访问

### Task 2：共享内存 tiling（1.5 小时）
- 把 A、B 切成 TILE×TILE 的块，先协同加载到 `__shared__`，再算
- 用 `__syncthreads()` 同步加载与计算阶段
- 关键：**合并访问**——相邻线程读相邻地址
- 预期：比 naive 快 5~20 倍

### Task 3：float4 向量化（1 小时）
- 每个线程用 `float4` 一次读写 4 个连续元素
- 减少指令数、提高访存效率
- 预期：在 tiling 基础上再提升 10~30%

### Task 4：cuBLAS 对照（30 分钟）
- `cublasSgemm` 一行调用，作为"最优"参考
- 对比手写版差距，理解还有多大优化空间
- 提示：cublasSgemm 还有寄存器重排、double buffering、循环展开等技巧

---

## 三、验收清单

- [ ] naive / tiled / vec4 三个版本正确性 PASS（与 CPU/naive 参考误差 < 1e-4）
- [ ] 有 GFLOPS 对比表（每个版本 × 1024/2048/4096 三种尺寸）
- [ ] 能解释：为什么 tiling 能提速？合并访问在哪里？
- [ ] 笔记记录每个版本与 cuBLAS 的差距

## 四、自测题（写在笔记里）

1. 为什么 naive 里 `B[k*N+j]` 的访问不合并？
2. TILE=32 时，一个 block 的 shared memory 需要多少字节？
3. `__syncthreads()` 不加会怎样？
4. 为什么 float4 能提速？它最多把访存指令数减少到几分之几？
