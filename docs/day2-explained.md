# Day 2 深入讲解：矩阵乘法每一步优化快在哪

> 数据来源：RTX 4070 Laptop（sm_89，FP32 算力 ~15 TFLOPS，带宽 256 GB/s），4096³ 实测。
> 配套代码：`day2_gemm/gemm.cu`（naive / tiled / vec4 / cuBLAS 四版本）

## 总览：四步阶梯

```
naive  1.1 TFLOPS → tiled  1.5 TFLOPS → vec4  3.4 TFLOPS → cuBLAS  13 TFLOPS
   │                     │                 │                  │
   │   访存冗余       减少访存         减少指令         寄存器/tensor core
   └──────────────── 每一步解决一个不同的瓶颈 ────────────────┘
```

## 第 1 步：naive 为什么慢？（1.1 TFLOPS）

### 机制
每个线程算 C 的一个元素，需要读 A 的一整行（K 个）+ B 的一整列（K 个）：
```c
for (k = 0; k < K; ++k)
    sum += A[row*K + k] * B[k*N + col];   // 每个 C 元素读 2K 个数
```

### 量化：访存冗余度
- C 有 M×N 个元素，每个读 2K 个数 → **总访存量 = 2·M·N·K 次读**
- 但 A 只有 MK 个、B 只有 KN 个元素
- 4096³ 冗余度 = 2·4096³/(4096²+4096²) ≈ **2048 倍**（每个 A 元素平均被读 2048 次）

### 数据验证
- 无缓存理论流量 = 2MNK×4B = **550 GB**，实际数据仅 192 MB
- 实测才 125ms（而非 550GB/256GB/s≈2.1s）→ 因为 **L2 缓存（32MB）兜底**了重复读
- naive 被 **L2 带宽 + 访存指令数** 卡住，只跑到 1.1 TFLOPS

> 关键：慢的第一性原理是**数据被反复从内存取**，不是计算慢。

## 第 2 步：tiling 快在哪？（1.5 TFLOPS，快 1.35x）

### 机制
把 A、B 切成 32×32 块，每个 block 把 tile **一次性搬进共享内存**（离计算单元近、快 ~100 倍），整块复用 32 次：
```c
As[threadIdx.y][threadIdx.x] = A[row*K + k0 + threadIdx.x];  // 只从全局读一次
Bs[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y)*N + col];
__syncthreads();
sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];              // 从 shared 复用
```

### 量化：访存量 ÷ TILE
- 全局访存从 2MNK → 2MNK/32（每元素只取 1 次，复用 32 次）
- 4096³：DRAM 理论流量 550GB → **17GB**
- 合并访问：加载 tile 时相邻线程读相邻地址 → 一次事务取整行

### 为什么只快 1.35x 而不是 32x？（最重要洞察）
1. naive 的 B 访问**本来就是合并的**，A 是 warp 广播（也高效）
2. naive 的重复读**大部分被 L2 命中**，不是每次都到 DRAM
3. naive 已跑出 1.1 TFLOPS，不算差实现

> 面试金句："tiling 把全局访存降到 2MNK/TILE，但实测只提升 1.3x——因为 naive 的冗余读大部分被 L2 缓存命中。优化要打真正的瓶颈，而不是想当然。"

## 第 3 步：float4 快在哪？（3.4 TFLOPS，比 tiled 快 2.3x）

### 机制
每线程从算 1 个元素变为算 1 行 × 4 列，用 `float4` 一次读写 16 字节：
```c
float4 b = *reinterpret_cast<const float4*>(B + br*N + bc);  // 1 条指令取 4 个 float
c.x += a*b.x; c.y += a*b.y; c.z += a*b.z; c.w += a*b.w;
*reinterpret_cast<float4*>(C + row*N + col) = c;             // 1 条指令写 4 个
```

### 量化：指令数 ÷ 4
- 之前读 4 个 float 发 4 条 `ld.global`；现在 1 条 `ld.global.v4`
- 访存指令总数降到 **1/4** → 指令发射带宽释放 → 算术单元跑得更满

### 为什么这步提升比 tiling 还大？
- tiled 解决"数据从哪来"（复用）；vec4 解决"取数据要发几条指令"（指令开销）
- 两个优化**正交**：一个省数据流量、一个省指令条数 → 叠加累计 3x

> float4 的本质 = **向量化**（CPU 的 SIMD/SSE/AVX 同一思想）。

## 第 4 步：cuBLAS 快在哪？（13 TFLOPS，比 vec4 再快 4x）

| 技巧 | 作用 |
|---|---|
| register tiling（每线程算 4×4~8×8 个 C 元素） | A/B 片段缓存在寄存器，复用 K 次，全局访存再降量级 |
| double buffering（加载下一 tile 同时计算） | 隐藏加载延迟，计算单元满负荷 |
| 循环展开 | 减少循环分支开销 |
| TF32/FP16 Tensor Core | 专用硬件吞吐远高于 FP32 CUDA core |
| 共享内存 padding | 避免 bank conflict |

## Roofline 视角：天花板在哪

- 算术强度 `AI = 2MNK/(MN+MK+KN) ≈ K/3`，4096³ ≈ **1365 flops/byte**
- 算力墙 15 TFLOPS / 带宽墙 256 GB/s → 拐点 ≈ **58 flops/byte**
- `AI=1365 >> 58` → GEMM 是**计算密集**，天花板 = 算力墙（15 TFLOPS）
- cuBLAS 13 TFLOPS = **87% 算力墙**（基本跑满）；naive 1.1 = **7% 算力墙**（离墙很远）

## 总结表

| 版本 | TFLOPS | 解决的瓶颈 | 核心机制 | 剩余瓶颈 |
|---|---|---|---|---|
| naive | 1.1 | —— | 每线程 1 元素，访存冗余 2048x | L2 带宽/指令数 |
| tiled | 1.5 | 数据流量 | 全局访存 ÷32，合并加载 | 指令数多 |
| vec4 | 3.4 | 指令数 | 访存指令 ÷4，FMA 并发 | 寄存器复用不足 |
| cuBLAS | 13.0 | 算力利用 | register tiling + 流水线 + tensor core | 接近理论墙 |

## 面试一句话版本

> naive 慢是因为同样的数据被重复取了几千次；tiling 让每个数据只取一次、复用好几次；float4 让取数据的指令少了 4 倍；cuBLAS 再让每个线程多算几个数、把算力顶到墙——每一步都在消除不同的瓶颈：先消除访存浪费，再消除指令浪费，最后把算力用满。
