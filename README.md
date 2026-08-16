# CUDA GEMM & Kernel Fusion

![CUDA](https://img.shields.io/badge/CUDA-13.3-76B900?logo=nvidia)
![C++](https://img.shields.io/badge/C%2B%2B-CUDA%20Kernels-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch)
![GPU](https://img.shields.io/badge/GPU-RTX%204070%20Laptop-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

从零手写 CUDA 高性能计算核心（**矩阵乘法 GEMM** + **算子融合**），量化每一步优化的性能提升，并实现简化版 **Flash Attention**（在线 softmax 融合）——一条完整的高性能计算 / AI 编译器学习主线。

**环境**：RTX 4070 Laptop（8GB · Ada Lovelace · sm_89）· CUDA 13.3 · MSVC · Python 3.13 + PyTorch 2.11

---

## 核心成果

| 模块 | 关键数字 |
|---|---|
| **SAXPY** | GPU 加速 **7.5x**，有效带宽 **215 GB/s**（理论 256 的 84%） |
| **SGEMM** | naive **1.1** → tiled **1.5** → float4 **3.4** → cuBLAS **13 TFLOPS** |
| **算子融合** | GEMM+Bias+ReLU 融合省 **36.6%**；softmax 归约类融合省 **16-22%** |
| **简化 Flash Attention** | 在线 softmax 数值等价（rel_err~1e-5），M=4096 省 **64MB** 显存 |
| **编译器分析** | PyTorch eager 启动 **4 个 kernel** vs 手写融合 **1 个 kernel** |

---

## 快速开始

```powershell
# 一键编译所有 CUDA 程序（需要 CUDA Toolkit + MSVC）
.\build_all.bat

# 或手动编译单个模块
cd 02_gemm && nvcc -O3 -arch=sm_89 gemm.cu -o gemm.exe -lcublas
.\gemm.exe 2048 2048 2048          # [M] [N] [K]
```

---

## 项目结构

```
├── 01_saxpy/        SAXPY + CUDA 线程模型 + torch.profiler（launch 开销实测）
├── 02_gemm/         手写 SGEMM：naive → tiled → float4 → cuBLAS 基准
├── 03_fusion/       算子融合：elementwise（ReLU）+ 归约类（Softmax + warp shuffle）
├── 04_torch/        PyTorch eager 融合现状（profiler 数 kernel + 控制变量）
├── 05_attention/         简化版 Flash Attention（在线 softmax，S 不写回显存）
├── docs/              实验数据、专题讲解、学习计划
├── build_all.bat      一键编译
├── LICENSE            MIT
└── README.md
```

---

## 优化方法论（完整叙事线）

```
访存冗余分析 ─→ 共享内存 tiling ─→ float4 向量化 ─→ 算子融合 ─→ 在线 softmax
(naive 慢 2048x)  (访存 ÷32)        (指令 ÷4)      (省 launch+中间读写) (Flash Attn 省 S 矩阵)
        ↓                    ↓                  ↓                   ↓
  阶段 1 量化            阶段 2 实测             阶段 3 融合           阶段 5 收官
```

每一步都有**量化数据 + 正确性校验**（相对误差）+ **控制变量**（公平对比）。
最后用 PyTorch eager 的实测（4 kernel vs 手写 1 kernel）说明 **AI 编译器（torch.compile）为什么存在**。

---

## 学习文档（docs/）

| 文档 | 内容 |
|---|---|
| `notes.md` | 全部实测数据 + 结论 + 踩坑 |
| `stage2-explained.md` | GEMM 每步优化的深入讲解（访存量/指令数/roofline） |
| `warp-reduction.md` | warp 归约专题（__shfl_xor_sync 代码模板） |
| `stage*-plan.md` | 学习计划与目标 |

---

## License

[MIT](LICENSE) © 2026 elijjah

---

## 性能数据摘要

### 阶段 1 — SAXPY（N=16M 元素，64MB）
| 版本 | 耗时 | 有效带宽 |
|---|---|---|
| CPU | 7.088 ms | - |
| GPU kernel1 | 0.945 ms | 212.9 GB/s |
| GPU kernel2（grid-stride） | 0.937 ms | 214.8 GB/s |

### 阶段 2 — SGEMM（4096³）
| 版本 | GFLOPS | vs naive |
|---|---|---|
| naive | 1,099 | 1x |
| tiled（共享内存） | 1,483 | 1.35x |
| tiled + float4 | 3,334 | 3.03x |
| cuBLAS | 13,057 | 11.9x |

### 阶段 3 — 算子融合
| 融合类型 | 小矩阵 | 大矩阵 |
|---|---|---|
| GEMM+Bias+ReLU（elementwise） | 省 **36.6%**（N=128） | 省 2.2%（N=2048） |
| GEMM+Bias+Softmax（归约类） | 省 **22.3%**（M=1024） | 省 16.2%（M=4096） |

### 阶段 4 — PyTorch eager vs 手写融合
| 写法 | GPU kernel 数 |
|---|---|
| `relu(x@w+b)`（eager） | **4** |
| `relu(F.linear(x,w,b))`（eager） | **3** |
| 手写融合 kernel | **1** |

### 阶段 5 — 简化 Flash Attention（d=64，在线 softmax）
| M | 朴素(3 kernel, 写回S) | 融合(1 kernel, 在线) | S 显存(朴素) | rel_err |
|---|---|---|---|---|
| 1024 | 0.21 ms | 1.98 ms | 4 MB | 6.5e-6 |
| 4096 | 4.01 ms | 30.6 ms | **64 MB** | 2.7e-5 |


