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
cd day2_gemm && nvcc -O3 -arch=sm_89 gemm.cu -o gemm.exe -lcublas
.\gemm.exe 2048 2048 2048          # [M] [N] [K]
```

---

## 项目结构

```
├── day1_saxpy/        SAXPY + CUDA 线程模型 + torch.profiler（launch 开销实测）
├── day2_gemm/         手写 SGEMM：naive → tiled → float4 → cuBLAS 基准
├── day3_fusion/       算子融合：elementwise（ReLU）+ 归约类（Softmax + warp shuffle）
├── day4_torch/        PyTorch eager 融合现状（profiler 数 kernel + 控制变量）
├── day5_attn/         简化版 Flash Attention（在线 softmax，S 不写回显存）
├── docs/              学习计划、实测数据、专题讲解、实习知识地图
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
  Day 1 量化            Day 2 实测             Day 3 融合           Day 5 收官
```

每一步都有**量化数据 + 正确性校验**（相对误差）+ **控制变量**（公平对比）。
最后用 PyTorch eager 的实测（4 kernel vs 手写 1 kernel）说明 **AI 编译器（torch.compile）为什么存在**。

---

## 学习文档（docs/）

| 文档 | 内容 |
|---|---|
| `notes.md` | 全部实测数据 + 结论 + 踩坑（含 Day1-5） |
| `day2-explained.md` | GEMM 每步优化的深入讲解（访存量/指令数/roofline） |
| `warp-reduction.md` | warp 归约专题（__shfl_xor_sync 代码模板） |
| `knowledge-map.md` | AI 编译器 / HPC 实习知识地图 |
| `day*-plan.md` / `*-study.md` | 学习计划 + 路径 + 实测 |

---

## 面试亮点

- **完整优化方法论**：从访存冗余分析到每一步优化，全部有量化数据支撑
- **两种融合层次**：elementwise（1 行折叠）与归约类（布局改造 + warp 归约），能讲清融合的边界
- **编译器视角**：实测证明 PyTorch eager 启动 4 个 kernel vs 手写融合 1 个，理解 torch.compile 的意义
- **收官作品**：简化 Flash Attention（在线 softmax）——综合全部知识，验证数值等价 + 省显存
- **工程规范**：相对误差校验、warmup 计时、边界检查、泛化验证、控制变量方法论

---

## License

[MIT](LICENSE) © 2026 elijjah



```
├── day1_saxpy/        SAXPY + CUDA 线程模型 + torch.profiler
│   ├── saxpy.cu          两个 kernel（每线程1元素 / grid-stride）+ CPU 基线
│   ├── check_env.py      GPU/驱动/CUDA/PyTorch 环境自检
│   └── profiler_demo.py  kernel launch 开销实测（~7µs）
├── day2_gemm/         手写 SGEMM（行主序，单精度）
│   └── gemm.cu          naive → tiled(共享内存) → float4 向量化 → cuBLAS 基准
├── day3_fusion/       算子融合（手写 CUDA，不依赖 Triton）
│   ├── fusion.cu         GEMM+Bias+ReLU：elementwise 融合（分离3 kernel vs 融合1）
│   └── softmax_fusion.cu GEMM+Bias+Softmax：归约类融合（block 覆盖整行 + warp shuffle）
├── day4_torch/        PyTorch eager 融合现状
│   ├── day4_eager.py   torch.profiler 统计 relu(x@w+b) 实际启动的 kernel 数
│   └── day4_eager2.py  补充实验: batch 影响 + 绝对时间对比(控制变量)
├── day5_attn/         收官: 简化版 Flash Attention（综合全部知识）
│   └── attn_fused.cu   O=softmax(QK^T/√d)V 朴素3 kernel vs 在线融合1 kernel
└── docs/              学习计划、实测数据、专题讲解、实习知识地图
```

## 复现指南

```powershell
# 环境要求：CUDA Toolkit 12.3+ / MSVC（VS 2022+）/ Python 3.13
# Windows 下需先加载 MSVC 环境（或使用项目内 build.bat）

# Day 1: SAXPY（接受命令行参数 N [threads] [blocks_gs]）
cd day1_saxpy && .\build.bat            # 或直接 nvcc -O3 -arch=sm_89 saxpy.cu -o saxpy.exe
python profiler_demo.py                 # launch 开销实验

# Day 2: SGEMM 四版本对比（需要 cuBLAS：-lcublas）
cd day2_gemm && nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 gemm.cu -o gemm.exe -lcublas
.\gemm.exe 2048 2048 2048               # [M] [N] [K]

# Day 3: 算子融合（不需要外部库）
cd day3_fusion && .\build.bat           # relu 融合（N=128/2048 两档）
.\softmax_fusion.exe 4096 1024          # softmax 归约类融合 [M] [K]

# Day 4: PyTorch eager kernel 数统计
cd day4_torch && python day4_eager.py
```

---

## 性能数据摘要

### Day 1 — SAXPY（N=16M 元素，64MB）
| 版本 | 耗时 | 有效带宽 |
|---|---|---|
| CPU | 7.088 ms | - |
| GPU kernel1 | 0.945 ms | 212.9 GB/s |
| GPU kernel2（grid-stride） | 0.937 ms | 214.8 GB/s |

### Day 2 — SGEMM（4096³）
| 版本 | GFLOPS | vs naive |
|---|---|---|
| naive | 1,099 | 1x |
| tiled（共享内存） | 1,483 | 1.35x |
| tiled + float4 | 3,334 | 3.03x |
| cuBLAS | 13,057 | 11.9x |

### Day 3 — 算子融合
| 融合类型 | 小矩阵 | 大矩阵 |
|---|---|---|
| GEMM+Bias+ReLU（elementwise） | 省 **36.6%**（N=128） | 省 2.2%（N=2048） |
| GEMM+Bias+Softmax（归约类） | 省 **22.3%**（M=1024） | 省 16.2%（M=4096） |

### Day 4 — PyTorch eager vs 手写融合
| 写法 | GPU kernel 数 |
|---|---|
| `relu(x@w+b)`（eager） | **4** |
| `relu(F.linear(x,w,b))`（eager） | **3** |
| 手写融合 kernel | **1** |

---

## 学习文档（docs/）

| 文档 | 内容 |
|---|---|
| `notes.md` | 全部实测数据 + 结论 + 踩坑 |
| `day2-explained.md` | GEMM 每步优化的深入讲解（访存量/指令数/roofline） |
| `warp-reduction.md` | warp 归约专题（__shfl_xor_sync 代码模板） |
| `day3-plan.md` / `day2-study.md` | 计划 + 学习路径 + 实测 |
| `knowledge-map.md` | AI 编译器/HPC 实习知识地图 |

---

## 面试亮点

- **完整优化方法论**：从访存冗余分析到每一步优化，全部有量化数据支撑
- **两种融合层次**：elementwise（1 行折叠）与归约类（布局改造 + warp 归约），能讲清融合的边界
- **编译器视角**：用实测证明 PyTorch eager 启动 4 个 kernel vs 手写融合 1 个，理解 torch.compile 存在的意义
- **工程规范**：正确性校验（相对误差）、warmup 计时、边界检查、泛化验证（非方阵/不可整除）


