# 性能报告：CUDA GEMM 优化与算子融合

> 从零手写 CUDA 高性能计算（GEMM + 算子融合 + Flash Attention），
> 全部性能数据在本机实测，正确性经相对误差校验（rel_err < 1e-3）。

## 实验环境

| 项 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Laptop（8GB，Ada Lovelace，sm_89） |
| 理论带宽 | 256 GB/s（GDDR6） |
| FP32 理论算力 | ~15 TFLOPS |
| 工具链 | CUDA 13.3（nvcc）/ MSVC / Python 3.13 + PyTorch 2.11 |
| 计时方法 | CUDA events（GPU 侧时间戳）+ warmup（cuBLAS heuristics 稳定） |
| 正确性 | 相对误差校验（浮点累积误差场景，rel_err < 1e-3 即 PASS） |

---

## 阶段 1 — SAXPY：访存带宽基准（N=16M 元素，64MB）

| 版本 | 耗时 | 有效带宽 |
|---|---|---|
| CPU（单线程） | 7.088 ms | - |
| GPU kernel（每线程 1 元素） | 0.945 ms | 212.9 GB/s |
| GPU kernel（grid-stride） | 0.937 ms | 214.8 GB/s |

**结论**：GPU 加速 **7.5x**；有效带宽达理论值 **84%**（访存密集任务）。

### 线程配置扫描（kernel1，每线程 1 元素）
| threads | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| 带宽 GB/s | 214.7 | 215.8 | 216.6 | 218.3 |

→ 线程数变化 <2%：**访存密集任务带宽是唯一瓶颈**（为阶段 2 的访存分析铺垫）。

### kernel launch 开销（torch.profiler，1024 元素 add）
- 总耗时 ~9 µs，其中 GPU 纯计算仅 **1.16 µs** → launch 开销 **~7-8 µs/次**
- 这是算子融合的动机：小算子上 launch 占比巨大

---

## 阶段 2 — SGEMM：优化阶梯（4096³，行主序单精度）

| 版本 | GFLOPS | vs naive | 关键机制 |
|---|---|---|---|
| naive | 1,099 | 1x | 三重循环，访存冗余 2048 倍 |
| tiled（共享内存 32×32） | 1,483 | 1.35x | 全局访存 ÷32，合并访问 |
| tiled + float4 向量化 | 3,334 | 3.03x | 访存指令数 ÷4 |
| cuBLAS | 13,057 | 11.9x | register tiling / double buffering / tensor core |

**为什么 naive 慢**：每个 A 元素被 N 个线程重复读、每个 B 元素被 M 个线程重复读，
总访存量 = 2·M·N·K·4B（4096³ 时约 550 GB，而实际数据仅 192 MB）。

**为什么 tiling 只提升 1.3x**（而非理论 32x）：naive 的重复读大部分被 L2（32MB）命中，
真实瓶颈是 L2 带宽与访存指令数——优化必须对准实际瓶颈。

### 泛化验证
| 场景 | naive | tiled | vec4 | cuBLAS | 结论 |
|---|---|---|---|---|---|
| 1024×4096×2048（非方阵） | 1.14 | 1.42 | 3.51 | 12.9 TFLOPS | kernel 不依赖方阵 |
| 1000³（N 非 32 倍数） | 1.31 | 1.53 | 3.13 | 11.8 TFLOPS | 边界检查可靠 |

---

## 阶段 3 — 算子融合：分离 3 kernel vs 融合 1 kernel

### GEMM + Bias + ReLU（elementwise 融合）
| N | 分离(3 kernel) | 融合(1 kernel) | 节省 |
|---|---|---|---|
| 128 | 0.0156 ms | 0.0099 ms | **36.6%** |
| 2048 | 12.01 ms | 11.74 ms | 2.2% |

### GEMM + Bias + Softmax（归约类融合）
| M | 分离(3 kernel) | 融合(1 kernel) | 节省 |
|---|---|---|---|
| 1024 | 0.2277 ms | 0.1768 ms | **22.3%** |
| 4096 | 1.6103 ms | 1.3492 ms | 16.2% |

**结论**：
- 融合省两部分开销：① N-1 次 kernel launch（~7µs/次）② N-1 次中间张量写回显存
- **小算子收益大**（36.6%）：launch 占比高；大算子收益小但仍是正收益
- **两类融合的难度差异**：
  - elementwise（bias/relu）：无跨线程依赖 → 1 行折叠进 GEMM 最后一步
  - 归约类（softmax）：需整行 max/sum → 重新设计布局（block 覆盖整行）+ warp 归约（`__shfl_xor_sync`）

---

## 阶段 4 — PyTorch eager vs 手写融合（编译器视角）

| 写法 | GPU kernel 数 | 拆解 |
|---|---|---|
| `relu(x@w+b)`（eager） | **4** | GEMM + splitK-reduce + bias + relu |
| `relu(F.linear(...))`（eager） | **3** | GEMM(含 bias epilogue) + relu + Memset |
| 手写融合 kernel | **1** | GEMM+bias+relu 全包 |

**结论**：
- **cuBLAS epilogue** 能融合 bias（`F.linear` 利用），但 **relu 在 eager 下永远单独 kernel**（eager 不做跨算子图优化）
- 这就是 **torch.compile 存在的意义**：把"手写才能做的融合"自动化（图优化 pass）
- **控制变量教训**：eager（cuBLAS 13 TFLOPS）1.63ms vs 手写融合（tiled 1.5 TFLOPS）11.74ms——差异是 GEMM 实现而非融合效果；比融合收益必须同 GEMM 实现

---

## 阶段 5 — Flash Attention 融合（O = softmax(QKᵀ/√d)·V，d=64）

| M | naive(3 kernel) | v1(在线) | v2(并行分块) | S 显存(朴素) | rel_err |
|---|---|---|---|---|---|
| 1024 | 0.22 ms | 2.12 ms | **0.37 ms** | 4 MB | 6.5e-6 |
| 4096 | 4.02 ms | 29.97 ms | **4.73 ms** | **64 MB** | 2.7e-5 |

**结论**：
- **在线 softmax 数值等价**：rel_err ~1e-5 证明 running max/sum 与标准 softmax 数学等价（Flash Attention 正确性基础）
- **S 矩阵不写回显存**：M=4096 省 64MB，M=16384 省 1GB → 长序列 Transformer 的关键
- **v2 并行分块**（每 block 8 行复用 K/V + warp 归约）比 v1 快 **5.7~6.3x**，性能接近朴素实现（4.73 vs 4.02 ms）且保留全部融合收益

---

## 关键发现汇总

1. **访存 vs 计算**：SAXPY 卡带宽（84% 理论），GEMM 卡访存效率（naive 仅 7% 算力），优化必须对准瓶颈
2. **优化阶梯**：tiling（访存 ÷32）→ float4（指令 ÷4）→ register tiling（cuBLAS），累计 11.9x
3. **融合价值**：小算子 launch 占比高收益大（36.6%）；融合前提是算子间无跨线程依赖
4. **编译器动机**：eager 不融合 → torch.compile 自动化（4 kernel → 1 kernel）
5. **极致融合**：在线 softmax + 并行分块 = Flash Attention（省 M×M 中间矩阵 + 数值等价）

## 方法论

- **计时**：CUDA events（GPU 侧）+ warmup（cuBLAS heuristics）
- **正确性**：相对误差（浮点累积误差下 rel_err < 1e-3）
- **控制变量**：比融合收益同 GEMM 实现，比库优化同算子
- **内存安全**：Compute Sanitizer 验证 kernel 无越界访问
