# Day 4 知识点清单 + 学习路径（PyTorch eager vs 手写融合）

> 目标：理解"PyTorch 默认执行到底启动几个 kernel"、"为什么需要 AI 编译器"。
> 背景：torch.compile 需 Triton（Windows+py3.13 装不上），改用 torch.profiler 数 kernel 实证。
> 代码：`day4_torch/day4_eager.py`、`day4_eager2.py`

## 一、知识点清单

### 1. PyTorch eager 执行模型
- 代码：day4_eager.py 的 `expr1`（`relu(x@w+b)`）
- 学习目标：理解 eager 模式**每个算子启动独立 kernel**，不做跨算子优化
- 实测：`relu(x@w+b)` = 4 个 kernel（GEMM + splitK-reduce + bias + relu）

### 2. cuBLAS epilogue ★ 核心
- 代码：day4_eager.py 的 `expr2`（`relu(F.linear(...))`）
- 学习目标：cuBLAS 允许 GEMM 输出时附加 bias/scale（epilogue）→ F.linear 融合 bias 只用了这一机制
- 实测：F.linear 版 = 3 个 kernel（GEMM含bias + relu + Memset）
- 类比：这就是"库层面的融合"，但 **relu 仍无法融入**（eager 无图优化）

### 3. kernel 数对比（融合现状）
| 写法 | kernel 数 |
|---|---|
| relu(x@w+b) | 4 |
| relu(F.linear) | 3 |
| 手写融合（Day 3） | 1 |

### 4. torch.compile 的意义 ★ 核心
- 学习目标：理解 torch.compile 通过**图优化 pass** 把 relu 折叠进 GEMM epilogue，自动化 Day 3 手写的工作
- 类比：Day 3 手动做的 == 编译器自动做的（Dynamo 捕获图 → Inductor 生成融合 kernel）

### 5. 控制变量方法论（Day 4 补充实验的教训）
- 代码：day4_eager2.py 实验 2
- 学习目标：**性能对比必须控制变量**（同 GEMM 实现比融合收益，同算子比库优化）
- 反面教材：eager(cuBLAS) 1.63ms vs 手写融合 11.74ms —— 差异是 GEMM 实现（13 vs 1.5 TFLOPS），不是融合效果

### 6. batch 对 kernel 数的影响
- 代码：day4_eager2.py 实验 1
- 实测：M=1 → 3 kernel（cuBLAS 自动切 gemv）；M=64/1024 → 4 kernel
- 结论：kernel 数不随 batch 变化，"eager 不融合"结论稳健

## 二、学习路径（约 2~3 小时）

```
阶段 A（0.5h）：回顾 Day 1 profiler 输出（怎么看 kernel 表）
   → 读 notes.md Day 1 部分，回忆 Self CUDA / cudaLaunchKernel

阶段 B（0.5h）：跑 day4_eager.py
   → python day4_eager.py
   → 观察 expr1 vs expr2 的 kernel 数差异（4 vs 3）

阶段 C（1h）：★理解 epilogue + 融合现状
   → 对比两个表达式的 kernel 列表
   → 回答：为什么 F.linear 少一个 kernel？（bias 被 cuBLAS epilogue 吃掉）

阶段 D（0.5h）：★控制变量 + torch.compile 原理
   → 读 day4_eager2.py 实验 2 的教训
   → 想清楚：eager 快是因为 cuBLAS 不是融合

阶段 E（0.5h）：自测 + 面试复盘
```

## 三、实测数据（RTX 4070 Laptop，torch 2.11，M=64/N=K=2048）

| 写法 | kernel 数 | kernel 列表 | 耗时 |
|---|---|---|---|
| relu(x@w+b) | 4 | sgemm(51.9µs)+splitK(4.0)+bias(2.5)+relu(1.8) | 66.2 µs |
| relu(F.linear) | 3 | sgemm含bias(62.2µs)+relu(1.7)+Memset | 76.3 µs |
| 手写融合（Day 3） | 1 | 全包 | —— |

batch 影响：M=1 → 3 kernel（gemv）；M=1024 → 4 kernel（sgemm，耗时 821µs）

## 四、自测题

1. eager 模式为什么不做跨算子融合？（无图优化，逐算子发射）
2. cuBLAS epilogue 是什么？F.linear 怎么利用它？（GEMM 输出附加 bias/scale）
3. 为什么 relu 在 eager 下永远是单独 kernel？（eager 没有"relu 可折叠"的依赖分析）
4. torch.compile 做了 Day 3 的什么事？（图优化 pass 自动融合）
5. 为什么"eager 1.63ms vs 手写 11.74ms"不能说明融合没用？（GEMM 实现不同，控制变量）
6. 怎么证明"eager 不融合"结论稳健？（测不同 batch，kernel 数不变 3-4）
