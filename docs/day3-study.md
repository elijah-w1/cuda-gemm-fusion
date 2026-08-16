# Day 3 知识点清单 + 学习路径（算子融合）

> 目标：吃透 fusion.cu，理解"算子融合为什么能加速"，为理解 AI 编译器（torch.compile/TVM/Triton）打基础。
> 每个知识点标注：fusion.cu 里的位置 + 学习目标。

## 一、知识点清单

### 1. 算子融合的概念（动机）★ 核心
- fusion.cu 位置：文件头注释（第 7-8 行）
- 学习目标：能说出融合省的两部分开销；能画出分离 vs 融合的数据流
- 资料：`docs/notes.md` Day 1 profiler 部分（launch overhead ~5-8µs 实测）
- 一句话：**launch 有固定开销 + 中间张量要写回显存 → 融合省掉 N-1 次 launch 和 N-1 次中间读写**

### 2. 分离式执行（3 个 kernel）
- fusion.cu 位置：`sgemm_tiled`（GEMM）、`bias_kernel`、`relu_kernel`
- 学习目标：认识"映射 kernel"（elementwise）——每个线程对 1 个元素做简单变换
- 关键：分离版 C 被写 3 次（GEMM 写一次、bias 读改写、relu 读改写），中间结果全部回显存

### 3. 融合实现：折叠（folding）★ 核心
- fusion.cu 位置：`sgemm_fused` 第 93 行（融合点）
- 学习目标：理解"把 elementwise 操作折叠进前一个 kernel 的最后一步"这个模式
  ```c
  C[row*N + col] = fmaxf(0.f, sum + bias[col]);  // 写 C 前顺便算完 bias+relu
  ```
- 关键：**bias/relu 不需要额外 kernel**，在 GEMM 算完 sum 的同一线程里顺手完成
- 为什么安全？因为 bias 只依赖 col、relu 只依赖本元素 → 无跨线程依赖，可安全内联

### 4. 数据依赖与顺序
- fusion.cu 位置：main 的 `launch_separate`（三个 kernel 顺序执行）
- 学习目标：理解为什么 bias 必须在 GEMM 之后（C 是 GEMM 的输出，也是 bias 的输入）
- CUDA 的隐式依赖：同一 stream 里 kernel 按顺序执行，无需手动同步

### 5. 性能测量与正确性（多 kernel 序列）
- fusion.cu 位置：`time_ms`（把 3 个 launch 放进一个计时循环）、`max_rel_err`（两版结果对比）
- 学习目标：怎么给"一组 kernel"计时；为什么融合版 vs 分离版用相对误差校验

### 6. 收益分析与 AI 编译器（拓展）
- fusion.cu 位置：实测数据（N=128 → 36.6%，N=2048 → 2.2%）
- 学习目标：能解释**为什么小算子融合收益大、大算子收益小**（launch 占比 vs 计算占比）
- 拓展：torch.compile / TVM / Triton 的 fusion pass 做的正是这件事（Day 4 验证）

## 二、学习路径（约 3~4 小时）

```
阶段 A（1h）：知识点 1+2 —— 概念 + 分离版
   → 复习 notes.md 的 Day 1 profiler 数据（launch ~5-8µs）
   → 读 fusion.cu 的 sgemm_tiled / bias_kernel / relu_kernel
   → 目标：能画分离版的数据流图（3 次 launch、C 被写 3 次）

阶段 B（1h）：★知识点 3+4 —— 融合点
   → 对比 sgemm_tiled 和 sgemm_fused 的差异（只有最后一行不同！）
   → 目标：解释"为什么 bias+relu 能安全折叠进 GEMM"（无跨线程依赖）

阶段 C（1h）：知识点 5 —— 实测
   → 编译运行 fusion.exe，复现 N=128（36.6%）和 N=2048（2.2%）
   → 把时间换成 launch 数：128 时 launch 占多少？2048 时呢？
   → 目标：亲手验证数据，能复述

阶段 D（0.5h）：★知识点 6 —— 收益分析 + 编译器视角
   → 回答：如果算子有 10 个 elementwise，融合收益是多大？（接近省 9/10 launch）
   → 想一想：AI 编译器（torch.compile）怎么自动发现这些融合机会？
```

## 三、自测题

1. 融合省掉的两部分开销是什么？
2. 为什么 bias/relu 能安全地折叠进 GEMM kernel？（不需要额外同步）
3. 为什么小矩阵（128）融合收益 36.6%，大矩阵（2048）只有 2.2%？
4. 如果有 10 个 elementwise 算子连在 GEMM 后面，融合 vs 分离差多少次 launch？
5. 融合 kernel 会不会破坏 GEMM 本身的优化？（不会——只是多算了 fmaxf，K 循环完全没变）

## 四、实测数据（RTX 4070 Laptop，手写 CUDA，rel_err=0）

| N | 分离 3 kernel | 融合 1 kernel | 节省 |
|---|---|---|---|
| 128 | 0.0156 ms | 0.0099 ms | **36.6%** |
| 2048 | 12.0106 ms | 11.7436 ms | 2.2% |

## 五、面试可用

- 融合省的两部分：① N-1 次 kernel launch ② N-1 次中间张量写回显存
- 融合 kernel 的实现模式：把 elementwise 折叠进前一个 kernel 的最后一步
- 融合的适用条件：连续算子且无跨线程数据依赖
- 与 Day 1 闭环：launch overhead 实测 → 融合动机 → 融合收益实测
