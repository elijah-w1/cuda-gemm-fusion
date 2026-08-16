# Warp 归约详解（__shfl_xor_sync）

> GPU 编程高级技巧 · 面试高频手写题 · Flash Attention 的思想基础
> 关联代码：`阶段 3_fusion/softmax_fusion.cu`（归约类算子融合）

## 一、warp 是什么

- **warp = 32 线程**，GPU 执行的最小调度单位
- block 内每 32 个连续编号的线程组成一个 warp，**同时执行同一条指令**（SIMT 模式）
- 同一个 warp 内的线程天然同步（不需要 `__syncthreads()`）

## 二、为什么"每行 = 一个 warp"（关键布局设计）

线程在 block 内的线性 ID：`ID = threadIdx.y * blockDim.x + threadIdx.x`（**x 方向最快**）。

softmax_fusion.cu 用 `block(32, 32)` → `blockDim.x = 32 = warp 大小`：

```
ty=0:  ID 0..31     → warp 0   (row = blockIdx.y*32 + 0)
ty=1:  ID 32..63    → warp 1   (row = ... + 1)
...
ty=31: ID 992..1023 → warp 31  (row = ... + 31)
```

**关键洞察**：每行的 32 个线程恰好占满一个 warp →
**行归约 = warp 归约**，无需共享内存和跨线程同步。

> 如果 blockDim.x=128，一行会跨 4 个 warp，归约必须跨 warp（共享内存方案），复杂得多。
> 所以"让 blockDim.x = warp 大小"是一个刻意的简化设计。

## 三、__shfl_xor_sync 原理

- **shuffle 指令**：让 warp 内两个线程**直接交换寄存器值**（不经过内存，最快）
- 语法：`__shfl_xor_sync(mask, value, laneMask)`
  - `mask`：参与线程掩码（全参与用 `0xffffffff`）
  - `value`：要交换的值
  - `laneMask`：目标线程 = `本线程lane ^ laneMask`（异或翻转某些 bit）
- 归约本质：每步让信息量翻倍传播

```
求整行 max/sum（log2(32) = 5 步）：
step ^16: 32 线程两两合并 → 16 组
step ^8 : → 8 组
step ^4 : → 4 组
step ^2 : → 2 组
step ^1 : → 1 组（全员拿到整行结果）
```

## 四、代码模板（面试默写用）

```c
// ===== warp 内归约模板：背下来，面试手写 kernel 用 =====

// 1) 求最大值（5 步：16,8,4,2,1 = log2(32)）
for (int off = 16; off > 0; off >>= 1)
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, off));

// 2) 求和
for (int off = 16; off > 0; off >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, off);

// 3) softmax_fusion.cu 的完整行归约（先 max 再 sum 再归一）
float local_max = v[0];                                   // 每线程 8 列的局部 max
for (int i = 1; i < CPT; ++i) local_max = fmaxf(local_max, v[i]);
for (int off = 16; off > 0; off >>= 1)                    // warp 归约 → 行 max
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));

float local_sum = 0.f;                                    // exp 后求和
for (int i = 0; i < CPT; ++i) { v[i] = __expf(v[i] - local_max); local_sum += v[i]; }
for (int off = 16; off > 0; off >>= 1)                    // warp 归约 → 行 sum
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, off);
// 之后每个线程用 v[i] / local_sum 归一自己的 8 列
```

要点：
- 掩码 `0xffffffff` = 32 位全 1，表示所有 32 个 lane 都参与
- `lane ^ off` 配对：每步信息翻倍（32→16→8→4→2→1）
- 归约后**所有 32 个线程都持有同样的结果**（不需要"只留线程 0"）

## 五、为什么 warp shuffle 优于共享内存

| | 共享内存归约 | warp shuffle |
|---|---|---|
| 数据通路 | 写共享内存再读 | 寄存器直通 |
| 同步 | 需要 __syncthreads | warp 内天然同步 |
| 延迟 | 高 | 极低（1 条指令） |

## 六、关联 Flash Attention

- Flash Attention 用"分块 + **在线归约（running max/sum）**"避免写回 QK^T 中间矩阵
- 本项目的"每行 = 一个 warp + shuffle 归约"是其微型版本
- 理解顺序：warp 归约 → 分块 softmax → 在线更新 → Flash Attention

## 七、自测

1. 为什么 block(32,32) 里每行恰好是一个 warp？（ID = ty*32+tx，x 最快）
2. __shfl_xor_sync 是走内存还是寄存器？（寄存器直通）
3. 归约需要几步？为什么？（5 步，log2(32)）
4. blockDim.x=128 时行归约要怎么办？（跨 warp，共享内存）
5. 归约后哪个线程有结果？（所有线程都有）
