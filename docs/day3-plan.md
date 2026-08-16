# Day 3 具体计划：算子融合（GEMM + Bias + ReLU）

> 背景：Triton 在 Windows + Python 3.13 无法安装（官方无 wheel，社区源被墙），
> 故 Day 3 改为**手写 CUDA 算子融合**——这正是 AI 编译器的核心优化，
> 不依赖任何库，效果同样有说服力。
> 预计用时：3~4 小时

## 一、目标

计算 `y = relu(A×B + bias)`（神经网络全连接层 + 激活的经典结构），
用两种方式实现并对比：

| 方式 | 实现 | GPU 操作次数 | 中间张量读写 |
|---|---|---|---|
| **分离**（不融合） | GEMM kernel → bias kernel → relu kernel | 3 次 launch | C 被写 3 次 |
| **融合**（融合） | 1 个 kernel：GEMM 算完直接加 bias + relu | 1 次 launch | C 只写 1 次 |

## 二、任务

1. **分离版**：复用 Day 2 的 tiled GEMM + 两个 elementwise kernel（bias、relu）
2. **融合版**：把 bias 和 relu 折叠进 GEMM kernel 的最后一步：
   `C[row][col] = fmaxf(0, sum + bias[col])`
3. **正确性**：两版结果一致（相对误差 < 1e-3）
4. **对比**：小矩阵（128³，launch 开销主导）和大矩阵（2048³，访存主导）两个尺寸

## 三、预期结论

- 小矩阵：融合版显著快（省 2 次 launch，~微秒级开销占大头）
- 大矩阵：融合版略快（省 2 次中间张量读写）
- 呼应 Day 1：kernel launch 有固定开销 + 中间张量写回显存 → 这就是融合动机

## 四、验收

- [ ] fusion.cu 编译运行通过，两版正确性一致
- [ ] 小矩阵对比表：分离 vs 融合
- [ ] 大矩阵对比表：分离 vs 融合
- [ ] 能解释：融合省了哪两部分开销

---

# Day 3.5 升级：归约类融合 softmax(A@B + bias)

> 背景：relu(elementwise) 融合只改 1 行，面试深度不够。
> 升级目标：softmax 需要整行归约（max/sum），展示"融合遇到数据依赖怎么办"。
> 代码：`day3_fusion/softmax_fusion.cu`
> 预计用时：3~4 小时

## 一、为什么 softmax 比 relu 难

| | relu 融合 | softmax 融合 |
|---|---|---|
| 依赖 | 无跨线程（每元素独立） | 需要整行 max/sum（跨线程归约） |
| 折叠方式 | 1 行 `fmaxf(0, sum+bias)` | 必须改布局 + warp 归约 |
| 难点 | 判断能不能折叠 | 布局设计 + 归约实现 |

## 二、关键设计

1. **block 覆盖完整行**：NCOLS=256，block(32,32) = 32 行 × 256 列
   - 每线程算 1 行 × 8 列（CPT=8）
   - 同一行 = 一个 warp（ID = ty*32+tx，x 最快）
2. **warp 归约**：`__shfl_xor_sync` 寄存器直通，5 步得整行 max/sum
3. **数值稳定 softmax**：先减行 max 再 exp（防溢出）
4. **共享内存预算**：Bs 32×256=32KB + As 32×32=4KB = 36KB < 48KB

## 三、任务

1. 融合 kernel `gemm_softmax_fused`：GEMM tiling 算完后就地做 softmax
2. 分离版 3 kernel：sgemm_tiled → bias → softmax
3. 正确性：rel_err = 0
4. 对比：不同 M/K 尺寸

## 四、实测（RTX 4070 Laptop，N=256 固定，rel_err=0）

| M | K | 分离 3 kernel | 融合 1 kernel | 节省 |
|---|---|---|---|---|
| 1024 | 512 | 0.2277 ms | 0.1768 ms | **22.3%** |
| 4096 | 1024 | 1.6103 ms | 1.3492 ms | **16.2%** |

## 五、验收

- [ ] softmax_fusion.cu 编译运行，rel_err=0
- [ ] 能解释：为什么 softmax 不能像 relu 那样 1 行折叠（行归约跨线程）
- [ ] 能默写 warp 归约模板（shfl 5 步：16,8,4,2,1）
- [ ] 能讲清与 Flash Attention 在线归约（running max/sum）的关系
