# 笔记（Day 1 复盘用）

## 硬件信息
- GPU 型号：NVIDIA GeForce RTX 4070 Laptop
- 显存：8 GiB
- Compute Capability：8.9（Ada Lovelace，sm_89）
- 驱动 CUDA 版本：13.3（nvcc V13.3.73，CUDA Toolkit v13.3）

## 环境自检结果
（运行 `python check_env.py` 后粘贴输出）
- Python 3.13.12 / pip 26 / numpy 2.5.2
- Visual Studio 18.1 Community（MSVC 用于编译 .cu）
- 坑1：pip 默认装 CPU 版 torch → 必须 `--index-url https://download.pytorch.org/whl/cu128`
- 坑2：源码含中文注释必须 `-Xcompiler /utf-8`，否则 GBK 代码页解析错乱

## SAXPY 实测（本机跑通，N=16M 元素 64MB，a=2.0）
| 版本 | 耗时 | vs CPU | 有效带宽 |
|---|---|---|---|
| CPU | 7.088 ms | 1x | - |
| GPU kernel1（每线程1元素） | 0.945 ms | 7.5x | 212.9 GB/s |
| GPU kernel2（grid-stride） | 0.937 ms | 7.6x | 214.8 GB/s |

（4070 理论带宽约 256 GB/s → 已达 83%，剩余损失来自非完美合并访存与 launch 开销）
- 正确性：max_err = 0 PASS

## threads/blocks 扫描实验（Task 5，N=16M 元素 64MB）
| threads | blocks | kernel1 ms | 有效带宽 GB/s |
|---|---|---|---|
| 128 | 131072 | 0.938 | 214.7 |
| 256 | 65536 | 0.933 | 215.8 |
| 512 | 32768 | 0.930 | 216.6 |
| 1024 | 16384 | 0.922 | 218.3 |

| blocks_gs (threads=256) | kernel2 ms | 有效带宽 GB/s |
|---|---|---|
| 128 | 0.941 | 214.0 |
| 1024 | 0.932 | 216.0 |
| 4096 | 0.932 | 216.0 |

结论：
- threads 128→1024：带宽 214.7→218.3 GB/s，变化 <2% → 访存密集 kernel 对线程数不敏感
- blocks_gs 128→4096：几乎无差别（214.0 vs 216.0 GB/s）
- 所有配置稳定在 ~215±3 GB/s（理论 256 GB/s 的 ~84%）→ 瓶颈是显存带宽，不是线程调度
- saxpy.cu 已支持命令行参数：`saxpy.exe N [threads] [blocks_gs]`

## 自测题答案
1. blockDim=256, gridDim=8 -> 线程总数 = 256*8 = 2048
2. 连续线程的 i 相邻 → 一个 warp 读的 32 个 float 在内存中连续 → 一次事务即可取回 → 合并访问
3. warp 大小 = 32 线程

## profiler 观察（实测）
- 1024 元素 add 耗时：0.009 ms（9 微秒），其中 GPU kernel 纯计算仅 **1.16 微秒**
- launch overhead 量级：约 7~8 微秒/次（9us - 1.2us），host 端 cudaLaunchKernel 主导
- 大张量 64MB (1<<24) 乘法：kernel 888 微秒 → 有效带宽 ~210 GB/s（与手写 saxpy 一致，纯访存型）
- 一句话理解算子融合动机：kernel launch 有固定开销 + 中间张量要写回显存，把多个小算子合并成一个大 kernel 能省掉这两部分；小算子越多、融合收益越大

## Day 1 总结
SAXPY 是访存密集型任务：瓶颈是显存带宽（~215 GB/s ≈ 理论 256 GB/s 的 84%），threads/blocks 怎么调性能几乎不变；kernel launch 固定开销（~5-8µs）在小算子上占比巨大，这是算子融合的动机。Day 2 的 GEMM 既有访存又有计算，优化空间更大。

---

# Day 2 笔记：手写 SGEMM

## 实测数据（RTX 4070 Laptop，行主序单精度，正确性全部 PASS）
| 尺寸 | 版本 | ms | GFLOPS | vs naive |
|---|---|---|---|---|
| 1024³ | naive | 1.912 | 1122.9 | 1x |
| 1024³ | tiled | 1.472 | 1458.8 | 1.30 |
| 1024³ | tiled+vec4 | 0.685 | 3134.3 | 2.79 |
| 1024³ | cuBLAS | 0.220 | 9740.6 | 8.67 |
| 2048³ | naive | 14.83 | 1158.7 | 1x |
| 2048³ | tiled | 11.86 | 1448.8 | 1.25 |
| 2048³ | tiled+vec4 | 5.014 | 3426.3 | 2.96 |
| 2048³ | cuBLAS | 1.327 | 12943.4 | 11.17 |
| 4096³ | naive | 125.0 | 1099.5 | 1x |
| 4096³ | tiled | 92.65 | 1483.4 | 1.35 |
| 4096³ | tiled+vec4 | 41.22 | 3334.2 | 3.03 |
| 4096³ | cuBLAS | 10.53 | 13057.0 | 11.88 |

## 结论
- naive 稳定 ~1.1 TFLOPS：**访存冗余**——每个 A 元素被 N 个线程重复读、每个 B 元素被 M 个线程重复读，总访存量 = 2·M·N·K·4B（**注意：B[k*N+col] 是合并访问**，col 随 threadIdx.x 变；A[row*K+k] 是广播+冗余）
- tiling 提升 ~1.3x：A/B 各从全局只读一次进 shared，复用 TILE 次
- **为什么 tiling 只有 1.3x 而不是计划预期的 5-20x？** ① naive 的 B 访问本来就是合并的，A 是 warp 内广播（也高效）② 小矩阵时数据全在 L2（4070 有 32MB），naive 的冗余读大部分被 L2 命中 ③ naive 已是 1.1 TFLOPS，不算差实现。tiling 的真实收益在数据超出 L2、DRAM 流量成为瓶颈时最明显
- float4 提升到 ~3.4 TFLOPS（总 ~3x）：访存指令数减到 1/4
- cuBLAS ~9.7-13 TFLOPS（FP32 理论 15 TFLOPS 的 ~80%）：还有寄存器分块、double buffering、循环展开、tensor core 等
- 手写 vs cuBLAS 差 ~3.8x → Day 3 优化方向（每线程算多个 C 元素、寄存器复用）

## 泛化验证（额外补充）
| 场景 | naive | tiled | vec4 | cuBLAS | 结论 |
|---|---|---|---|---|---|
| 1024×4096×2048（非方阵） | 1.14 | 1.42 | 3.51 | 12.9 TFLOPS | kernel 不依赖方阵 |
| 1000³（N 不能被 32 整除） | 1.31 | 1.53 | 3.13 | 11.8 TFLOPS | 边界检查正确 |

## 踩坑
- **cuBLAS 首次调用有 heuristics 开销**：计时前必须 warmup，否则 1024³ 只有 348 GFLOPS（实际 9740）

## Day 2 自测题
1. naive 慢的根因？→ **不是 B 不合并**（B 列访问是合并的），而是访存冗余：A 每元素被 N 线程重复读、B 每元素被 M 线程重复读，总访存量 2·M·N·K·4B
2. TILE=32 → As+Bs 各 32×32×4B=4KB，共 8KB/block shared memory
3. 不加 `__syncthreads()` → 读共享内存时其他线程还没写完 → 数据竞争，结果错误
4. float4 把访存指令数减到 1/4（4 个元素/次）

---

# Day 3 笔记：算子融合（GEMM + Bias + ReLU）

> 计算 y = relu(A×B + bias)。Triton 在 Windows+py3.13 装不上，改为手写 CUDA 融合。
> 分离版 = 3 kernel（gemm/bias/relu），融合版 = 1 kernel。

## 实测（RTX 4070 Laptop，手写 CUDA tiled GEMM，rel_err=0）
| N | 分离 3 kernel | 融合 1 kernel | 节省 |
|---|---|---|---|
| 128 | 0.0156 ms | 0.0099 ms | **36.6%** |
| 2048 | 12.0106 ms | 11.7436 ms | 2.2% |

## 结论
- 小算子（小矩阵）：launch 开销占比大 → 融合收益显著（36.6%）
- 大算子（大矩阵）：GEMM 计算/访存主导 → 融合收益小（2.2%），但仍省 2 次中间张量读写
- 与 Day 1 profiler 发现完全一致：kernel launch 固定 ~5-8µs，**小算子融合价值最大**
- 这就是 AI 编译器（torch.compile / TVM / Triton）做算子融合的根本原因

## 面试可用
- 融合省的两部分：① N-1 次 kernel launch ② N-1 次中间张量写回显存
- 融合 kernel 的实现：把 elementwise 操作（bias/relu）折叠进前一个 kernel 的最后一步

## 一行融合的难点解析（fusion.cu 第 93 行）

```c
C[row * N + col] = fmaxf(0.f, sum + bias[col]);   // 融合点：只改了这一行
```

代码改动只有 1 行，但背后的**设计判断**不简单（面试区分度就在这）：

### 难点 1：判断"能不能折叠"（数据依赖分析）
- bias 只依赖列号 col、relu 只依赖本元素 → **无跨线程依赖** → 才能安全折叠
- 换成 softmax / LayerNorm（需要整行归约）→ 这 1 行不成立，必须改布局（见 Day 3.5）
- 这正是 AI 编译器做的依赖分析：torch.compile 判断哪些算子可融合

### 难点 2：严格数值等价
- 与分离版运算顺序完全一致（`sum` → `+bias` → `fmaxf`）→ rel_err 严格为 0
- 若改变顺序（如先 relu 后 bias），数值与语义都会变 → 融合必须保持原运算顺序

### 难点 3：访存语义（bias 的广播读）
- `bias[col]` 是"列广播"：同一列的 M 行线程读同一个 bias 值
- 全局内存对同一地址的并行读是广播（高效、无冲突）→ 不破坏合并访问

### 难点 4：收益的量化判断
- 省 C 写回 2 次（3 次→1 次）+ 省 2 次 launch
- 小矩阵收益大（36.6%）：launch 占比高
- 大矩阵收益小（2.2%）：计算主导，但访存带宽仍省（C 的 2 次写回是 M×N×4B×2）

### 难点 5：为什么"1 行"值得写进简历
- 代码改动小 ≠ 简单：难点在**依赖分析、数值等价、边界判断**
- 面试答好"为什么能这么改、什么情况不能" = 体现理解编译器优化原理

---

# Day 3.5 笔记：归约类融合 softmax(A@B + bias)

> 升级：relu(elementwise) 融合太简单，加 softmax(归约类) 展示真正的难点。
> 关键设计：**每个 block 覆盖完整行(N=256)** → 行归约在 block 内完成（warp shuffle）。
> 代码：`day3_fusion/softmax_fusion.cu`

## 实测（RTX 4070 Laptop，N=256 固定，rel_err=0）
| M | K | 分离 3 kernel | 融合 1 kernel | 节省 |
|---|---|---|---|---|
| 1024 | 512 | 0.2277 ms | 0.1768 ms | **22.3%** |
| 4096 | 1024 | 1.6103 ms | 1.3492 ms | **16.2%** |

## 核心技术点（面试可讲）
1. **归约类融合 vs elementwise**：bias/relu 无跨线程依赖可简单折叠；
   softmax 需要整行 max/sum → 必须重新设计布局（block 覆盖整行）
2. **布局改造**：block(32,32) = 32 行 × 256 列，每线程算 1 行 × 8 列
   → 同一行的 32 个线程恰好是一个 warp → 用 __shfl_xor_sync 做行内归约
3. **数值稳定 softmax**：先减行 max 再 exp（防止 exp 溢出）
4. **共享内存**：Bs 32×256=32KB + As 32×32=4KB = 36KB < 48KB 限制
5. **Flash Attention 的思想基础**：分块 + 在线归约（这里 block 覆盖整行，
   FA 更进一步用 running max/sum 支持任意行长度）
6. 收益 16-22%（比 relu 略低）：softmax 的归约本身是计算，融合主要省
   中间张量 M×256 的写读 + 2 次 launch

## 与 relu 融合的对比（完整故事）
| | relu 融合 | softmax 融合 |
|---|---|---|
| 实现难度 | 1 行（折叠） | 布局改造 + warp 归约 |
| 依赖分析 | 无跨线程 | 需要行归约 |
| 教学价值 | 融合的动机 | 融合的边界与设计 |

## warp 归约（独立文档：docs/warp-reduction.md）
> 面试高频手写题，完整讲解 + 代码模板已独立成文 `docs/warp-reduction.md`。
> 核心要点回顾：
> - block(32,32) 中每行恰好一个 warp（ID = ty*32 + tx，x 最快）→ 行归约 = warp 归约
> - `__shfl_xor_sync` 寄存器直通，5 步（16,8,4,2,1）完成 32 线程归约
> - 代码模板（求 max / 求和）可 10 秒默写
> - 关联 Flash Attention 的在线归约（running max/sum）思想

---

# Day 4 笔记：PyTorch eager 融合现状 vs 手写融合

> torch.compile 需 Triton（Windows+py3.13 装不上），改用 torch.profiler 数 kernel。
> 目标：`y = relu(x@W + b)`，看 PyTorch 默认启动几个 kernel。

## 实测（RTX 4070 Laptop，M=64, N=K=2048，torch 2.11）
| 写法 | kernel 数 | GPU kernel | 平均耗时 |
|---|---|---|---|
| `relu(x@w+b)` | **4** | sgemm(51.9µs) + splitK-reduce(4.0µs) + bias-add(2.6µs) + relu(1.7µs) | 66.2 µs |
| `relu(F.linear(x,w,b))` | **3** | sgemm含bias-epilogue(62.2µs) + relu(1.7µs) + Memset | 76.3 µs |
| 手写融合（Day 3） | **1** | GEMM+bias+relu 全包 | —— |

## 结论
- **cuBLAS epilogue 能融合 bias**（F.linear 利用这点）：GEMM 算完顺便加 bias，不单独开 kernel
- **relu 在 eager 下永远是单独 kernel**：PyTorch eager 不做跨算子融合
- 手写融合 = 1 kernel，比 PyTorch eager 更极致（省 relu 的 launch + 中间读写）
- **torch.compile 的意义**：把"手写才能做的融合"自动化（图优化 pass）
- 与 Day 1/3 完全闭环：launch 开销实测 → 融合动机 → 融合收益 → 编译器原理

## 面试可用
- "为什么需要 AI 编译器？"→ eager 模式 relu 无法融合进 GEMM，需要编译器做图优化
- "cuBLAS epilogue 是什么？"→ cuBLAS 允许在 GEMM 输出时附加 bias/scale 等操作
- "手写融合和 torch.compile 的关系？"→ 同一件事，手动 vs 自动

## Day 4 补充实验（batch 影响 + 绝对时间，代码 day4_eager2.py）

### 实验 1：不同 batch(M) 对 kernel 数与耗时（relu(x@w+b)，N=K=2048）
| M | GPU kernels | avg 耗时 | kernel 列表 |
|---|---|---|---|
| 1 | 3 | 33.3 µs | gemvx(28.1) + elementwise(1.2) + elementwise(1.2) |
| 64 | 4 | 67.3 µs | sgemm(51.9) + splitK(4.0) + bias(2.5) + relu(1.8) |
| 1024 | 4 | 821.5 µs | sgemm(761.6) + bias(21.4) + relu(12.6) + Memset |

结论：
- **kernel 数 3~4 个（随 batch 微变），手写融合始终 1 个** → "eager 不融合 relu"不随 batch 改变
- M=1 时 cuBLAS 自动用 gemv（矩阵×向量）而非 sgemm，无 splitK-reduce
- 耗时随 batch 线性增长（GEMM 主导），relu 单独 kernel 的占比随 batch 增大而缩小

### 实验 2：2048³ 绝对时间（⚠️ 注意控制变量）
| 实现 | 时间 |
|---|---|
| PyTorch eager（cuBLAS GEMM + relu） | 1.6303 ms |
| 手写融合 kernel（手写 tiled GEMM + bias + relu） | 11.7436 ms |

**重要解读（方法论教训）**：这个对比**不能**说明"eager 比融合好"！
- 两者 **GEMM 实现不同**：eager 用 cuBLAS（13 TFLOPS），手写用 tiled（1.5 TFLOPS）
- 手写融合的价值在"**相同 GEMM 下**省 launch + 中间读写"（Day 3 已证明：同实现省 2.2%）
- 这个数据展示的是 **cuBLAS vs 手写 GEMM 的库优化差距**（Day 2 已量化：9x）
- **性能对比必须控制变量**：比融合收益要同 GEMM 实现，比库优化要同算子

---

# Day 5 收官笔记：简化版 Flash Attention（attn_fused.cu）

## 计算
`O = softmax(Q·K^T / sqrt(d))·V`（M 行 × d 维），朴素 vs 在线融合

## 实测（RTX 4070 Laptop，d=64，rel_err 全部 PASS）
| M | 朴素(3 kernel, 写回S) | 融合(1 kernel, 在线) | S 显存(朴素) | rel_err |
|---|---|---|---|---|
| 1024 | 0.2145 ms | 1.9768 ms | 4 MB | 6.5e-6 |
| 4096 | 4.0055 ms | 30.5971 ms | **64 MB** | 2.7e-5 |

## 核心成果（工程意义）
1. **在线 softmax 数值等价验证**：rel_err ~1e-5 证明 running max/sum 与标准 softmax 数学等价 → Flash Attention 的正确性基础
2. **S 矩阵不写回显存**：M=4096 省 64MB，M=16384 省 1GB → 长序列 Transformer 的关键
3. **kernel 融合**：3 → 1
4. **诚实方法论**：教学简化版（每行一 block 串行扫块）时间不占优，真实 FA 需跨 block 并行分块 K/V——如实记录，不夸大

## 综合 4 天知识映射
| 代码 | 知识来源 |
|---|---|
| sgemm_tiled（朴素 GEMM） | Day 2：共享内存 tiling、合并访问 |
| softmax_kernel（warp 归约） | Day 3.5：__shfl_xor_sync、数值稳定 |
| attn_fused（在线 softmax） | Day 3：融合思想 + Flash Attention 精髓 |
| time_ms / warmup / 控制变量 | Day 1/4：计时方法 + 公平对比 |
| 相对误差校验 | Day 2/4：浮点累积误差处理 |

## 简历一句话
"实现简化版 Flash Attention：在线 softmax（running max/sum）融合进 GEMM，S 矩阵不写回显存，M=4096 时省 64MB 显存，数值等价验证 PASS（rel_err~1e-5）"
