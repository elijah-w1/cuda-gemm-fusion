# Day 2 知识点清单 + 学习路径

> 目标：吃透 gemm.cu 的每个细节，为 Day 3 Triton 打基础。
> 每个知识点标注了：gemm.cu 里的位置 + 本地资料路径 + 学习目标。

## 一、知识点清单（对应 gemm.cu 的流程）

### 1. 矩阵乘法与行主序存储
- gemm.cu 位置：`A[row*K + k]`、`C[row*N + col]`（naive kernel）
- 学习目标：能写出任意 (i,j) 的线性索引 `i*K + j`；能解释为什么行主序连续元素是同行相邻列
- 资料：数学书/任何 C 数组教程，手算一个 3×3 即可

### 2. GFLOPS 与运算量
- gemm.cu 位置：`double flops = 2.0 * M * N * K`（main）
- 学习目标：知道 GEMM 运算量 = 2MNK（乘+加各一次）；能算任意尺寸的 GFLOPS

### 3. 二维线程组织（blockIdx.y / threadIdx.y / dim3）
- gemm.cu 位置：`dim3 block(16,16)`、`blockIdx.y * blockDim.y + threadIdx.y`（三个 kernel）
- 学习目标：会用二维索引把 2D 数据映射到 grid/block；区分 x/y 维的布局
- 资料：本地 `src/02-thread-organization/hello5.cu`（二维 block）

### 4. 全局内存与合并访问 ★ 核心
- gemm.cu 位置：naive 里 `B[k*N+col]`（col 随 threadIdx.x → 合并）vs `A[row*K+k]`（row 随 threadIdx.y → 广播）
- 学习目标：能判断任意访存是否合并；理解 32 字节/128 字节事务
- 资料：本地 `src/07-global-memory/matrix.cu`（copy / transpose1/2/3 四版本对比）
  - 运行：`build.bat matrix.cu -run`（需要 error.cuh，先复制过去）

### 5. 共享内存 + 同步 + 协同加载 ★ 核心
- gemm.cu 位置：tiled 的 `__shared__ As/Bs`、两个 `__syncthreads()`、协同加载
- 学习目标：共享内存生命周期（block 内）、48KB/block 限制、两个同步点的作用、
  为什么"每线程 1 元素协同加载"能合并访问
- 资料：本地 `src/08-shared-memory/reduce2gpu.cu`（归约，共享内存入门）、`bank.cu`（bank conflict）
- 自测：TILE=32 → shared 用量 8KB；TILE=64 → 32KB，为什么会影响占用率？

### 6. 向量化 float4 与内存对齐
- gemm.cu 位置：vec4 的 `reinterpret_cast<const float4*>`、`make_float4`
- 学习目标：float4 一次 16 字节；地址必须 16 字节对齐；访存指令数减到 1/4
- 资料：NVIDIA 官方编程指南 / 搜索 "CUDA float4"

### 7. 计时与正确性校验
- gemm.cu 位置：time_ms（CUDA events + warmup）、max_rel_err（相对误差）
- 学习目标：CUDA events 为什么比 CPU 时钟准；warmup 为什么必须；
  为什么用相对误差而非绝对 0（浮点累积误差）
- 资料：本地 `src/05-prerequisites-for-speedup/add1cpu.cu`（CUDA events 计时）

### 8. cuBLAS 与列主序
- gemm.cu 位置：cublasSgemm 传参（m=N,n=M,k=K, A=d_B,B=d_A）
- 学习目标：行主序 C=A*B 等价于列主序 C'=B'*A'；leading dimension 概念
- 资料：NVIDIA cuBLAS 文档 / 本地 `src/14-libraries/`（cublas 示例）

## 二、学习路径（推荐顺序，约 6~8 小时）

```
阶段 1（1h）：知识点 1+2  手算矩阵乘 → 读 main() 的 flops 与数据生成
阶段 2（0.5h）：知识点 3   读 hello5.cu（二维 block）→ 回看 gemm 的 dim3
阶段 3（2h）：★知识点 4    读 matrix.cu 四个版本 → 编译跑 → 回看 naive 的访存
阶段 4（2h）：★知识点 5    读 reduce2gpu.cu → 编译跑 → 回看 tiled 的共享内存
阶段 5（1h）：知识点 6+7    看 vec4 的 float4 → 看 time_ms/校验
阶段 6（0.5h）：知识点 8    看懂 cuBLAS 调用 → 会讲列主序 trick
```

每阶段结束自测：不看代码，用一句话解释该知识点"为什么重要、gemm.cu 哪里用到了"。

## 三、示例实测数据（本机 RTX 4070 Laptop，N=1024）

### matrix.cu：矩阵转置 4 版本（合并访问对比）
| 版本 | 时间 | 说明 |
|---|---|---|
| copy | 0.0176 ms | 基准（读+写都合并） |
| transpose1 | 0.0737 ms | 合并读、**非合并写** |
| transpose2 | 0.0283 ms | 非合并读、**合并写** |
| transpose3 | 0.0284 ms | 合并写 + `__ldg()` 读 |

> **结论：非合并写比非合并读代价大得多（0.074 vs 0.028，差 2.6x）**。优化时优先保证写的合并。这解释了 naive GEMM 慢的另一面。

### bank.cu：bank conflict 的影响
| 版本 | 时间 |
|---|---|
| 有 bank conflict | 0.0296 ms |
| 无 bank conflict | 0.0165 ms |

> 消除冲突快 **1.8 倍**。shared memory 有 32 个 bank，同一 warp 同时访问同一 bank 不同地址 → 串行化。

### reduce2gpu.cu：归约 3 种写法（共享内存语法对比）
- 三个版本 `sum` 全部 = 123633392（**正确性一致**）
- reduce_global：只用全局内存（无 shared）
- reduce_shared：静态 `__shared__ real s_y[128]`
- reduce_dynamic：`extern __shared__`（启动配置第 3 参数传字节数）
- 核心算法：归约树 `for (offset = blockDim>>1; offset>0; offset>>=1)`，每轮一半线程合并另一半，`__syncthreads()` 保证同步
- 性能差异被 memcpy 主导（N=1e8），教学重点是**三种语法 + 同步**

## 四、快速启动（编译书里示例）

```powershell
# 复制 error.cuh 到示例目录
$src = 'D:\mingw64\elijjah\hh\cuda_study\CUDA-Programming-master\CUDA-Programming-master\src'
Copy-Item "$src\07-global-memory\error.cuh" "$src\07-global-memory\" -ErrorAction SilentlyContinue
# 用 day2 的 build.bat 模板编译（注意 -o 输出名）
cd $src\07-global-memory
nvcc -O3 -arch=sm_89 matrix.cu -o matrix.exe
.\matrix.exe 1024
```
