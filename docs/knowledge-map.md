# AI 编译器 / 高性能计算 实习知识地图

> 用途：求职 HPC / AI 编译器方向的实习。标注 [x] 表示本项目已覆盖/已实践。

## 一、必掌握（面试核心）

### 1. GPU 编程基础
- [x] CUDA 线程模型：thread / block / grid、索引公式 `i=blockIdx.x*blockDim.x+threadIdx.x`
- [x] 内存层次：寄存器 / 共享内存 / 全局内存 的延迟与带宽差异
- [x] 合并访问（coalesced access）：相邻线程读相邻地址
- [x] 共享内存 tiling（GEMM 分块复用）
- [x] 向量化：float4 一次读写 4 元素
- [x] kernel launch 开销（~微秒级）

### 2. 性能分析
- [x] 访存密集型 vs 计算密集型的判定
- [x] 有效带宽 / GFLOPS 计算
- [x] CUDA events 计时、torch.profiler
- [ ] Roofline 模型（能画图解释"天花板"在哪）
- [ ] Nsight Compute（会读 occupancy / 带宽报告）

### 3. AI 编译器核心概念
- [x] 算子融合的动机（launch 开销 + 中间张量写回）
- [ ] Triton block 编程模型（tiling 抽象）
- [ ] 图优化：算子融合、死代码消除
- [ ] torch.compile 的工作方式

### 4. 深度学习算子基础
- [ ] 常见算子：GEMM、Conv、Attention、LayerNorm、Softmax
- [ ] 数值精度：FP32 / FP16 / BF16 / TF32、Tensor Core

### 5. 工程能力
- [x] C++：指针、内存、模板、编译链接
- [x] Python + PyTorch
- [ ] Git / GitHub、性能报告写作

## 二、加分项（简历深挖，能讲出细节）

- 用 Triton 手写 GEMM / softmax kernel
- 复现一个简化版 Flash Attention
- 用 Nsight 分析一次 kernel 优化前后的差异（有数据记录）
- 知道 TVM / MLIR / Triton 三者的定位差异（一句话能讲清）

## 三、可能被问到的面试题

1. 为什么共享内存 tiling 能让 GEMM 提速？快在哪？
2. 如何判断一个 kernel 是访存型还是计算型？（对比算术强度）
3. kernel launch 开销大约多少？怎么减少？
4. Tensor Core 是什么？FP16 比 FP32 快多少？
5. 算子融合为什么能加速？省掉了哪两部分开销？
6. 合并访问是什么？不合并会发生什么？
7. 手写 GEMM 到 cuBLAS 之间还差哪些优化技巧？

## 四、实习方向（可投递）

- 大厂 AI 基础设施：字节（ByteCC/内核优化）、阿里（PAI/编译）、腾讯（Angel）、华为（昇腾 CANN）、百度（Paddle/昆仑）
- 芯片公司：NVIDIA、AMD、寒武纪、壁仞、摩尔线程、沐曦
- AI 编译器团队：TVM / MLIR / Triton / XLA 相关岗位
- 独角兽/创业：深势科技、MiniMax、月之暗面（推理优化）

投递关键词：CUDA / GPU Kernel 优化 / AI Compiler / 推理引擎 / TensorRT / Triton
