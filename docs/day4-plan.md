# Day 4 具体计划：PyTorch eager 的融合现状 vs 手写融合

> 背景：torch.compile 需 Triton，Windows+py3.13 无法安装（官方无 wheel、
> 社区源失效），故 Day 4 改用 torch.profiler 数 kernel 的方式对比。
> 效果等价：证明"框架默认做多少融合 vs 手写能做多少融合"。
> 预计用时：2 小时

## 一、目标

用 torch.profiler 数 PyTorch 执行 `y = relu(x@W + b)` 实际启动的 kernel 数，
与你 Day 3 手写融合 kernel（1 个）对比，回答：
1. PyTorch eager 默认融合了多少？
2. 手写融合比它更极致在哪？
3. 这印证了 AI 编译器（torch.compile）存在的必要性

## 二、实测结果（RTX 4070 Laptop，M=64, N=K=2048）

| 写法 | kernel 数 | GPU kernel 列表 | 平均耗时 |
|---|---|---|---|
| `relu(x@w+b)` | **4** | GEMM, splitK-reduce, bias-add, relu | 66.2 µs |
| `relu(F.linear(x,w,b))` | **3** | GEMM(含bias epilogue), relu, Memset | 76.3 µs |
| 手写融合 kernel（Day 3） | **1** | GEMM+bias+relu 全包 | —— |

## 三、结论

1. **PyTorch eager 的融合程度**：cuBLAS 的 epilogue 能融合 bias（F.linear 就利用了这点），
   **但 relu 永远是单独 kernel**——eager 模式不做跨算子融合
2. **手写融合更极致**：1 个 kernel 全包，省掉 relu 那次 launch + 中间读写
3. **这就是 torch.compile / Triton 存在的意义**：把"手写才能做的融合"自动化
   （图优化 pass 发现 relu 可折叠进 GEMM epilogue）
4. 与 Day 1 闭环：launch 开销 ~5-8µs 实测 → 融合动机 → 融合收益实测 → 编译器原理

## 四、验收

- [ ] 能说出 eager 模式下 `relu(x@w+b)` 启动几个 kernel、为什么
- [ ] 能解释 cuBLAS epilogue 是什么
- [ ] 能解释为什么 torch.compile 能把 relu 也融掉