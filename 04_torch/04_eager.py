"""阶段 4_eager.py — 阶段 4: PyTorch eager 模式的算子融合现状
对比 PyTorch 默认执行 vs 阶段 3 手写 CUDA 融合 kernel (fusion.exe)

背景: torch.compile 需要 Triton, 而 Windows+py3.13 无法安装 Triton
     (官方无 wheel, 社区源失效)。所以聚焦:
     "PyTorch 默认(eager)执行 relu(x@W+b) 到底启动几个 kernel?"

运行: python 阶段 4_eager.py
"""
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity


def bench(fn, iters=30):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


torch.manual_seed(0)
N = 2048
M = 64
x = torch.randn(M, N, device="cuda")
w = torch.randn(N, N, device="cuda")
b = torch.randn(N, device="cuda")


# 表达式 1: 最朴素 y = relu(x@w + b)  —— 可能 3 kernel (mm, add, relu)
def expr1():
    return torch.relu(x @ w + b)


# 表达式 2: F.linear 本身融合了 bias —— 可能 2 kernel (gemm含bias, relu)
def expr2():
    return torch.relu(F.linear(x, w, b))


for name, fn in [("relu(x@w+b)", expr1), ("relu(linear(x,w,b))", expr2)]:
    print(f"\n=== {name} ===")
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            fn()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))
    t = bench(fn)
    print(f"avg time: {t*1000:.1f} us")
