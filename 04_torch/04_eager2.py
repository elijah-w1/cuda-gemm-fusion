"""阶段 4_eager2.py — 阶段 4 补充实验
1) 不同 batch(M) 对 kernel 数和耗时的影响
2) 2048^3 方阵: PyTorch eager vs 手写融合(fusion.exe 阶段 3 数据)

运行: python 阶段 4_eager2.py
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
N = 2048  # 特征维度（N=K）


print("=== 实验 1: 不同 batch(M) 对 kernel 数与耗时的影响 (relu(x@w+b)) ===")
for M in [1, 64, 1024]:
    x = torch.randn(M, N, device="cuda")
    w = torch.randn(N, N, device="cuda")
    b = torch.randn(N, device="cuda")

    def fn():
        return torch.relu(x @ w + b)

    fn(); fn(); fn()  # warmup（让 cuBLAS heuristics 稳定）
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            fn()
    keya = prof.key_averages()
    gpu_kernels = [k for k in keya if k.self_device_time_total > 0]
    t = bench(fn)
    print(f"M={M:5d}: GPU kernels={len(gpu_kernels)}  avg={t*1000:7.1f} us")
    for k in gpu_kernels:
        print(f"    {k.key[:55]:<57} {k.self_device_time_total/k.count:7.1f} us x{k.count}")

print()
print("=== 实验 2: 2048^3 方阵绝对时间对比 ===")
M = N = K = 2048
x = torch.randn(M, N, device="cuda")
w = torch.randn(N, N, device="cuda")
b = torch.randn(N, device="cuda")
def big():
    return torch.relu(F.linear(x, w, b))
t_eager = bench(big, 5)
print(f"PyTorch eager  relu(linear(2048^3)) : {t_eager:.4f} ms")
print(f"手写融合 kernel (阶段 3 fusion.exe)  : 11.7436 ms  (N=2048 实测)")
print(f"eager / 手写融合 = {t_eager/11.7436:.2f}x")
