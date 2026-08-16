"""profiler_demo.py — 阶段 1 Task 6
观察: kernel launch overhead 在小张量时占比巨大, 这是 阶段 4 算子融合的动机。
运行: python profiler_demo.py
"""
import torch
from torch.profiler import profile, ProfilerActivity

def bench(tag, fn, iters=100):
    fn()  # warmup
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    print(f"{tag:<38} 单次平均 {start.elapsed_time(end)/iters:8.3f} ms")

def main():
    torch.manual_seed(0)
    print("=== 张量加法耗时对比 (RTX 4070 Laptop) ===")
    for n in [1024, 1 << 16, 1 << 20, 1 << 24]:
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        z = torch.empty_like(x)
        bench(f"n={n:<10} ({(n*4)/1024/1024:>7.2f} MB)  add", lambda: torch.add(x, y, out=z))

    print()
    print("=== profiler: 看小张量加法的 kernel 明细 ===")
    x = torch.randn(1024, device="cuda")
    y = torch.randn(1024, device="cuda")
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            torch.add(x, y)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))

    print()
    print("=== profiler: 大张量(1<<24)乘法对比: 看 kernel 名与耗时 ===")
    a = torch.randn(1 << 24, device="cuda")
    b = torch.randn(1 << 24, device="cuda")
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            c = a * b
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))

    print()
    print("""
解读要点(写进 notes.md):
1. 1024 元素加法耗时约 __ 微秒, 其中纯计算只需不到 1 微秒
   -> 大部分时间花在 kernel launch 上 (每次 ~5-10 微秒)
2. 这就是为什么 AI 编译器要做"算子融合":
   把多个小 kernel 合并成一个大 kernel, 减少 launch 次数与中间张量读写
""")

if __name__ == "__main__":
    main()
