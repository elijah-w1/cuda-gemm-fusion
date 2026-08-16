"""阶段 1 环境自检脚本: python check_env.py"""
import sys, subprocess, platform

print("=== 环境自检 ===")
print(f"Python        : {sys.version.split()[0]}")
print(f"OS            : {platform.platform()}")

try:
    import torch
    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA runtime  : {torch.version.cuda}")
    print(f"GPU available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"Device        : {name}")
        print(f"Compute Cap.  : {cc[0]}.{cc[1]}  (目标 arch: sm_{cc[0]}{cc[1]})")
        print(f"Memory        : {mem:.1f} GiB")
except ImportError:
    print("PyTorch       : 未安装 -> pip install torch")

try:
    r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    print("nvcc          : 可用")
    print(r.stdout.splitlines()[3] if r.stdout else "  (无输出)")
except FileNotFoundError:
    print("nvcc          : 未安装 / 不在 PATH -> 需要安装 CUDA Toolkit")

print("""
自检标准:
  [OK]  PyTorch 已装且 GPU available = True
  [OK]  nvcc 可用  (否则无法编译 .cu)
  [OK]  驱动中 CUDA 版本 >= 12.x
""")
