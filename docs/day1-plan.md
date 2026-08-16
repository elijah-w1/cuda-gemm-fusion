# Day 1 具体计划：环境搭建 + CUDA 线程模型 + SAXPY

> 你的硬件：RTX 4070 Laptop（8GB，Ada Lovelace，**sm_89**）
> 预计用时：6~8 小时（含安装等待）

---

## 一、总览（时间表）

| 时间段 | 任务 | 产出 |
|---|---|---|
| 0:00-0:30 | 硬件确认 + 环境自检 | 记录 GPU/驱动/CUDA 版本 |
| 0:30-2:00 | 安装 CUDA Toolkit | `nvcc --version` 有输出 |
| 2:00-2:30 | 安装 PyTorch (CUDA 版) | `check_env.py` 全 [OK] |
| 2:30-4:00 | 学 CUDA 线程模型 | 看懂 kernel 索引公式 |
| 4:00-6:00 | 编译运行 SAXPY + 计时 | 性能对比表 |
| 6:00-7:00 | torch.profiler 实验 | 理解 kernel launch overhead |
| 7:00-7:30 | 复盘 + 写笔记 | 明天 Day 2 的起点 |

---

## 二、任务卡（按顺序做）

### Task 1：环境自检（30 分钟）
```powershell
# 1) 看 GPU 与驱动支持的最高 CUDA 版本
nvidia-smi
#   记下顶部 "CUDA Version: xx.x"（这是驱动支持的，>=12 即可）

# 2) 运行我写好的自检脚本
cd D:\mingw64\elijjah\hh\day1_saxpy
python check_env.py
```

### Task 2：安装 CUDA Toolkit（1.5 小时，含下载）
**目的：获得 nvcc 编译器**（写 .cu 必须）。
- 方式 A（推荐，图形界面）：
  1. 打开 https://developer.nvidia.com/cuda-downloads
  2. 选 Windows → x86_64 → 最新版（12.x 或 13.x 均可）→ exe (local)
  3. 安装时选 **Custom**，只勾选：`CUDA > Compiler (nvcc)` 和 `CUDA > Runtime`，其余（Nsight、Visual Studio Integration、GeForce 等）可不装
  4. 安装后终端会自动有 nvcc（安装器会写 PATH）
- 方式 B（命令行，需管理员）：
  ```powershell
  winget install Nvidia.CUDA --accept-source-agreements --accept-package-agreements
  ```
- 验证：
  ```powershell
  nvcc --version     # 有输出即成功
  ```

> ⚠️ 若装的是 exe(local) 版，安装完请**重新打开终端**再验证 PATH。

### Task 3：安装 PyTorch（30 分钟）
> ⚠️ **Windows 实测坑**：`pip install torch` 默认装的是 **CPU 版**（`2.13.0+cpu`，`torch.cuda.is_available()=False`）！
> 必须从 PyTorch 官方 CUDA 源安装：
```powershell
# 先卸载 CPU 版（如果之前装过）
pip uninstall torch -y

# 从 CUDA 12.8 源安装（RTX 4070 支持；若要 cu126 把 cu128 换成 cu126）
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy          # torch 依赖 numpy，顺手装掉

# 验证（期望 True）
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### Task 4：学 CUDA 线程模型（1.5 小时）
**必读**：NVIDIA CUDA C++ 编程指南，只读 3 节：
- *2.2 Thread Hierarchy*（线程层次）
- *2.3 Memory Hierarchy*（内存层次）
- *3.2.1 Vector Addition*（向量加法示例）

**必须弄懂并默写**：
```
threadIdx.x  : 线程在自己 block 内的编号
blockIdx.x   : 线程所在 block 的编号
blockDim.x   : 一个 block 有多少线程
gridDim.x    : 一共有多少个 block
全局线程 id = blockIdx.x * blockDim.x + threadIdx.x
```
**自测题**（写在笔记里）：
1. blockDim=256，gridDim=8，一共多少个线程？
2. 为什么 `i = blockIdx.x*blockDim.x + threadIdx.x` 的访问模式是"合并访问"（coalesced）？
3. 一个 warp 有几个线程？（32）

### Task 5：编译运行 SAXPY（1.5 小时）
```powershell
cd D:\mingw64\elijjah\hh\day1_saxpy
.\build.bat
```
观察输出中的 4 个数字：CPU 耗时、GPU kernel1、GPU kernel2、有效带宽。
**任务**：
- 把 `saxpy.cu` 里的 `threads` 改成 128 / 512 / 1024，重跑，记录有效带宽变化
- 把 `blocks_gs` 改成 128 / 4096，重跑 grid-stride 版本
- 回答：为什么有效带宽接近 ~256 GB/s（4070 理论带宽）后就上不去了？
  → **提示**：这个 kernel 是纯访存型，算力（TFLOPS）远没用满。

### Task 6：torch.profiler 看 kernel launch 开销（1 小时）
```powershell
cd D:\mingw64\elijjah\hh\day1_saxpy
python profiler_demo.py
```
**核心观察**：
- 一个小张量（如 1024 元素）的加法和一个 100 万元素的加法，耗时差多少？
- 小张量时 **launch overhead（~5-10 微秒）** 占比巨大 → 这是 Day 4 算子融合要解决的问题。

### Task 7：复盘笔记（30 分钟）
在 `docs/notes.md` 里记录：
- 我的 GPU 型号 / 显存 / compute capability / 驱动 CUDA 版本
- 环境自检结果（截图或文本）
- SAXPY 性能表（threads、blocks 两维变化时的带宽）
- 三个自测题的答案
- 一句话总结：**"SAXPY 是访存密集型，瓶颈是带宽；明天 GEMM 既有访存又有计算，优化空间更大"**

---

## 三、验收清单（全部 ☑ 才算 Day 1 完成）

- [ ] `nvcc --version` 有输出
- [ ] `python check_env.py` 全部 [OK]
- [ ] `build.bat` 编译成功且输出正确性 Max error = 0
- [ ] 跑过 threads 与 blocks 的对比实验，笔记里有表
- [ ] 能向别人解释索引公式和 coalesced access
- [ ] 看过 profiler 输出，知道 launch overhead 量级
- [ ] `docs/notes.md` 已建立

## 四、常见坑

1. **nvcc 报 "No supported version of Visual Studio"** → CUDA 需要 MSVC。装一个 **Visual Studio Build Tools 2022**（勾选"使用 C++ 的桌面开发"），或者用 winget：`winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --add Microsoft.VisualStudio.Workload.VCTools"`。或者最省事：`nvcc -allow-unsupported-compiler`（不推荐，只在无 VS 时应急）。
2. **saxpy.exe 一闪而过** → 在 `build.bat` 最后加 `pause`，或直接 PowerShell 里跑 `.\saxpy.exe`。
3. **torch 装完 import 报错** → Python 3.13 需 torch>=2.6，`pip install torch` 默认会满足。
4. **winget 装 CUDA 失败** → 用 Task 2 的方式 A 手动下载安装。
