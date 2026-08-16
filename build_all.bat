@echo off
REM ============================================================
REM  build_all.bat — 一键编译所有 CUDA 程序 (Windows)
REM  需要: CUDA Toolkit 12.3+ / Visual Studio 2022+ (MSVC)
REM  编译前请确认 vcvars64.bat 与 nvcc 路径与本机一致
REM ============================================================
setlocal

call "D:\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo [ERROR] vcvars64.bat not found & exit /b 1)
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%"

echo === [1/4] Day 1: SAXPY ===
cd day1_saxpy
nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 saxpy.cu -o saxpy.exe
if errorlevel 1 (echo [FAIL] saxpy & exit /b 1)
cd ..

echo === [2/4] Day 2: SGEMM (needs cuBLAS) ===
cd day2_gemm
nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 gemm.cu -o gemm.exe -lcublas
if errorlevel 1 (echo [FAIL] gemm & exit /b 1)
cd ..

echo === [3/4] Day 3: Operator fusion ===
cd day3_fusion
nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 fusion.cu -o fusion.exe
if errorlevel 1 (echo [FAIL] fusion & exit /b 1)
nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 softmax_fusion.cu -o softmax_fusion.exe
if errorlevel 1 (echo [FAIL] softmax_fusion & exit /b 1)
cd ..

echo === [4/4] Day 5: Flash-attention style fusion ===
cd day5_attn
nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 attn_fused.cu -o attn_fused.exe
if errorlevel 1 (echo [FAIL] attn_fused & exit /b 1)
cd ..

echo.
echo ============ ALL BUILD OK ============
echo Run examples:  day1_saxpy\saxpy.exe / day2_gemm\gemm.exe 2048 2048 2048
echo                day3_fusion\fusion.exe / day5_attn\attn_fused.exe
endlocal
pause
