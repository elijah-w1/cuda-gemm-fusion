@echo off
REM ============================================================
REM  build_all.bat - one-click build for all CUDA programs
REM  Requires: CUDA Toolkit 12.3+ / Visual Studio 2022+ (MSVC)
REM  Adjust vcvars64.bat / nvcc paths to match your machine
REM ============================================================
setlocal

call "D:\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo [ERROR] vcvars64.bat not found & exit /b 1)
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%"

echo === [1/4] Stage 1: SAXPY ===
cd 01_saxpy
nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 saxpy.cu -o saxpy.exe
if errorlevel 1 (echo [FAIL] saxpy & exit /b 1)
cd ..

echo === [2/4] Stage 2: SGEMM (needs cuBLAS) ===
cd 02_gemm
nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 gemm.cu -o gemm.exe -lcublas
if errorlevel 1 (echo [FAIL] gemm & exit /b 1)
cd ..

echo === [3/4] Stage 3: Operator fusion ===
cd 03_fusion
nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 fusion.cu -o fusion.exe
if errorlevel 1 (echo [FAIL] fusion & exit /b 1)
nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 softmax_fusion.cu -o softmax_fusion.exe
if errorlevel 1 (echo [FAIL] softmax_fusion & exit /b 1)
cd ..

echo === [4/4] Stage 5: Flash-attention style fusion ===
cd 05_attention
nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 attn_fused.cu -o attn_fused.exe
if errorlevel 1 (echo [FAIL] attn_fused & exit /b 1)
cd ..

echo.
echo ============ ALL BUILD OK ============
echo Run examples:  01_saxpy\saxpy.exe  /  02_gemm\gemm.exe 2048 2048 2048
echo                03_fusion\fusion.exe  /  05_attention\attn_fused.exe
endlocal
pause

