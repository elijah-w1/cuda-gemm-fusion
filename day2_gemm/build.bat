@echo off
REM build gemm.cu (links cuBLAS). ASCII only!
call "D:\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%"
nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 gemm.cu -o gemm.exe -lcublas
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo BUILD OK
gemm.exe 1024 1024 1024
