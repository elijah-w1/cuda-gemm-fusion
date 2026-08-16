@echo off
REM build fusion.cu (no external libs needed). ASCII only!
call "D:\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%"
nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 fusion.cu -o fusion.exe
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo BUILD OK
fusion.exe
