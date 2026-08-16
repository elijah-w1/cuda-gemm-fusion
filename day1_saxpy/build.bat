@echo off
REM Day 1 build script (Windows)
REM Steps: 1) load MSVC env (Visual Studio), 2) find nvcc, 3) compile & run
REM
REM Key points:
REM   -arch=sm_89      : RTX 4070 Laptop = Ada Lovelace
REM                     other GPUs: 3070/3080/3090 -> sm_86, 4090/4070 -> sm_89,
REM                     A100 -> sm_80, H100 -> sm_90, RTX 5090 -> sm_120
REM   -Xcompiler /utf-8 : required for Chinese comments in .cu source;
REM                      otherwise GBK codepage breaks parsing
REM
REM NOTE: keep this .bat ASCII-only! cmd.exe reads batch files in the
REM system codepage (GBK on Chinese Windows); UTF-8 Chinese text here
REM would be mangled and break the script.

REM ---- locate vcvars64.bat (Visual Studio) ----
set "VCVARS="
for %%V in (
  "D:\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
  "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
  "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
) do if exist %%V set "VCVARS=%%~V"
if not defined VCVARS (
  echo [ERROR] vcvars64.bat not found. Install Visual Studio Build Tools.
  exit /b 1
)
call "%VCVARS%" >nul

REM ---- ensure nvcc is available (CUDA Toolkit) ----
if not exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" (
  echo [ERROR] nvcc not found. Install CUDA Toolkit.
  exit /b 1
)
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%"

echo === Building saxpy.cu ===
nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 saxpy.cu -o saxpy.exe
if %errorlevel% neq 0 (echo BUILD FAILED & exit /b 1)
echo BUILD OK
echo.
echo === Running (N = 16M elements, 64 MB) ===
saxpy.exe 16777216
pause

