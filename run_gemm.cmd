@echo off
setlocal
cd /d "%~dp0"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo vswhere not found. Install Visual Studio or Build Tools, or call vcvars64.bat manually.
  exit /b 1
)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -property installationPath`) do (
  call "%%i\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)

nvcc -O3 -std=c++14 -o gemm.exe gemm.cu -lcublas && gemm.exe
exit /b %ERRORLEVEL%
