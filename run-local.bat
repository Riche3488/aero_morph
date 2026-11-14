@echo off
setlocal EnableExtensions
REM ==========================================================================
REM  run-local.bat  -  start the drag-morphing browser GUI (morph_gui.py) in WSL
REM
REM  Usage
REM    * double-click            -> choose a model from the menu
REM    * drag an .stl onto this  -> open that file directly
REM    * run-local.bat <path>    -> Windows path (C:\..., \\wsl.localhost\...) or WSL path (/home/...)
REM
REM  The browser opens automatically. Close this window (or press Ctrl+C) to stop the GUI.
REM  This file can be copied anywhere (e.g. the Desktop): it only uses absolute paths.
REM ==========================================================================

REM ---- settings ------------------------------------------------------------
set "DISTRO=Ubuntu-22.04"
set "PROJECT=/home/riche3488/aero_project"
set "PORT=8765"

title Drag Morph GUI
cd /d "%~dp0" 2>nul || cd /d "%USERPROFILE%"

where wsl.exe >nul 2>&1
if errorlevel 1 (
  echo [ERROR] wsl.exe not found. Install / enable WSL first.
  pause & exit /b 1
)

set "MODEL="
if not "%~1"=="" goto :from_arg

:menu
echo.
echo  ============================================================
echo    Drag Morph GUI  -  choose a model
echo  ============================================================
echo    1. IONIQ 6  rough (gen3d)           gen3d/runs/ioniq6/ioniq6_rough.stl
echo    2. IONIQ 5  rough (gen3d)           gen3d/runs/ioniq5/ioniq5_rough.stl
echo    3. IONIQ 5  parametric body         1_vehicle_geometry/output/ioniq5_body.stl
echo    4. IONIQ 5  CFD body (baseline)     3_cfd/baseline_case/constant/geometry/body.stl
echo    5. other file (type a path)
echo    Q. quit
echo.
set "CHOICE="
set /p "CHOICE=  Select [1-5, Q]: "
if /i "%CHOICE%"=="1" set "MODEL=%PROJECT%/gen3d/runs/ioniq6/ioniq6_rough.stl"
if /i "%CHOICE%"=="2" set "MODEL=%PROJECT%/gen3d/runs/ioniq5/ioniq5_rough.stl"
if /i "%CHOICE%"=="3" set "MODEL=%PROJECT%/1_vehicle_geometry/output/ioniq5_body.stl"
if /i "%CHOICE%"=="4" set "MODEL=%PROJECT%/3_cfd/baseline_case/constant/geometry/body.stl"
if /i "%CHOICE%"=="q" exit /b 0
if /i "%CHOICE%"=="5" (
  set /p "RAW=  STL path (Windows or WSL): "
  call :to_wsl_path "%%RAW%%"
)
if "%MODEL%"=="" (
  echo   invalid choice
  goto :menu
)
goto :run

:from_arg
call :to_wsl_path "%~1"
goto :run

REM ---- convert a Windows path to a WSL path (WSL paths pass through) ----------
:to_wsl_path
set "IN=%~1"
if "%IN:~0,1%"=="/" (
  set "MODEL=%IN%"
  exit /b 0
)
for /f "usebackq delims=" %%P in (`wsl.exe -d %DISTRO% --exec wslpath -a "%IN%"`) do set "MODEL=%%P"
exit /b 0

:run
wsl.exe -d %DISTRO% --exec test -f "%MODEL%"
if errorlevel 1 (
  echo.
  echo [ERROR] file not found in WSL: %MODEL%
  pause & exit /b 1
)
echo.
echo   model : %MODEL%
echo   modes : %PROJECT%/2_shape_morphing/modes/
echo   output: %PROJECT%/2_shape_morphing/gui_output/
echo.
echo   Starting server ... the browser opens automatically.
echo   Keep this window open while you work. Close it or press Ctrl+C to stop.
echo.
wsl.exe -d %DISTRO% --cd "%PROJECT%" --exec python3 2_shape_morphing/morph_gui.py "%MODEL%" --port %PORT%
echo.
echo   GUI stopped.
pause
endlocal
