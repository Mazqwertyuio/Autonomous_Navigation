@echo off
chcp 65001 >nul
title DJI Tello Live 3D Autonomous Flight Launcher
cls
echo ======================================================================
echo    🛸 DJI TELLO REAL DRONE AUTONOMOUS FLIGHT CONTROLLER
echo ======================================================================
echo.
echo  [1] MonoNav Person Follower (🎯 Тільки стеження за людиною + 3D-захист від зіткнень)
echo  [2] MonoNav Metric Ultimate (🏆 Вільний політ 3D LiDAR + 180° Escape)
echo  [3] HKUST EGO-Planner v2    (Неперервна градієнтна оптимізація)
echo  [4] DiffusionPolicy4Drone   (Архів: дифузійний тест для аналізу)
echo  [5] Dry-run тест            (Вебкамера без підключення до дрона)
echo.
echo ======================================================================
set /p opt="Оберіть режим [1-5] (Enter = 1): "

if "%opt%"=="" set opt=1
if "%opt%"=="1" (
    .\venv\Scripts\python.exe tello_live_autopilot.py --mode follow
) else if "%opt%"=="2" (
    .\venv\Scripts\python.exe tello_live_autopilot.py --mode ultimate
) else if "%opt%"=="3" (
    .\venv\Scripts\python.exe tello_live_autopilot.py --mode ego
) else if "%opt%"=="4" (
    .\venv\Scripts\python.exe tello_live_autopilot.py --mode dp
) else if "%opt%"=="5" (
    .\venv\Scripts\python.exe tello_live_autopilot.py --mode dryrun
) else (
    .\venv\Scripts\python.exe tello_live_autopilot.py --mode follow
)

echo.
pause
