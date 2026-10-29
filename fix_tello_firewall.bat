@echo off
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Please run this file as Administrator!
    echo Right-click fix_tello_firewall.bat -> Run as administrator
    pause
    exit /b 1
)

echo Adding Windows Firewall rules for DJI Tello...
netsh advfirewall firewall add rule name="DJI Tello UDP Control" dir=in action=allow protocol=UDP localport=8889 profile=any
netsh advfirewall firewall add rule name="DJI Tello UDP State" dir=in action=allow protocol=UDP localport=8890 profile=any
netsh advfirewall firewall add rule name="DJI Tello UDP Video" dir=in action=allow protocol=UDP localport=11111 profile=any
netsh advfirewall set publicprofile state off

echo Successfully configured firewall for DJI Tello!
pause
