@echo off
setlocal enabledelayedexpansion

set PORT=9317
set ROOT=%~dp0
set PYTHONDONTWRITEBYTECODE=1

if "%~1"=="" goto serve
if /i "%~1"=="web" goto serve
if /i "%~1"=="test" goto test
if /i "%~1"=="reconstruct" goto reconstruct
goto usage

:usage
echo Usage: run [web^|test^|reconstruct]
echo.
echo   web          start the browser app on port %PORT% (default)
echo   test         run the python and javascript suites
echo   reconstruct  run the node end-to-end benchmark
exit /b 1

:checknode
where node >nul 2>&1
if errorlevel 1 (
  echo Node.js is required. Install it from https://nodejs.org and run this again.
  exit /b 1
)
if not exist "%ROOT%web\node_modules" (
  echo Installing javascript dependencies...
  pushd "%ROOT%web"
  call npm install --silent
  set INSTALL_FAILED=!errorlevel!
  popd
  if not "!INSTALL_FAILED!"=="0" exit /b 1
)
exit /b 0

:checkpython
where python >nul 2>&1
if errorlevel 1 (
  echo Python 3.11 or newer is required. Install it from https://python.org and run this again.
  exit /b 1
)
python -c "import numpy, cv2" >nul 2>&1
if errorlevel 1 (
  echo Installing python dependencies...
  python -m pip install --quiet --disable-pip-version-check numpy opencv-python
  if errorlevel 1 exit /b 1
)
exit /b 0

:serve
call :checknode || exit /b 1
if not exist "%ROOT%data\templeSparseRing" (
  echo.
  echo No sample images found in data\templeSparseRing.
  echo Download templeSparseRing from vision.middlebury.edu/mview/data
  echo and unpack it there, or pick your own photos in the browser.
  echo.
)
echo Serving %ROOT%web on http://localhost:%PORT%
echo Pick the images, enter a focal length, press the button. Ctrl+C to stop.
echo.
pushd "%ROOT%web"
call npx --yes http-server . -p %PORT% -c-1
popd
exit /b 0

:test
call :checkpython || exit /b 1
call :checknode || exit /b 1
echo Running python tests...
python -m pytest tests -q -p no:cacheprovider
if errorlevel 1 exit /b 1
echo.
echo Running javascript tests...
pushd "%ROOT%web"
call npm test
set TEST_FAILED=!errorlevel!
popd
exit /b !TEST_FAILED!

:reconstruct
call :checknode || exit /b 1
if not exist "%ROOT%web\tests\temple.json" (
  echo web\tests\temple.json is missing. Build it with:
  echo   python scripts\export_dataset.py
  exit /b 1
)
pushd "%ROOT%web"
call node scripts/reconstruct.mjs
popd
exit /b 0