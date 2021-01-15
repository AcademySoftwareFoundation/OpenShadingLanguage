@REM /* --------------------- */
@REM /* 2020 madoodia.com */
@REM /* Copyright Contributors to the Open Shading Language project. */
@REM /* --------------------- */

@REM Running Visual Studio 2019 in debug mode for editing and debugging OSL

@echo off



call osl_env_vars_setup.bat

cls

set PATH=%PYTHON_LOCATION%;%PYTHON_LOCATION%/Scripts;%QT_LOCATION%/bin;%QT_LOCATION%/lib;%BASE_LOCATION%/osl_debug/bin;%BASE_LOCATION%/osl_debug/lib;%PATH%
set PYTHONPATH=%PYTHON_LOCATION%/Lib/site-packages;

%BASE_LOCATION%\osl_debug\build\OpenShadingLanguage\OSL.sln
