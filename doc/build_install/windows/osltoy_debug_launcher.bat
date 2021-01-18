@REM /* --------------------- */
@REM Copyright Contributors to the Open Shading Language project.
@REM SPDX-License-Identifier: BSD-3-Clause
@REM /* --------------------- */

@REM Running OSLToy in debug mode

@echo off


call osl_env_vars_setup.bat

cls

set PATH=%PYTHON_LOCATION%;%PYTHON_LOCATION%/Scripts;%QT_LOCATION%/bin;%QT_LOCATION%/lib;%BASE_LOCATION%/osl_debug/bin;%BASE_LOCATION%/osl_debug/lib;%PATH%
set PYTHONPATH=%PYTHON_LOCATION%/Lib/site-packages;

osltoy
