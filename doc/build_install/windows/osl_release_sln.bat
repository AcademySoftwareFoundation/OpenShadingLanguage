@REM /* --------------------- */
@REM Copyright Contributors to the Open Shading Language project.
@REM SPDX-License-Identifier: BSD-3-Clause
@REM /* --------------------- */

@REM Main batch file for installing OSL

@echo off


call osl_env_vars_setup.bat

cls

set PATH=%PYTHON_LOCATION%;%PYTHON_LOCATION%/Scripts;%QT_LOCATION%/bin;%QT_LOCATION%/lib;%BASE_LOCATION%/osl_release/bin;%BASE_LOCATION%/osl_release/lib;%PATH%
set PYTHONPATH=%PYTHON_LOCATION%/Lib/site-packages;

%BASE_LOCATION%\osl_release\build\OpenShadingLanguage\OSL.sln
