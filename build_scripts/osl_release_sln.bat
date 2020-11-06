@echo off


call osl_env_vars_setup.bat

cls

set PATH=%PYTHON_LOCATION%;%PYTHON_LOCATION%/Scripts;%QT_LOCATION%/bin;%QT_LOCATION%/lib;%BASE_LOCATION%/osl_release/bin;%BASE_LOCATION%/osl_release/lib;%PATH%
set PYTHONPATH=%PYTHON_LOCATION%/Lib/site-packages;%BASE_LOCATION%/osl_release/lib/python3.7;

%BASE_LOCATION%\osl_release\build\OpenShadingLanguage\OSL.sln
