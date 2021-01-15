@REM /* --------------------- */
@REM /* (C) 2020 madoodia.com */
@REM /*  All Rights Reserved  */
@REM /* --------------------- */

@REM Main batch file for installing OSL

@echo off


call osl_env_vars_setup.bat

cls

set OSL_INPUT_CONFIG=%1

if "%OSL_INPUT_CONFIG%" equ "debug" ( 
  set OSL_LOCATION_DIR=osl_debug
  set OSL_BUILD_CONFIG=--debug
  goto RUN
)
if "%OSL_INPUT_CONFIG%" equ "release" ( 
  set OSL_LOCATION_DIR=osl_release
  set OSL_BUILD_CONFIG=
  goto RUN
)
if "%OSL_INPUT_CONFIG%" equ "" ( 
  echo --= Enter debug or release as argument
  exit
)
@REM if argument is not debug or release
if "%OSL_INPUT_CONFIG%" neq "" (
  echo --= Enter debug or release as argument
  exit
)

:RUN

set PATH=%CMAKE_LOCATION%/bin;GIT_LOCATION/cmd;%PYTHON_LOCATION%;%PYTHON_LOCATION%/Scripts;%QT_LOCATION%/bin;%QT_LOCATION%/lib;%NASM_LOCATION%;%PATH%
set PYTHONPATH=%PYTHON_LOCATION%/Lib/site-packages;
set Qt5_ROOT=%QT_LOCATION%/lib/cmake

call %VCVARS_LOCATION%/vcvarsall.bat x64

cls

python build_osl.py --generator "Visual Studio 16 2019" --osl --python %OSL_BUILD_CONFIG% --zlib --boost --llvm --clang --pugixml --openexr --tiff --jpeg --png --flex --bison --opencolorio --openimageio --libraw --pybind11 %BASE_LOCATION%/%OSL_LOCATION_DIR%

