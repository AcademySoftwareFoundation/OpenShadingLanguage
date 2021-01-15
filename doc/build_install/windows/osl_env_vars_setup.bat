@REM /* --------------------- */
@REM /* (C) 2020 madoodia.com */
@REM /*  All Rights Reserved  */
@REM /* --------------------- */

@REM Running Visual Studio 2019 in release mode for running OSL or editing code

@echo off

@REM TIP:
@REM for modifying env vars based on your paths
@REM if your path has space in it, put whole path between
@REM double quotation


set BASE_LOCATION=D:/madoodia/sdks
set PYTHON_LOCATION=C:/Python37
set QT_LOCATION=C:/Qt/5.15.0/msvc2019_64
set NASM_LOCATION=C:/NASM
set GIT_LOCATION="C:/Program Files/Git"
set CMAKE_LOCATION="C:/Program Files/CMake"
set VCVARS_LOCATION="C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build"

echo --= -------------------------------------------- =--
echo --= Environmment Variables created successfully! =--
echo BASE_LOCATION     :   %BASE_LOCATION%
echo PYTHON_LOCATION   :   %PYTHON_LOCATION%
echo QT_LOCATION       :   %QT_LOCATION%
echo NASM_LOCATION     :   %NASM_LOCATION%
echo GIT_LOCATION      :   %GIT_LOCATION%
echo CMAKE_LOCATION    :   %CMAKE_LOCATION%
echo VCVARS_LOCATION   :   %VCVARS_LOCATION%
echo --= -------------------------------------------- =--
