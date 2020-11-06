@echo off

cd ./build_scripts

call build_osl.bat debug

call osl_debug_launcher.bat
