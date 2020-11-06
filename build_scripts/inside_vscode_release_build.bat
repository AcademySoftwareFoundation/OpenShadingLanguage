@echo off


cd ./build_scripts

call build_osl.bat release

call osl_release_launcher.bat
