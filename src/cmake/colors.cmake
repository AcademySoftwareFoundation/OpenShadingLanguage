# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


# Define color codes for pretty terminal output
string (ASCII 27 ColorEsc)
set (ColorReset       "${ColorEsc}[m")
set (ColorRed         "${ColorEsc}[31m")
set (ColorGreen       "${ColorEsc}[32m")
set (ColorYellow      "${ColorEsc}[33m")
set (ColorBoldWhite   "${ColorEsc}[1;37m")
