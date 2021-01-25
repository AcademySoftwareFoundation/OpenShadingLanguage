#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade("-g 64 64 -od uint8 -o Cout color.tif -o Cout_dx dx.tif -o Cout_dy dy.tif test")

outputs = [ "out.txt", "color.tif", "dx.tif", "dy.tif" ]
