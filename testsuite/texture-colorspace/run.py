#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += oiiotool ("-pattern constant:color=0.5,0.5,0.5 64x64 3 -d half -otex grey.exr")
command += testshade("-g 1 1 --center -od half -o Cout out.exr test")
outputs = [ "out.txt" ]
