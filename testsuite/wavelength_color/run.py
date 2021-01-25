#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade("-g 1000 64 -od float -o Cout out.exr test")
outputs = [ "out.txt", "out.exr" ]
