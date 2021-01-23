#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command = testshade("-g 256 64 -od uint8 -o Cout out.tif test")
outputs = [ "out.tif", "out.txt" ]
