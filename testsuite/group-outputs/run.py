#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command = testshade("--groupoutputs -od half -o a a.exr -o b b.exr -o c c.exr -g 64 64 test")
outputs = [ "a.exr", "b.exr", "c.exr", "out.txt" ]
