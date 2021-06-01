#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade("-t 1 -g 8 8 test")
command += testshade("-t 1 -g 8 8 test2")
command += testshade("-t 1 -g 8 8 test3")

outputs = [ 
    "out.txt"
]


