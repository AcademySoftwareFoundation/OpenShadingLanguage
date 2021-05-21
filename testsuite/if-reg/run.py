#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade("-t 1 -g 64 64 if_varying -od uint8 -o rgb out.tif")
command += testshade("-t 1 -g 64 64 if_varying_B -od uint8 -o rgb out_B.tif")

outputs = [ 
    "out.tif",
    "out_B.tif"
]

# expect a few LSB failures
failthresh = 0.008
failpercent = 3
