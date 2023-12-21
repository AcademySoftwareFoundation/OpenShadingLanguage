#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade("-t 1 -g 64 64 -od uint8 test -o cout rt_out.tif")
command += testshade("-t 1 -g 64 64 -od uint8 raytype_all -o cout rta_out.tif")
command += testshade("-t 1 -g 64 64 -od uint8 raytype_varying_result -o cout rtvr_out.tif")

# NOTE: if using this regression test for GPUs might need to adjust this 
command += testshade("-t 1 -g 64 64 -od uint8 raytype_varying_name -o cout rtvn_out.tif")

outputs = [ 
    "rt_out.tif",
    "rta_out.tif",
    "rtvr_out.tif",
    "rtvn_out.tif",
]

# expect a few LSB failures
failthresh = 0.008
failpercent = 3


