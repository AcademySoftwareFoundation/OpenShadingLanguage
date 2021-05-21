#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade("-t 1 -g 64 64 -od uint8 u_string_u_result -o cout uu_out.tif")
command += testshade("-t 1 -g 64 64 -od uint8 v_string_u_result -o cout vu_out.tif")
command += testshade("-t 1 -g 64 64 -od uint8 u_string_v_result -o cout uv_out.tif")
command += testshade("-t 1 -g 64 64 -od uint8 v_string_v_result -o cout vv_out.tif")

outputs = [ 
    "uu_out.tif",
    "vu_out.tif",
    "uv_out.tif",
    "vv_out.tif",
]


# expect a few LSB failures
failthresh = 0.008
failpercent = 3











