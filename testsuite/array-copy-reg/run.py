#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_arraycopy_u_color.tif test_arraycopy_u_color")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_arraycopy_v_color.tif test_arraycopy_v_color")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_arraycopy_v_matrix.tif test_arraycopy_v_matrix")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_arraycopy_u_matrix.tif test_arraycopy_u_matrix")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_arraycopy_uv_matrix.tif test_arraycopy_uv_matrix")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_arraycopy_vu_matrix.tif test_arraycopy_vu_matrix")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_arraycopy_u_float.tif test_arraycopy_u_float")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_arraycopy_v_float.tif test_arraycopy_v_float")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_arraycopy_u_int.tif test_arraycopy_u_int")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_arraycopy_v_int.tif test_arraycopy_v_int")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_arraycopy_u_string.tif test_arraycopy_u_string")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_arraycopy_v_string.tif test_arraycopy_v_string")
outputs = [ 
    "out_arraycopy_u_color.tif",
    "out_arraycopy_v_color.tif",
    "out_arraycopy_v_matrix.tif",
    "out_arraycopy_u_matrix.tif",
    "out_arraycopy_uv_matrix.tif",
    "out_arraycopy_vu_matrix.tif",
    "out_arraycopy_u_float.tif",
    "out_arraycopy_v_float.tif",
    "out_arraycopy_u_int.tif",
    "out_arraycopy_v_int.tif",
    "out_arraycopy_u_string.tif",
    "out_arraycopy_v_string.tif"
]

# expect a few LSB failures
failthresh = 0.008
failpercent = 3






