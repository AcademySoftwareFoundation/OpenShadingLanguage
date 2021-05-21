#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Uncomment matrix tests after tranform is implemented
#command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_varying_index_matrix.tif test_varying_index_matrix")

command += testshade("--center -t 1 -g 64 64 -od uint8 -o Cout out_conditional_index_int.tif test_conditional_index_int")
command += testshade("--center -t 1 -g 64 64 -od uint8 -o Cout out_u_index_conditional_int.tif test_u_index_conditional_int")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_varying_index_float.tif test_varying_index_float")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_varying_index_int.tif test_varying_index_int")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_varying_index_string.tif test_varying_index_string")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_varying_index_color.tif test_varying_index_color")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_varying_index_point.tif test_varying_index_point")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_varying_index_vector.tif test_varying_index_vector")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_varying_index_normal.tif test_varying_index_normal")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_varying_index_ray.tif test_varying_index_ray")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_varying_index_cube.tif test_varying_index_cube")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_varying_out_of_bounds_index_int.tif test_varying_out_of_bounds_index_int")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_varying_out_of_bounds_index_float.tif test_varying_out_of_bounds_index_float")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_varying_out_of_bounds_index_string.tif test_varying_out_of_bounds_index_string")

outputs = [ 
#    "out_varying_index_matrix.tif"
    "out_conditional_index_int.tif",
    "out_u_index_conditional_int.tif",
    "out_varying_index_float.tif",
    "out_varying_index_int.tif",
    "out_varying_index_string.tif",
    "out_varying_index_color.tif",
    "out_varying_index_point.tif",
    "out_varying_index_vector.tif",
    "out_varying_index_normal.tif",
    "out_varying_out_of_bounds_index_int.tif",
    "out_varying_out_of_bounds_index_float.tif",
    "out_varying_out_of_bounds_index_string.tif",
    "out_varying_index_ray.tif",
    "out_varying_index_cube.tif"
]


# expect a few LSB failures
failthresh = 0.008
failpercent = 3









