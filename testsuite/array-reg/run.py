#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_index_float.tif test_varying_index_float")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_index_int.tif test_varying_index_int")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_index_string.tif test_varying_index_string")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_index_matrix.tif test_varying_index_matrix")
outputs.append ("out_varying_index_float.tif")
outputs.append ("out_varying_index_int.tif")
outputs.append ("out_varying_index_string.tif")
outputs.append ("out_varying_index_matrix.tif")


command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_index_color.tif test_varying_index_color")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_index_point.tif test_varying_index_point")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_index_vector.tif test_varying_index_vector")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_index_normal.tif test_varying_index_normal")
outputs.append ("out_varying_index_color.tif")
outputs.append ("out_varying_index_point.tif")
outputs.append ("out_varying_index_vector.tif")
outputs.append ("out_varying_index_normal.tif")


command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_out_of_bounds_index_int.tif test_varying_out_of_bounds_index_int")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_out_of_bounds_index_float.tif test_varying_out_of_bounds_index_float")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_out_of_bounds_index_string.tif test_varying_out_of_bounds_index_string")
outputs.append ("out_varying_out_of_bounds_index_int.tif")
outputs.append ("out_varying_out_of_bounds_index_float.tif")
outputs.append ("out_varying_out_of_bounds_index_string.tif")


command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_index_ray.tif test_varying_index_ray")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_index_cube.tif test_varying_index_cube")
outputs.append ("out_varying_index_ray.tif")
outputs.append ("out_varying_index_cube.tif")


command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_index_varying_float.tif test_varying_index_varying_float")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_index_varying_int.tif test_varying_index_varying_int")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_index_varying_point.tif test_varying_index_varying_point")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_index_varying_normal.tif test_varying_index_varying_normal")
outputs.append ("out_varying_index_varying_float.tif")
outputs.append ("out_varying_index_varying_int.tif")
outputs.append ("out_varying_index_varying_point.tif")
outputs.append ("out_varying_index_varying_normal.tif")


command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_index_varying_vector.tif test_varying_index_varying_vector")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_index_varying_color.tif test_varying_index_varying_color")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_index_varying_string.tif test_varying_index_varying_string")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_index_varying_matrix.tif test_varying_index_varying_matrix")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_varying_index_varying_ray.tif test_varying_index_varying_ray")
outputs.append ("out_varying_index_varying_vector.tif")
outputs.append ("out_varying_index_varying_color.tif")
outputs.append ("out_varying_index_varying_string.tif")
outputs.append ("out_varying_index_varying_matrix.tif")
outputs.append ("out_varying_index_varying_ray.tif")


command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_uniform_index_varying_float.tif test_uniform_index_varying_float")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_uniform_index_varying_int.tif test_uniform_index_varying_int")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_uniform_index_varying_point.tif test_uniform_index_varying_point")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_uniform_index_varying_normal.tif test_uniform_index_varying_normal")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_uniform_index_varying_vector.tif test_uniform_index_varying_vector")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_uniform_index_varying_color.tif test_uniform_index_varying_color")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_uniform_index_varying_string.tif test_uniform_index_varying_string")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_uniform_index_varying_matrix.tif test_uniform_index_varying_matrix")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_uniform_index_varying_ray.tif test_uniform_index_varying_ray")
outputs.append ("out_uniform_index_varying_float.tif")
outputs.append ("out_uniform_index_varying_int.tif")
outputs.append ("out_uniform_index_varying_point.tif")
outputs.append ("out_uniform_index_varying_normal.tif")
outputs.append ("out_uniform_index_varying_vector.tif")
outputs.append ("out_uniform_index_varying_color.tif")
outputs.append ("out_uniform_index_varying_string.tif")
outputs.append ("out_uniform_index_varying_matrix.tif")
outputs.append ("out_uniform_index_varying_ray.tif")


# expect a few LSB failures
failthresh = 0.008
failpercent = 3

