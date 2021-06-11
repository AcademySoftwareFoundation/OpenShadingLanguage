#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# multiply float and matrix
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_float_mul_u_matrix.tif test_u_float_mul_u_matrix")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_float_mul_v_matrix.tif test_u_float_mul_v_matrix")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_float_mul_u_matrix.tif test_v_float_mul_u_matrix")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_float_mul_v_matrix.tif test_v_float_mul_v_matrix")
outputs.append ("out_u_float_mul_u_matrix.tif")
outputs.append ("out_u_float_mul_v_matrix.tif")
outputs.append ("out_v_float_mul_u_matrix.tif")
outputs.append ("out_v_float_mul_v_matrix.tif")

command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_float_mul_u_matrix_masked.tif test_u_float_mul_u_matrix_masked")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_float_mul_v_matrix_masked.tif test_u_float_mul_v_matrix_masked")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_float_mul_u_matrix_masked.tif test_v_float_mul_u_matrix_masked")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_float_mul_v_matrix_masked.tif test_v_float_mul_v_matrix_masked")
outputs.append ("out_u_float_mul_u_matrix_masked.tif")
outputs.append ("out_u_float_mul_v_matrix_masked.tif")
outputs.append ("out_v_float_mul_u_matrix_masked.tif")
outputs.append ("out_v_float_mul_v_matrix_masked.tif")

# multiply matrix and float 
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_matrix_mul_u_float.tif test_u_matrix_mul_u_float")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_matrix_mul_v_float.tif test_u_matrix_mul_v_float")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_matrix_mul_u_float.tif test_v_matrix_mul_u_float")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_matrix_mul_v_float.tif test_v_matrix_mul_v_float")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_matrix_mul_v_int.tif test_v_matrix_mul_v_int")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_matrix_mul_v_conditional.tif test_v_matrix_mul_v_conditional")
outputs.append ("out_u_matrix_mul_u_float.tif")
outputs.append ("out_u_matrix_mul_v_float.tif")
outputs.append ("out_v_matrix_mul_u_float.tif")
outputs.append ("out_v_matrix_mul_v_float.tif")
outputs.append ("out_v_matrix_mul_v_int.tif")
outputs.append ("out_v_matrix_mul_v_conditional.tif")

command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_matrix_mul_u_float_masked.tif test_u_matrix_mul_u_float_masked")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_matrix_mul_v_float_masked.tif test_u_matrix_mul_v_float_masked")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_matrix_mul_u_float_masked.tif test_v_matrix_mul_u_float_masked")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_matrix_mul_v_float_masked.tif test_v_matrix_mul_v_float_masked")
outputs.append ("out_u_matrix_mul_u_float_masked.tif")
outputs.append ("out_u_matrix_mul_v_float_masked.tif")
outputs.append ("out_v_matrix_mul_u_float_masked.tif")
outputs.append ("out_v_matrix_mul_v_float_masked.tif")


# multiply matrix and matrix 
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_matrix_mul_u_matrix.tif test_u_matrix_mul_u_matrix")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_matrix_mul_v_matrix.tif test_u_matrix_mul_v_matrix")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_matrix_mul_u_matrix.tif test_v_matrix_mul_u_matrix")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_matrix_mul_v_matrix.tif test_v_matrix_mul_v_matrix")
outputs.append ("out_u_matrix_mul_u_matrix.tif")
outputs.append ("out_u_matrix_mul_v_matrix.tif")
outputs.append ("out_v_matrix_mul_u_matrix.tif")
outputs.append ("out_v_matrix_mul_v_matrix.tif")

command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_matrix_mul_u_matrix_masked.tif test_u_matrix_mul_u_matrix_masked")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_matrix_mul_v_matrix_masked.tif test_u_matrix_mul_v_matrix_masked")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_matrix_mul_u_matrix_masked.tif test_v_matrix_mul_u_matrix_masked")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_matrix_mul_v_matrix_masked.tif test_v_matrix_mul_v_matrix_masked")
outputs.append ("out_u_matrix_mul_u_matrix_masked.tif")
outputs.append ("out_u_matrix_mul_v_matrix_masked.tif")
outputs.append ("out_v_matrix_mul_u_matrix_masked.tif")
outputs.append ("out_v_matrix_mul_v_matrix_masked.tif")


# divide float by matrix
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_float_div_u_matrix.tif test_u_float_div_u_matrix")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_float_div_v_matrix.tif test_u_float_div_v_matrix")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_float_div_u_matrix.tif test_v_float_div_u_matrix")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_float_div_v_matrix.tif test_v_float_div_v_matrix")
outputs.append ("out_u_float_div_u_matrix.tif")
outputs.append ("out_u_float_div_v_matrix.tif")
outputs.append ("out_v_float_div_u_matrix.tif")
outputs.append ("out_v_float_div_v_matrix.tif")

command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_float_div_u_matrix_masked.tif test_u_float_div_u_matrix_masked")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_float_div_v_matrix_masked.tif test_u_float_div_v_matrix_masked")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_float_div_u_matrix_masked.tif test_v_float_div_u_matrix_masked")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_float_div_v_matrix_masked.tif test_v_float_div_v_matrix_masked")
outputs.append ("out_u_float_div_u_matrix_masked.tif")
outputs.append ("out_u_float_div_v_matrix_masked.tif")
outputs.append ("out_v_float_div_u_matrix_masked.tif")
outputs.append ("out_v_float_div_v_matrix_masked.tif")

# divide matrix by float 
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_matrix_div_u_float.tif test_u_matrix_div_u_float")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_matrix_div_v_float.tif test_u_matrix_div_v_float")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_matrix_div_u_float.tif test_v_matrix_div_u_float")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_matrix_div_v_float.tif test_v_matrix_div_v_float")
outputs.append ("out_u_matrix_div_u_float.tif")
outputs.append ("out_u_matrix_div_v_float.tif")
outputs.append ("out_v_matrix_div_u_float.tif")
outputs.append ("out_v_matrix_div_v_float.tif")

command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_matrix_div_u_float_masked.tif test_u_matrix_div_u_float_masked")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_matrix_div_v_float_masked.tif test_u_matrix_div_v_float_masked")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_matrix_div_u_float_masked.tif test_v_matrix_div_u_float_masked")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_matrix_div_v_float_masked.tif test_v_matrix_div_v_float_masked")
outputs.append ("out_u_matrix_div_u_float_masked.tif")
outputs.append ("out_u_matrix_div_v_float_masked.tif")
outputs.append ("out_v_matrix_div_u_float_masked.tif")
outputs.append ("out_v_matrix_div_v_float_masked.tif")

# divide matrix by matrix 
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_matrix_div_u_matrix.tif test_u_matrix_div_u_matrix")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_matrix_div_v_matrix.tif test_u_matrix_div_v_matrix")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_matrix_div_u_matrix.tif test_v_matrix_div_u_matrix")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_matrix_div_v_matrix.tif test_v_matrix_div_v_matrix")
outputs.append ("out_u_matrix_div_u_matrix.tif")
outputs.append ("out_u_matrix_div_v_matrix.tif")
outputs.append ("out_v_matrix_div_u_matrix.tif")
outputs.append ("out_v_matrix_div_v_matrix.tif")

command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_matrix_div_u_matrix_masked.tif test_u_matrix_div_u_matrix_masked")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_u_matrix_div_v_matrix_masked.tif test_u_matrix_div_v_matrix_masked")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_matrix_div_u_matrix_masked.tif test_v_matrix_div_u_matrix_masked")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_v_matrix_div_v_matrix_masked.tif test_v_matrix_div_v_matrix_masked")
outputs.append ("out_u_matrix_div_u_matrix_masked.tif")
outputs.append ("out_u_matrix_div_v_matrix_masked.tif")
outputs.append ("out_v_matrix_div_u_matrix_masked.tif")
outputs.append ("out_v_matrix_div_v_matrix_masked.tif")


# negate matrix
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_neg_u_matrix.tif test_neg_u_matrix")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_neg_v_matrix.tif test_neg_v_matrix")
outputs.append ("out_neg_u_matrix.tif")
outputs.append ("out_neg_v_matrix.tif")


# expect a few LSB failures
failthresh = 0.008
failpercent = 3

