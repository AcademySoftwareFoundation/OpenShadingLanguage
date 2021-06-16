#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_compref_u_matrix_const_index.tif test_compref_u_matrix_const_index")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_compref_v_matrix_const_index.tif test_compref_v_matrix_const_index")
outputs.append ("out_compref_u_matrix_const_index.tif")
outputs.append ("out_compref_v_matrix_const_index.tif")

command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_compref_u_matrix_u_index.tif test_compref_u_matrix_u_index")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_compref_v_matrix_u_index.tif test_compref_v_matrix_u_index")
outputs.append ("out_compref_u_matrix_u_index.tif")
outputs.append ("out_compref_v_matrix_u_index.tif")

command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_compref_u_matrix_v_index.tif test_compref_u_matrix_v_index")
command += testshade("-t 1 -g 256 256 -od uint8 -o Cout out_compref_v_matrix_v_index.tif test_compref_v_matrix_v_index")
outputs.append ("out_compref_u_matrix_v_index.tif")
outputs.append ("out_compref_v_matrix_v_index.tif")

# expect a few LSB failures
failthresh = 0.008
failpercent = 3

