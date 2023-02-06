#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#NOTE: Smoothstep with color, vector, normal, and point 
#      are implemented in stdosl.h interms of smoothstep float.
#      Although this tests all types

def run_multi_type_test (opname, tagname, options) :
    global command
    command += testshade("-g 32 32 --param numStripes 16 "+options+" -od uint8 -o out_float out_float_"+tagname+".tif -o out_color out_color_"+tagname+".tif -o out_point out_point_"+tagname+".tif -o out_vector out_vector_"+tagname+".tif -o out_normal out_normal_"+tagname+".tif test_"+opname)
    global outputs     
    outputs.append ("out_float_"+tagname+".tif")
    outputs.append ("out_color_"+tagname+".tif")
    outputs.append ("out_point_"+tagname+".tif")
    outputs.append ("out_vector_"+tagname+".tif")
    outputs.append ("out_normal_"+tagname+".tif")
    return

def run_ternary_tests (opname) :
    run_multi_type_test (opname, opname, "")
    run_multi_type_test (opname, "Dx_"+opname, "--vary_pdxdy --param derivX 1")
    run_multi_type_test (opname, "Dy_"+opname, "--vary_pdxdy --param derivY 1")
    return

# Run ternary tests with mixing combinations of varying(v), uniform(u), and constant(c) parameters.
# Test with constants to pass non-derivative parameters to functions
# Choose not to test variation of 3 constant argument2, as that should be constant folded
# Choose not to test variation of constant and uniform arguments, assuming 
# testing of varying and uniform arguments provided enough coverage.
 
run_ternary_tests ("smoothstep")
run_ternary_tests ("smoothstep_u_u_u")
run_ternary_tests ("smoothstep_u_u_v")
run_ternary_tests ("smoothstep_u_v_u")
run_ternary_tests ("smoothstep_v_u_u")
run_ternary_tests ("smoothstep_u_v_v")
run_ternary_tests ("smoothstep_v_u_v")
run_ternary_tests ("smoothstep_v_v_u")

run_ternary_tests ("smoothstep_c_c_v")
run_ternary_tests ("smoothstep_c_v_c")
run_ternary_tests ("smoothstep_v_c_c")
run_ternary_tests ("smoothstep_c_v_v")
run_ternary_tests ("smoothstep_v_c_v")
run_ternary_tests ("smoothstep_v_v_c")

# expect a few LSB failures
failthresh = 0.008
failpercent = 3

