#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

def run_multi_type_test (opname, tagname, options) :
    global command
    command += testshade("--center -t 1 -g 32 32 --param numStripes 16 "+options+" -od uint8 -o out_float out_float_"+tagname+".tif -o out_color out_color_"+tagname+".tif -o out_point out_point_"+tagname+".tif -o out_vector out_vector_"+tagname+".tif -o out_normal out_normal_"+tagname+".tif test_"+opname)
    global outputs     
    outputs.append ("out_float_"+tagname+".tif")
    outputs.append ("out_color_"+tagname+".tif")
    outputs.append ("out_point_"+tagname+".tif")
    outputs.append ("out_vector_"+tagname+".tif")
    outputs.append ("out_normal_"+tagname+".tif")
    return

def run_unary_tests (opname) :
    run_multi_type_test (opname, opname, "")
    run_multi_type_test (opname, "Dx_"+opname, "--vary_pdxdy --param derivX 1")
    run_multi_type_test (opname, "Dy_"+opname, "--vary_pdxdy --param derivY 1")
    return

# Run unary tests with varying and a 2nd time with uniform(u) argument to the op
# Choose not to test variation of a constant argument, as that should be constant folded
run_unary_tests ("neg")
run_unary_tests ("neg_u")

run_unary_tests ("sqrt")
run_unary_tests ("sqrt_u")

run_unary_tests ("inversesqrt")
run_unary_tests ("inversesqrt_u")

run_unary_tests ("abs")
run_unary_tests ("abs_u")

run_unary_tests ("floor")
run_unary_tests ("floor_u")

run_unary_tests ("ceil")
run_unary_tests ("ceil_u")

run_unary_tests ("trunc")
run_unary_tests ("trunc_u")

run_unary_tests ("round")
run_unary_tests ("round_u")

def run_int_test (opname, tagname, options) :
    global command
    command += testshade("--center -t 1 -g 32 32 --param numStripes 16 "+options+" -od uint8 -o out_int out_int_"+tagname+".tif test_"+opname)
    global outputs     
    outputs.append ("out_int_"+tagname+".tif")
    return

def run_unary_int_tests (opname) :
    run_int_test (opname, opname, "")

run_unary_int_tests ("neg_int")
run_unary_int_tests ("neg_u_int")

run_unary_int_tests ("abs_int")
run_unary_int_tests ("abs_u_int")


def run_binary_tests (opname) :
    run_multi_type_test (opname, opname, "")
    run_multi_type_test (opname, "Dx_"+opname, "--vary_pdxdy --param derivX 1")
    run_multi_type_test (opname, "Dy_"+opname, "--vary_pdxdy --param derivY 1")
    return

# Run binary tests with mixing combinations of varying(v), uniform(u), and constant(c) parameters.
# Test with constants to pass non-deriviative parameters to functions
# Choose not to test variation of 2 constant argument, as that should be constant folded 
run_binary_tests ("fmod")
run_binary_tests ("fmod_u_u")
run_binary_tests ("fmod_v_u")
run_binary_tests ("fmod_u_v")
run_binary_tests ("fmod_v_c")
run_binary_tests ("fmod_c_v")
run_binary_tests ("fmod_u_c")
run_binary_tests ("fmod_c_u")

run_binary_tests ("fmod_u_u_float")
run_binary_tests ("fmod_v_u_float")
run_binary_tests ("fmod_u_v_float")
run_binary_tests ("fmod_v_v_float")
run_binary_tests ("fmod_v_c_float")
run_binary_tests ("fmod_c_v_float")
run_binary_tests ("fmod_u_c_float")
run_binary_tests ("fmod_c_u_float")

run_binary_tests ("step")
run_binary_tests ("step_u_u")
run_binary_tests ("step_v_u")
run_binary_tests ("step_u_v")
run_binary_tests ("step_v_c")
run_binary_tests ("step_c_v")
run_binary_tests ("step_u_c")
run_binary_tests ("step_c_u")


run_binary_tests ("add")
run_binary_tests ("add_u_u")
run_binary_tests ("add_v_u")
run_binary_tests ("add_u_v")
run_binary_tests ("add_v_c")
run_binary_tests ("add_c_v")
run_binary_tests ("add_u_c")
run_binary_tests ("add_c_u")

run_binary_tests ("sub")
run_binary_tests ("sub_u_u")
run_binary_tests ("sub_v_u")
run_binary_tests ("sub_u_v")
run_binary_tests ("sub_v_c")
run_binary_tests ("sub_c_v")
run_binary_tests ("sub_u_c")
run_binary_tests ("sub_c_u")

run_binary_tests ("mul")
run_binary_tests ("mul_u_u")
run_binary_tests ("mul_v_u")
run_binary_tests ("mul_u_v")
run_binary_tests ("mul_v_c")
run_binary_tests ("mul_c_v")
run_binary_tests ("mul_u_c")
run_binary_tests ("mul_c_u")

run_binary_tests ("div")
run_binary_tests ("div_u_u")
run_binary_tests ("div_v_u")
run_binary_tests ("div_u_v")
run_binary_tests ("div_v_c")
run_binary_tests ("div_c_v")
run_binary_tests ("div_u_c")
run_binary_tests ("div_c_u")


# expect a few LSB failures
failthresh = 0.008
failpercent = 3
