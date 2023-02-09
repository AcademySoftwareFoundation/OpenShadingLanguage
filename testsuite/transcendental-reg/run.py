#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

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

def run_unary_tests (opname) :
    run_multi_type_test (opname, opname, "")
    run_multi_type_test (opname, "Dx_"+opname, "--vary_pdxdy --param derivX 1")
    run_multi_type_test (opname, "Dy_"+opname, "--vary_pdxdy --param derivY 1")
    return


# Run unary tests with varying and a 2nd time with uniform(u) argument to the op
# Choose not to test variation of a constant argument, as that should be constant folded 
run_unary_tests ("logb")
run_unary_tests ("logb_u")

run_unary_tests ("log")
run_unary_tests ("log_u")

run_unary_tests ("log2")
run_unary_tests ("log2_u")

run_unary_tests ("log10")
run_unary_tests ("log10_u")

run_unary_tests ("exp")
run_unary_tests ("exp_u")

run_unary_tests ("exp2")
run_unary_tests ("exp2_u")

run_unary_tests ("expm1")
run_unary_tests ("expm1_u")

run_unary_tests ("cbrt")
run_unary_tests ("cbrt_u")

def run_float_test (opname, tagname, options) :
    global command
    command += testshade("-g 32 32 --param numStripes 16 "+options+" -od uint8 -o out_float out_float_"+tagname+".tif test_"+opname)
    global outputs     
    outputs.append ("out_float_"+tagname+".tif")
    return

def run_unary_float_tests (opname) :
    run_float_test (opname, opname, "")
    run_float_test (opname, "Dx_"+opname, "--vary_pdxdy --param derivX 1")
    run_float_test (opname, "Dy_"+opname, "--vary_pdxdy --param derivY 1")
    return

run_unary_float_tests ("erf")
run_unary_float_tests ("erf_u")

run_unary_float_tests ("erfc")
run_unary_float_tests ("erfc_u")


def run_binary_tests (opname) :
    run_multi_type_test (opname, opname, "")
    run_multi_type_test (opname, "Dx_"+opname, "--vary_pdxdy --param derivX 1")
    run_multi_type_test (opname, "Dy_"+opname, "--vary_pdxdy --param derivY 1")
    return

# Run binary tests with mixing combinations of varying(v), uniform(u), and constant(c) parameters.
# Test with constants to pass non-derivative parameters to functions
# Choose not to test variation of 2 constant argument, as that should be constant folded 
run_binary_tests ("pow")
run_binary_tests ("pow_u_u")
run_binary_tests ("pow_v_u")
run_binary_tests ("pow_u_v")
run_binary_tests ("pow_v_c")
run_binary_tests ("pow_c_v")
run_binary_tests ("pow_u_c")
run_binary_tests ("pow_c_u")

run_binary_tests ("pow_u_u_float")
run_binary_tests ("pow_v_u_float")
run_binary_tests ("pow_u_v_float")
run_binary_tests ("pow_v_v_float")
run_binary_tests ("pow_v_c_float")
run_binary_tests ("pow_c_v_float")
run_binary_tests ("pow_u_c_float")
run_binary_tests ("pow_c_u_float")

# expect a few LSB failures
failthresh = 0.008
failpercent = 3

