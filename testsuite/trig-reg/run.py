#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

def run_multi_type_test (opname, tagname, options) :
    global command
    command += testshade("-g 32 32 -center --param numStripes 16 "+options+" -od uint8 -o out_float out_float_"+tagname+".tif -o out_color out_color_"+tagname+".tif -o out_point out_point_"+tagname+".tif -o out_vector out_vector_"+tagname+".tif -o out_normal out_normal_"+tagname+".tif test_"+opname)
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
run_unary_tests ("sin")
run_unary_tests ("sin_u")

run_unary_tests ("cos")
run_unary_tests ("cos_u")

run_unary_tests ("tan")
run_unary_tests ("tan_u")


run_unary_tests ("asin")
run_unary_tests ("asin_u")

run_unary_tests ("acos")
run_unary_tests ("acos_u")

run_unary_tests ("atan")
run_unary_tests ("atan_u")

run_unary_tests ("cosh")
run_unary_tests ("cosh_u")

run_unary_tests ("tanh")
run_unary_tests ("tanh_u")


def run_binary_tests (opname) :
    run_multi_type_test (opname, opname, "")
    run_multi_type_test (opname, "Dx_"+opname, "--vary_pdxdy --param derivX 1")
    run_multi_type_test (opname, "Dy_"+opname, "--vary_pdxdy --param derivY 1")
    return

# Run binary tests with mixing combinations of varying(v), uniform(u), and constant(c) parameters.
# Test with constants to pass non-deriviative parameters to functions
# Choose not to test variation of 2 constant argument, as that should be constant folded 
run_binary_tests ("atan2")
run_binary_tests ("atan2_u_u")
run_binary_tests ("atan2_v_u")
run_binary_tests ("atan2_u_v")
run_binary_tests ("atan2_v_c")
run_binary_tests ("atan2_c_v")
run_binary_tests ("atan2_u_c")
run_binary_tests ("atan2_c_u")


def run_sincos_tests (opname) :
    run_multi_type_test (opname, opname, "")
    run_multi_type_test (opname, "DxSin_"+opname, "--vary_pdxdy --param derivSinX 1")
    run_multi_type_test (opname, "DxCos_"+opname, "--vary_pdxdy --param derivCosX 1")
    run_multi_type_test (opname, "DxSin_DxCos_"+opname, "--vary_pdxdy --param derivSinX 1 --param derivCosX 1")
    run_multi_type_test (opname, "DySin_"+opname, "--vary_pdxdy --param derivSinY 1")
    run_multi_type_test (opname, "DyCos_"+opname, "--vary_pdxdy --param derivCosY 1")
    run_multi_type_test (opname, "DySin_DyCos_"+opname, "--vary_pdxdy --param derivSinY 1 --param derivCosY 1")
    return

run_sincos_tests("sincos")
run_sincos_tests("sincos_u")


# expect a few LSB failures
failthresh = 0.008
failpercent = 3

