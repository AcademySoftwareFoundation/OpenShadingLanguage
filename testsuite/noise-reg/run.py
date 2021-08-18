#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

def run_multi_type_test (suffix, dimension, tagname, options) :
    global command
    command += testshade("-g 512 512 -center --param numStripes 64 "+options+" -od uint8 -o out_float out_float_"+tagname+".tif -o out_color out_color_"+tagname+".tif test_noise"+dimension+"_"+suffix)
    global outputs     
    outputs.append ("out_float_"+tagname+".tif")
    outputs.append ("out_color_"+tagname+".tif")
    return

def run_noise_tests (suffix, dimension) :
    run_multi_type_test (suffix, dimension, dimension+"_"+suffix, "")
    run_multi_type_test (suffix, dimension, dimension+"_DxDy_"+suffix, "--vary_pdxdy --param derivs 1")
    return

# Run 1d noise tests with varying and a 2nd time with uniform(u) argument to the op
# Choose not to test variation of a constant argument, as that should be constant folded 
run_noise_tests ("varying", "1d")
run_noise_tests ("uniform", "1d")

# Run 2d noise tests with permuations of varying, uniform, and constant arguments
# Choose not to test variation of only constant arguments, as that should be constant folded 
run_noise_tests ("v_v", "2d")
run_noise_tests ("u_u", "2d")
run_noise_tests ("u_v", "2d")
run_noise_tests ("v_u", "2d")
run_noise_tests ("c_v", "2d")
run_noise_tests ("v_c", "2d")
run_noise_tests ("u_c", "2d")
run_noise_tests ("c_u", "2d")

# Run 3d noise tests with varying and a 2nd time with uniform(u) argument to the op
run_noise_tests ("varying", "3d")
run_noise_tests ("uniform", "3d")

# Run 4d noise tests with permuations of varying, uniform, and constant arguments
# Choose not to test variation of only constant arguments, as that should be constant folded 
run_noise_tests ("v_v", "4d")
run_noise_tests ("u_u", "4d")
run_noise_tests ("u_v", "4d")
run_noise_tests ("v_u", "4d")
run_noise_tests ("c_v", "4d")
run_noise_tests ("v_c", "4d")
run_noise_tests ("u_c", "4d")
run_noise_tests ("c_u", "4d")


# expect a few LSB failures
failthresh = 0.008
failpercent = 3
hardfail = 0.08
