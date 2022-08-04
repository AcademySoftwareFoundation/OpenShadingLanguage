#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

def run_multi_type_test (suffix, dimension, tagname, options) :
    global command
    command += testshade("-g 512 512 -center --param numStripes 64 "+options+" -od uint8 -o out_float out_float_"+tagname+".tif -o out_color out_color_"+tagname+".tif test_pnoise"+dimension+"_"+suffix)
    global outputs     
    outputs.append ("out_float_"+tagname+".tif")
    outputs.append ("out_color_"+tagname+".tif")
    return

def run_pnoise_tests (suffix, dimension) :
    run_multi_type_test (suffix, dimension, dimension+"_"+suffix, "")
    run_multi_type_test (suffix, dimension, dimension+"_DxDy_"+suffix, "--vary_pdxdy --param derivs 1")
    return

# Run 1d noise tests with permuations of varying, uniform, and constant arguments
# NOTE: 2nd argument is the period
# Choose not to test variation of a constant argument, as that should be constant folded 
run_pnoise_tests ("v_v", "1d")
run_pnoise_tests ("u_u", "1d")
run_pnoise_tests ("u_v", "1d")
run_pnoise_tests ("v_u", "1d")
run_pnoise_tests ("c_v", "1d")
run_pnoise_tests ("v_c", "1d")
run_pnoise_tests ("u_c", "1d")
run_pnoise_tests ("c_u", "1d")

# Run 2d noise tests with permuations of varying, uniform, and constant arguments
# but leaving 3rd and 4th arguments (the period parameters) as varying to
# limit the number of permutations we are testing 
# Choose not to test variation of only constant arguments, as that should be constant folded 
run_pnoise_tests ("v_v_v_v", "2d")
run_pnoise_tests ("u_u_v_v", "2d")
run_pnoise_tests ("u_v_v_v", "2d")
run_pnoise_tests ("v_u_v_v", "2d")
run_pnoise_tests ("c_v_v_v", "2d")
run_pnoise_tests ("v_c_v_v", "2d")
run_pnoise_tests ("u_c_v_v", "2d")
run_pnoise_tests ("c_u_v_v", "2d")

# Run 3d noise tests with permuations of varying, uniform, and constant arguments
# NOTE: 2nd argument is the period
run_pnoise_tests ("v_v", "3d")
run_pnoise_tests ("u_u", "3d")
run_pnoise_tests ("u_v", "3d")
run_pnoise_tests ("v_u", "3d")
run_pnoise_tests ("c_v", "3d")
run_pnoise_tests ("v_c", "3d")
run_pnoise_tests ("u_c", "3d")
run_pnoise_tests ("c_u", "3d")

# Run 4d noise tests with permuations of varying, uniform, and constant arguments
# but leaving 3rd and 4th arguments (the period parameters) as varying to
# limit the number of permutations we are testing 
# Choose not to test variation of only constant arguments, as that should be constant folded 
run_pnoise_tests ("v_v_v_v", "4d")
run_pnoise_tests ("u_u_v_v", "4d")
run_pnoise_tests ("u_v_v_v", "4d")
run_pnoise_tests ("v_u_v_v", "4d")
run_pnoise_tests ("c_v_v_v", "4d")
run_pnoise_tests ("v_c_v_v", "4d")
run_pnoise_tests ("u_c_v_v", "4d")
run_pnoise_tests ("c_u_v_v", "4d")


# expect a few LSB failures
failthresh = 0.008
failpercent = 3

