#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_color_u_float.tif test_color_u_float")
command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_color_v_float.tif test_color_v_float")
command += testshade("--vary_udxdy --vary_udxdy -t 1 -g 32 32 -od uint8 -o Cout out_color_v_dfloat.tif test_color_v_dfloat")
outputs.append ("out_color_u_float.tif")
outputs.append ("out_color_v_float.tif")
outputs.append ("out_color_v_dfloat.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_color_3xu_float.tif test_color_3xu_float")
command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_color_3xv_float.tif test_color_3xv_float")
command += testshade("--vary_udxdy --vary_udxdy -t 1 -g 32 32 -od uint8 -o Cout out_color_3xv_dfloat.tif test_color_3xv_dfloat")
outputs.append ("out_color_3xu_float.tif")
outputs.append ("out_color_3xv_float.tif")
outputs.append ("out_color_3xv_dfloat.tif")

def run_colorspace_tests (colorspace) :
    global command
    command += testshade("-t 1 -g 32 32 -param colorspace "+colorspace+" -od uint8 -o Cout out_"+colorspace+"_3xu_float.tif test_color_u_space_3xu_float")
    command += testshade("-t 1 -g 32 32 -param colorspace "+colorspace+" -od uint8 -o Cout out_"+colorspace+"_3xv_float.tif test_color_u_space_3xv_float")
    # NOTE: current single point impl just 0's the derivs out, tests are to make sure we don't miss a fix for that
    #       So expect all deriv based outputs to be black images
    command += testshade("-t 1 -g 32 32 --vary_udxdy --vary_udxdy -param colorspace "+colorspace+" -od uint8 -o Cout out_"+colorspace+"_3xv_dfloat.tif test_color_u_space_3xv_dfloat")
    
    global outputs     
    outputs.append ("out_"+colorspace+"_3xu_float.tif")
    outputs.append ("out_"+colorspace+"_3xv_float.tif")
    outputs.append ("out_"+colorspace+"_3xv_dfloat.tif")
    return

run_colorspace_tests ("rgb")
run_colorspace_tests ("RGB")
run_colorspace_tests ("hsv")
run_colorspace_tests ("hsl")
run_colorspace_tests ("YIQ")
run_colorspace_tests ("XYZ")
run_colorspace_tests ("xyY")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_v_space_3xu_float.tif test_color_v_space_3xu_float")
command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_v_space_3xv_float.tif test_color_v_space_3xv_float")
command += testshade("-t 1 -g 32 32 --vary_udxdy --vary_udxdy -od uint8 -o Cout out_v_space_3xv_dfloat.tif test_color_v_space_3xv_dfloat")
outputs.append ("out_v_space_3xu_float.tif")
outputs.append ("out_v_space_3xv_float.tif")
outputs.append ("out_v_space_3xv_dfloat.tif")



# expect a few LSB failures
failthresh = 0.008
failpercent = 3

