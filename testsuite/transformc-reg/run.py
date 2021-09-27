#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


def run_2colorspace_tests (fromspace, tospace) :
    global command
    command += testshade("-t 1 -g 32 32 -param fromspace "+fromspace+" -param tospace "+tospace+" -od uint8 -o Cout out_transformc_"+fromspace+"_"+tospace+"_u_color.tif test_transformc_u_space_u_space_u_color")
    command += testshade("-t 1 -g 32 32 -param fromspace "+fromspace+" -param tospace "+tospace+" -od uint8 -o Cout out_transformc_"+fromspace+"_"+tospace+"_v_color.tif test_transformc_u_space_u_space_v_color")
    
    
    global outputs     
    outputs.append ("out_transformc_"+fromspace+"_"+tospace+"_u_color.tif")
    outputs.append ("out_transformc_"+fromspace+"_"+tospace+"_v_color.tif")
    return


def run_colorspace_tests (colorspace) :
    global command
    command += testshade("-t 1 -g 32 32 -param colorspace "+colorspace+" -od uint8 -o Cout out_transformc_"+colorspace+"_u_color.tif test_transformc_u_space_u_color")
    command += testshade("-t 1 -g 32 32 -param colorspace "+colorspace+" -od uint8 -o Cout out_transformc_"+colorspace+"_v_color.tif test_transformc_u_space_v_color")
    
    global outputs     
    outputs.append ("out_transformc_"+colorspace+"_u_color.tif")
    outputs.append ("out_transformc_"+colorspace+"_v_color.tif")
    
    run_2colorspace_tests (colorspace, "rgb")
    run_2colorspace_tests (colorspace, "RGB")
    run_2colorspace_tests (colorspace, "hsv")
    run_2colorspace_tests (colorspace, "hsl")
    run_2colorspace_tests (colorspace, "YIQ")
    run_2colorspace_tests (colorspace, "XYZ")
    run_2colorspace_tests (colorspace, "xyY")
    
    command += testshade("-t 1 -g 32 32 -param fromspace "+colorspace+" -od uint8 -o Cout out_transformc_"+colorspace+"_v_space_v_color.tif test_transformc_u_space_v_space_v_color")
    outputs.append ("out_transformc_"+colorspace+"_v_space_v_color.tif")

    command += testshade("-t 1 -g 32 32 -param tospace "+colorspace+" -od uint8 -o Cout out_transformc_v_space_"+colorspace+"_v_color.tif test_transformc_v_space_u_space_v_color")
    outputs.append ("out_transformc_v_space_"+colorspace+"_v_color.tif")
    
    return

run_colorspace_tests ("rgb")
run_colorspace_tests ("RGB")
run_colorspace_tests ("hsv")
run_colorspace_tests ("hsl")
run_colorspace_tests ("YIQ")
run_colorspace_tests ("XYZ")
run_colorspace_tests ("xyY")


command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_transformc_v_space_u_color.tif test_transformc_v_space_u_color")
outputs.append ("out_transformc_v_space_u_color.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_transformc_v_space_v_color.tif test_transformc_v_space_v_color")
outputs.append ("out_transformc_v_space_v_color.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_transformc_v_space_v_space_v_color.tif test_transformc_v_space_v_space_v_color")
outputs.append ("out_transformc_v_space_v_space_v_color.tif")


# expect a few LSB failures
failthresh = 0.008
failpercent = 3

