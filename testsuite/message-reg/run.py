#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

def run_test (suffix) :
    global command
    tagname = suffix;
    command += testshade("-t 1 -g 64 64 -od uint8"
                         +" --param numStripes 16 --layer alayer a_"+suffix
                         +" --param numStripes 8 --layer blayer b_"+suffix
                         +" --connect alayer dummy blayer dummy"
                         +" -o out_string out_string_"+tagname+".tif"
                         +" -o out_int out_int_"+tagname+".tif"
                         +" -o out_float out_float_"+tagname+".tif"
                         +" -o out_color out_color_"+tagname+".tif"
                         +" -o out_matrix out_matrix_"+tagname+".tif"
                         +" -o out_strings0 out_strings0_"+tagname+".tif"
                         +" -o out_strings1 out_strings1_"+tagname+".tif"
                         +" -o out_strings2 out_strings2_"+tagname+".tif"
                         +" -o out_ints0 out_ints0_"+tagname+".tif"
                         +" -o out_ints1 out_ints1_"+tagname+".tif"
                         +" -o out_ints2 out_ints2_"+tagname+".tif"
                         +" -o out_floats0 out_floats0_"+tagname+".tif"
                         +" -o out_floats1 out_floats1_"+tagname+".tif"
                         +" -o out_floats2 out_floats2_"+tagname+".tif"
                         +" -o out_colors0 out_colors0_"+tagname+".tif"
                         +" -o out_colors1 out_colors1_"+tagname+".tif"
                         +" -o out_colors2 out_colors2_"+tagname+".tif"
                         +" -o out_matrices0 out_matrices0_"+tagname+".tif"
                         +" -o out_matrices1 out_matrices1_"+tagname+".tif"
                         +" -o out_matrices2 out_matrices2_"+tagname+".tif"
                         +" -o out_result out_result_"+tagname+".tif"
                         )
    
    global outputs     
    outputs.append ("out_string_"+tagname+".tif")
    outputs.append ("out_int_"+tagname+".tif")
    outputs.append ("out_float_"+tagname+".tif")
    outputs.append ("out_color_"+tagname+".tif")
    outputs.append ("out_matrix_"+tagname+".tif")
    outputs.append ("out_strings0_"+tagname+".tif")
    outputs.append ("out_strings1_"+tagname+".tif")
    outputs.append ("out_strings2_"+tagname+".tif")
    outputs.append ("out_ints0_"+tagname+".tif")
    outputs.append ("out_ints1_"+tagname+".tif")
    outputs.append ("out_ints2_"+tagname+".tif")
    outputs.append ("out_floats0_"+tagname+".tif")
    outputs.append ("out_floats1_"+tagname+".tif")
    outputs.append ("out_floats2_"+tagname+".tif")
    outputs.append ("out_colors0_"+tagname+".tif")
    outputs.append ("out_colors1_"+tagname+".tif")
    outputs.append ("out_colors2_"+tagname+".tif")
    outputs.append ("out_matrices0_"+tagname+".tif")
    outputs.append ("out_matrices1_"+tagname+".tif")
    outputs.append ("out_matrices2_"+tagname+".tif")
    outputs.append ("out_result_"+tagname+".tif")    
    return

run_test ("c")
run_test ("u")
run_test ("v")

# expect a few LSB failures
failthresh = 0.008
failpercent = 3

