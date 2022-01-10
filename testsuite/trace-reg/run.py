#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

def run_test (suffix) :
    global command
    tagname = suffix;
    command += testshade("-t 1 -g 64 64 --param numStripes 16 -od uint8 -o Cout out.tif test_trace")
    
    global outputs     
    outputs.append ("out.tif")
    return

run_test ("")

# expect a few LSB failures
failthresh = 0.008
failpercent = 3

