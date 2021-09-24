#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#################################################
# select(float, float, int)
#################################################
command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_float_u_float_u_int.tif test_u_float_u_float_u_int")
outputs.append ("u_float_u_float_u_int.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_float_u_float_u_int.tif test_v_float_u_float_u_int")
outputs.append ("v_float_u_float_u_int.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_float_v_float_u_int.tif test_u_float_v_float_u_int")
outputs.append ("u_float_v_float_u_int.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_float_v_float_u_int.tif test_v_float_v_float_u_int")
outputs.append ("v_float_v_float_u_int.tif")


command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_float_u_float_v_int.tif test_u_float_u_float_v_int")
outputs.append ("u_float_u_float_v_int.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_float_u_float_v_int.tif test_v_float_u_float_v_int")
outputs.append ("v_float_u_float_v_int.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_float_v_float_v_int.tif test_u_float_v_float_v_int")
outputs.append ("u_float_v_float_v_int.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_float_v_float_v_int.tif test_v_float_v_float_v_int")
outputs.append ("v_float_v_float_v_int.tif")


#################################################
# select(float, float, float)
#################################################
command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_float_u_float_u_float.tif test_u_float_u_float_u_float")
outputs.append ("u_float_u_float_u_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_float_u_float_u_float.tif test_v_float_u_float_u_float")
outputs.append ("v_float_u_float_u_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_float_v_float_u_float.tif test_u_float_v_float_u_float")
outputs.append ("u_float_v_float_u_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_float_v_float_u_float.tif test_v_float_v_float_u_float")
outputs.append ("v_float_v_float_u_float.tif")


command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_float_u_float_v_float.tif test_u_float_u_float_v_float")
outputs.append ("u_float_u_float_v_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_float_u_float_v_float.tif test_v_float_u_float_v_float")
outputs.append ("v_float_u_float_v_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_float_v_float_v_float.tif test_u_float_v_float_v_float")
outputs.append ("u_float_v_float_v_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_float_v_float_v_float.tif test_v_float_v_float_v_float")
outputs.append ("v_float_v_float_v_float.tif")


#################################################
# select(color, color, int)
#################################################
command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_color_u_color_u_int.tif test_u_color_u_color_u_int")
outputs.append ("u_color_u_color_u_int.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_color_u_color_u_int.tif test_v_color_u_color_u_int")
outputs.append ("v_color_u_color_u_int.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_color_v_color_u_int.tif test_u_color_v_color_u_int")
outputs.append ("u_color_v_color_u_int.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_color_v_color_u_int.tif test_v_color_v_color_u_int")
outputs.append ("v_color_v_color_u_int.tif")


command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_color_u_color_v_int.tif test_u_color_u_color_v_int")
outputs.append ("u_color_u_color_v_int.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_color_u_color_v_int.tif test_v_color_u_color_v_int")
outputs.append ("v_color_u_color_v_int.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_color_v_color_v_int.tif test_u_color_v_color_v_int")
outputs.append ("u_color_v_color_v_int.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_color_v_color_v_int.tif test_v_color_v_color_v_int")
outputs.append ("v_color_v_color_v_int.tif")


#################################################
# select(color, color, float)
#################################################
command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_color_u_color_u_float.tif test_u_color_u_color_u_float")
outputs.append ("u_color_u_color_u_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_color_u_color_u_float.tif test_v_color_u_color_u_float")
outputs.append ("v_color_u_color_u_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_color_v_color_u_float.tif test_u_color_v_color_u_float")
outputs.append ("u_color_v_color_u_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_color_v_color_u_float.tif test_v_color_v_color_u_float")
outputs.append ("v_color_v_color_u_float.tif")


command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_color_u_color_v_float.tif test_u_color_u_color_v_float")
outputs.append ("u_color_u_color_v_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_color_u_color_v_float.tif test_v_color_u_color_v_float")
outputs.append ("v_color_u_color_v_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_color_v_color_v_float.tif test_u_color_v_color_v_float")
outputs.append ("u_color_v_color_v_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_color_v_color_v_float.tif test_v_color_v_color_v_float")
outputs.append ("v_color_v_color_v_float.tif")


#################################################
# select(color, color, float)
#################################################
command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_color_u_color_u_color.tif test_u_color_u_color_u_color")
outputs.append ("u_color_u_color_u_color.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_color_u_color_u_color.tif test_v_color_u_color_u_color")
outputs.append ("v_color_u_color_u_color.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_color_v_color_u_color.tif test_u_color_v_color_u_color")
outputs.append ("u_color_v_color_u_color.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_color_v_color_u_color.tif test_v_color_v_color_u_color")
outputs.append ("v_color_v_color_u_color.tif")


command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_color_u_color_v_color.tif test_u_color_u_color_v_color")
outputs.append ("u_color_u_color_v_color.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_color_u_color_v_color.tif test_v_color_u_color_v_color")
outputs.append ("v_color_u_color_v_color.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout u_color_v_color_v_color.tif test_u_color_v_color_v_color")
outputs.append ("u_color_v_color_v_color.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout v_color_v_color_v_color.tif test_v_color_v_color_v_color")
outputs.append ("v_color_v_color_v_color.tif")








# expect a few LSB failures
failthresh = 0.008
failpercent = 3

