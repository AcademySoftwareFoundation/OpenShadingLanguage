#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_color_u_color.tif test_luminance_u_color")
outputs.append ("out_color_u_color.tif")

command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_color_v_color.tif test_luminance_v_color")
outputs.append ("out_color_v_color.tif")

command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o Cout out_color_v_dcolor.tif test_luminance_v_dcolor")
outputs.append ("out_color_v_dcolor.tif")

# expect a few LSB failures
failthresh = 0.008
failpercent = 3

