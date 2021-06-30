#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_area_point_u_point.tif test_area_u_point")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o Cout out_area_point_v_point.tif test_area_v_point")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o Cout out_area_point_v_dpoint.tif test_area_v_dpoint")
outputs.append ("out_area_point_u_point.tif")
outputs.append ("out_area_point_v_point.tif")
outputs.append ("out_area_point_v_dpoint.tif")

command += testshade("--vary_pdxdy -t 1 -g 64 64 -od uint8 -o Cout out_area_point_v_point_B.tif test_area_v_point_B")
command += testshade("--vary_pdxdy -t 1 -g 64 64 -od uint8 -o Cout out_area_point_v_dpoint_B.tif test_area_v_dpoint_B")
outputs.append ("out_area_point_v_point_B.tif")
outputs.append ("out_area_point_v_dpoint_B.tif")


# expect a few LSB failures
failthresh = 0.008
failpercent = 3

