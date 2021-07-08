#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


command += testshade("-t 1 -g 32 32 --vary_pdxdy -od uint8 -o Cout calculatenormal_u_point.tif test_calculatenormal_u_point")
command += testshade("-t 1 -g 32 32 --vary_pdxdy -od uint8 -o Cout calculatenormal_v_point.tif test_calculatenormal_v_point")
command += testshade("-t 1 -g 32 32 --vary_pdxdy -od uint8 -o Cout calculatenormal_dpoint.tif test_calculatenormal_dpoint")

outputs.append ("calculatenormal_u_point.tif")
outputs.append ("calculatenormal_v_point.tif")
outputs.append ("calculatenormal_dpoint.tif")

# expect a few LSB failures
failthresh = 0.008
failpercent = 3

