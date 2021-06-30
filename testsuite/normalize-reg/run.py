#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


command += testshade("--center -t 1 -g 32 32 -od uint8 -o Cout normalize_u_vector.tif test_normalize_u_vector")
command += testshade("--center -t 1 -g 32 32 -od uint8 -o Cout normalize_v_vector.tif test_normalize_v_vector")
command += testshade("--center --vary_udxdy --vary_udxdy -t 1 -g 32 32 -od uint8 -o Cout normalize_v_dvector.tif test_normalize_v_dvector")
outputs.append ("normalize_u_vector.tif")
outputs.append ("normalize_v_vector.tif")
outputs.append ("normalize_v_dvector.tif")

command += testshade("--center -t 1 -g 32 32 -od uint8 -o Cout normalize_u_normal.tif test_normalize_u_normal")
command += testshade("--center -t 1 -g 32 32 -od uint8 -o Cout normalize_v_normal.tif test_normalize_v_normal")
command += testshade("--center --vary_udxdy --vary_udxdy -t 1 -g 32 32 -od uint8 -o Cout normalize_v_dnormal.tif test_normalize_v_dnormal")
outputs.append ("normalize_u_normal.tif")
outputs.append ("normalize_v_normal.tif")
outputs.append ("normalize_v_dnormal.tif")


# expect a few LSB failures
failthresh = 0.008
failpercent = 3

