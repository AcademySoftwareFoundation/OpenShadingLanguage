#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


command += testshade("-t 1 -g 64 64 -od uint8 -o Cout sout_vector_u_vector.tif test_length_u_vector")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout sout_vector_v_vector.tif test_length_v_vector")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o Cout sout_vector_v_dvector.tif test_length_v_dvector")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout sout_normal_u_normal.tif test_length_u_normal")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout sout_normal_v_normal.tif test_length_v_normal")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o Cout sout_normal_v_dnormal.tif test_length_v_dnormal")
outputs.append ("sout_vector_u_vector.tif")
outputs.append ("sout_vector_v_vector.tif")
outputs.append ("sout_vector_v_dvector.tif")
outputs.append ("sout_normal_u_normal.tif")
outputs.append ("sout_normal_v_normal.tif")
outputs.append ("sout_normal_v_dnormal.tif")


# expect a few LSB failures
failthresh = 0.008
failpercent = 3

