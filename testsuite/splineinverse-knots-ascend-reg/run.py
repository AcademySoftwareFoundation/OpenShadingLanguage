#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade("-t 1 -g 64 64 --center -od uint8 -o Fout splineinverse_c_float_v_floatarray.tif test_splineinverse_c_float_v_floatarray")
command += testshade("-t 1 -g 64 64 --center -od uint8 -o Fout splineinverse_c_float_u_floatarray.tif test_splineinverse_c_float_u_floatarray")
command += testshade("-t 1 -g 64 64 --center -od uint8 -o Fout splineinverse_c_float_c_floatarray.tif test_splineinverse_c_float_c_floatarray")
outputs.append ("splineinverse_c_float_v_floatarray.tif")
outputs.append ("splineinverse_c_float_u_floatarray.tif")
outputs.append ("splineinverse_c_float_c_floatarray.tif")

command += testshade("-t 1 -g 64 64 --center -od uint8 -o Fout splineinverse_u_float_v_floatarray.tif test_splineinverse_u_float_v_floatarray")
command += testshade("-t 1 -g 64 64 --center -od uint8 -o Fout splineinverse_u_float_u_floatarray.tif test_splineinverse_u_float_u_floatarray")
command += testshade("-t 1 -g 64 64 --center -od uint8 -o Fout splineinverse_u_float_c_floatarray.tif test_splineinverse_u_float_c_floatarray")
outputs.append ("splineinverse_u_float_v_floatarray.tif")
outputs.append ("splineinverse_u_float_u_floatarray.tif")
outputs.append ("splineinverse_u_float_c_floatarray.tif")

command += testshade("-t 1 -g 64 64 --center -od uint8 -o Fout splineinverse_v_float_v_floatarray.tif test_splineinverse_v_float_v_floatarray")
command += testshade("-t 1 -g 64 64 --center -od uint8 -o Fout splineinverse_v_float_u_floatarray.tif test_splineinverse_v_float_u_floatarray")
command += testshade("-t 1 -g 64 64 --center -od uint8 -o Fout splineinverse_v_float_c_floatarray.tif test_splineinverse_v_float_c_floatarray")
outputs.append ("splineinverse_v_float_v_floatarray.tif")
outputs.append ("splineinverse_v_float_u_floatarray.tif")
outputs.append ("splineinverse_v_float_c_floatarray.tif")


command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 --center -od uint8 -o ValDxDyOut deriv_splineinverse_c_float_v_floatarray.tif test_deriv_splineinverse_c_float_v_floatarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 --center -od uint8 -o ValDxDyOut deriv_splineinverse_c_float_u_floatarray.tif test_deriv_splineinverse_c_float_u_floatarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 --center -od uint8 -o ValDxDyOut deriv_splineinverse_c_float_c_floatarray.tif test_deriv_splineinverse_c_float_c_floatarray")
outputs.append ("deriv_splineinverse_c_float_v_floatarray.tif")
outputs.append ("deriv_splineinverse_c_float_u_floatarray.tif")
outputs.append ("deriv_splineinverse_c_float_c_floatarray.tif")

command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 --center -od uint8 -o ValDxDyOut deriv_splineinverse_u_float_v_floatarray.tif test_deriv_splineinverse_u_float_v_floatarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 --center -od uint8 -o ValDxDyOut deriv_splineinverse_u_float_u_floatarray.tif test_deriv_splineinverse_u_float_u_floatarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 --center -od uint8 -o ValDxDyOut deriv_splineinverse_u_float_c_floatarray.tif test_deriv_splineinverse_u_float_c_floatarray")
outputs.append ("deriv_splineinverse_u_float_v_floatarray.tif")
outputs.append ("deriv_splineinverse_u_float_u_floatarray.tif")
outputs.append ("deriv_splineinverse_u_float_c_floatarray.tif")

command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 --center -od uint8 -o ValDxDyOut deriv_splineinverse_v_float_v_floatarray.tif test_deriv_splineinverse_v_float_v_floatarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 --center -od uint8 -o ValDxDyOut deriv_splineinverse_v_float_u_floatarray.tif test_deriv_splineinverse_v_float_u_floatarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 --center -od uint8 -o ValDxDyOut deriv_splineinverse_v_float_c_floatarray.tif test_deriv_splineinverse_v_float_c_floatarray")
outputs.append ("deriv_splineinverse_v_float_v_floatarray.tif")
outputs.append ("deriv_splineinverse_v_float_u_floatarray.tif")
outputs.append ("deriv_splineinverse_v_float_c_floatarray.tif")



# expect a few LSB failures
failthresh = 0.008
failpercent = 3

