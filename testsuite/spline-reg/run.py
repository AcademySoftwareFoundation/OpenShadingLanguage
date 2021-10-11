#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


command += testshade("-t 1 -g 64 64 -od uint8 -o Fout spline_c_float_v_floatarray.tif test_spline_c_float_v_floatarray")
command += testshade("-t 1 -g 64 64 -od uint8 -o Fout spline_c_float_u_floatarray.tif test_spline_c_float_u_floatarray")
command += testshade("-t 1 -g 64 64 -od uint8 -o Fout spline_c_float_c_floatarray.tif test_spline_c_float_c_floatarray")
outputs.append ("spline_c_float_v_floatarray.tif")
outputs.append ("spline_c_float_u_floatarray.tif")
outputs.append ("spline_c_float_c_floatarray.tif")

command += testshade("-t 1 -g 64 64 -od uint8 -o Fout spline_u_float_v_floatarray.tif test_spline_u_float_v_floatarray")
command += testshade("-t 1 -g 64 64 -od uint8 -o Fout spline_u_float_u_floatarray.tif test_spline_u_float_u_floatarray")
command += testshade("-t 1 -g 64 64 -od uint8 -o Fout spline_u_float_c_floatarray.tif test_spline_u_float_c_floatarray")
outputs.append ("spline_u_float_v_floatarray.tif")
outputs.append ("spline_u_float_u_floatarray.tif")
outputs.append ("spline_u_float_c_floatarray.tif")

command += testshade("-t 1 -g 64 64 -od uint8 -o Fout spline_v_float_v_floatarray.tif test_spline_v_float_v_floatarray")
command += testshade("-t 1 -g 64 64 -od uint8 -o Fout spline_v_float_u_floatarray.tif test_spline_v_float_u_floatarray")
command += testshade("-t 1 -g 64 64 -od uint8 -o Fout spline_v_float_c_floatarray.tif test_spline_v_float_c_floatarray")
outputs.append ("spline_v_float_v_floatarray.tif")
outputs.append ("spline_v_float_u_floatarray.tif")
outputs.append ("spline_v_float_c_floatarray.tif")


command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValDxDyOut deriv_spline_c_float_v_floatarray.tif test_deriv_spline_c_float_v_floatarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValDxDyOut deriv_spline_c_float_u_floatarray.tif test_deriv_spline_c_float_u_floatarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValDxDyOut deriv_spline_c_float_c_floatarray.tif test_deriv_spline_c_float_c_floatarray")
outputs.append ("deriv_spline_c_float_v_floatarray.tif")
outputs.append ("deriv_spline_c_float_u_floatarray.tif")
outputs.append ("deriv_spline_c_float_c_floatarray.tif")

command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValDxDyOut deriv_spline_u_float_v_floatarray.tif test_deriv_spline_u_float_v_floatarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValDxDyOut deriv_spline_u_float_u_floatarray.tif test_deriv_spline_u_float_u_floatarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValDxDyOut deriv_spline_u_float_c_floatarray.tif test_deriv_spline_u_float_c_floatarray")
outputs.append ("deriv_spline_u_float_v_floatarray.tif")
outputs.append ("deriv_spline_u_float_u_floatarray.tif")
outputs.append ("deriv_spline_u_float_c_floatarray.tif")

command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValDxDyOut deriv_spline_v_float_v_floatarray.tif test_deriv_spline_v_float_v_floatarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValDxDyOut deriv_spline_v_float_u_floatarray.tif test_deriv_spline_v_float_u_floatarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValDxDyOut deriv_spline_v_float_c_floatarray.tif test_deriv_spline_v_float_c_floatarray")
outputs.append ("deriv_spline_v_float_v_floatarray.tif")
outputs.append ("deriv_spline_v_float_u_floatarray.tif")
outputs.append ("deriv_spline_v_float_c_floatarray.tif")



command += testshade("-t 1 -g 64 64 -od uint8 -o Cout spline_c_float_v_colorarray.tif test_spline_c_float_v_colorarray")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout spline_c_float_u_colorarray.tif test_spline_c_float_u_colorarray")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout spline_c_float_c_colorarray.tif test_spline_c_float_c_colorarray")
outputs.append ("spline_c_float_v_colorarray.tif")
outputs.append ("spline_c_float_u_colorarray.tif")
outputs.append ("spline_c_float_c_colorarray.tif")

command += testshade("-t 1 -g 64 64 -od uint8 -o Cout spline_u_float_v_colorarray.tif test_spline_u_float_v_colorarray")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout spline_u_float_u_colorarray.tif test_spline_u_float_u_colorarray")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout spline_u_float_c_colorarray.tif test_spline_u_float_c_colorarray")
outputs.append ("spline_u_float_v_colorarray.tif")
outputs.append ("spline_u_float_u_colorarray.tif")
outputs.append ("spline_u_float_c_colorarray.tif")

command += testshade("-t 1 -g 64 64 -od uint8 -o Cout spline_v_float_v_colorarray.tif test_spline_v_float_v_colorarray")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout spline_v_float_u_colorarray.tif test_spline_v_float_u_colorarray")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout spline_v_float_c_colorarray.tif test_spline_v_float_c_colorarray")
outputs.append ("spline_v_float_v_colorarray.tif")
outputs.append ("spline_v_float_u_colorarray.tif")
outputs.append ("spline_v_float_c_colorarray.tif")

command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValOut deriv_spline_c_float_v_colorarray.tif -o DxOut deriv_spline_c_float_v_colorarrayDx.tif -o DyOut deriv_spline_c_float_v_colorarrayDy.tif test_deriv_spline_c_float_v_colorarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValOut deriv_spline_c_float_u_colorarray.tif -o DxOut deriv_spline_c_float_u_colorarrayDx.tif -o DyOut deriv_spline_c_float_u_colorarrayDy.tif test_deriv_spline_c_float_u_colorarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValOut deriv_spline_c_float_c_colorarray.tif -o DxOut deriv_spline_c_float_c_colorarrayDx.tif -o DyOut deriv_spline_c_float_c_colorarrayDy.tif test_deriv_spline_c_float_c_colorarray")
outputs.append ("deriv_spline_c_float_v_colorarray.tif")
outputs.append ("deriv_spline_c_float_v_colorarrayDx.tif")
outputs.append ("deriv_spline_c_float_v_colorarrayDy.tif")
outputs.append ("deriv_spline_c_float_u_colorarray.tif")
outputs.append ("deriv_spline_c_float_u_colorarrayDx.tif")
outputs.append ("deriv_spline_c_float_u_colorarrayDy.tif")
outputs.append ("deriv_spline_c_float_c_colorarray.tif")
outputs.append ("deriv_spline_c_float_c_colorarrayDx.tif")
outputs.append ("deriv_spline_c_float_c_colorarrayDy.tif")

command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValOut deriv_spline_u_float_v_colorarray.tif -o DxOut deriv_spline_u_float_v_colorarrayDx.tif -o DyOut deriv_spline_u_float_v_colorarrayDy.tif test_deriv_spline_u_float_v_colorarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValOut deriv_spline_u_float_u_colorarray.tif -o DxOut deriv_spline_u_float_u_colorarrayDx.tif -o DyOut deriv_spline_u_float_u_colorarrayDy.tif test_deriv_spline_u_float_u_colorarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValOut deriv_spline_u_float_c_colorarray.tif -o DxOut deriv_spline_u_float_c_colorarrayDx.tif -o DyOut deriv_spline_u_float_c_colorarrayDy.tif test_deriv_spline_u_float_c_colorarray")
outputs.append ("deriv_spline_u_float_v_colorarray.tif")
outputs.append ("deriv_spline_u_float_v_colorarrayDx.tif")
outputs.append ("deriv_spline_u_float_v_colorarrayDy.tif")
outputs.append ("deriv_spline_u_float_u_colorarray.tif")
outputs.append ("deriv_spline_u_float_u_colorarrayDx.tif")
outputs.append ("deriv_spline_u_float_u_colorarrayDy.tif")
outputs.append ("deriv_spline_u_float_c_colorarray.tif")
outputs.append ("deriv_spline_u_float_c_colorarrayDx.tif")
outputs.append ("deriv_spline_u_float_c_colorarrayDy.tif")

command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValOut deriv_spline_v_float_v_colorarray.tif -o DxOut deriv_spline_v_float_v_colorarrayDx.tif -o DyOut deriv_spline_v_float_v_colorarrayDy.tif test_deriv_spline_v_float_v_colorarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValOut deriv_spline_v_float_u_colorarray.tif -o DxOut deriv_spline_v_float_u_colorarrayDx.tif -o DyOut deriv_spline_v_float_u_colorarrayDy.tif test_deriv_spline_v_float_u_colorarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValOut deriv_spline_v_float_c_colorarray.tif -o DxOut deriv_spline_v_float_c_colorarrayDx.tif -o DyOut deriv_spline_v_float_c_colorarrayDy.tif test_deriv_spline_v_float_c_colorarray")
outputs.append ("deriv_spline_v_float_v_colorarray.tif")
outputs.append ("deriv_spline_v_float_v_colorarrayDx.tif")
outputs.append ("deriv_spline_v_float_v_colorarrayDy.tif")
outputs.append ("deriv_spline_v_float_u_colorarray.tif")
outputs.append ("deriv_spline_v_float_u_colorarrayDx.tif")
outputs.append ("deriv_spline_v_float_u_colorarrayDy.tif")
outputs.append ("deriv_spline_v_float_c_colorarray.tif")
outputs.append ("deriv_spline_v_float_c_colorarrayDx.tif")
outputs.append ("deriv_spline_v_float_c_colorarrayDy.tif")

command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValOut deriv_spline_vNoDeriv_float_v_colorarray.tif -o DxOut deriv_spline_vNoDeriv_float_v_colorarrayDx.tif -o DyOut deriv_spline_vNoDeriv_float_v_colorarrayDy.tif test_deriv_spline_vNoDeriv_float_v_colorarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValOut deriv_spline_vNoDeriv_float_u_colorarray.tif -o DxOut deriv_spline_vNoDeriv_float_u_colorarrayDx.tif -o DyOut deriv_spline_vNoDeriv_float_u_colorarrayDy.tif test_deriv_spline_vNoDeriv_float_u_colorarray")
command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValOut deriv_spline_vNoDeriv_float_c_colorarray.tif -o DxOut deriv_spline_vNoDeriv_float_c_colorarrayDx.tif -o DyOut deriv_spline_vNoDeriv_float_c_colorarrayDy.tif test_deriv_spline_vNoDeriv_float_c_colorarray")
outputs.append ("deriv_spline_vNoDeriv_float_v_colorarray.tif")
outputs.append ("deriv_spline_vNoDeriv_float_v_colorarrayDx.tif")
outputs.append ("deriv_spline_vNoDeriv_float_v_colorarrayDy.tif")
outputs.append ("deriv_spline_vNoDeriv_float_u_colorarray.tif")
outputs.append ("deriv_spline_vNoDeriv_float_u_colorarrayDx.tif")
outputs.append ("deriv_spline_vNoDeriv_float_u_colorarrayDy.tif")
outputs.append ("deriv_spline_vNoDeriv_float_c_colorarray.tif")
outputs.append ("deriv_spline_vNoDeriv_float_c_colorarrayDx.tif")
outputs.append ("deriv_spline_vNoDeriv_float_c_colorarrayDy.tif")


command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValOut deriv_spline_v_float_vNoDeriv_colorarray.tif -o DxOut deriv_spline_v_float_vNoDeriv_colorarrayDx.tif -o DyOut deriv_spline_v_float_vNoDeriv_colorarrayDy.tif test_deriv_spline_v_float_vNoDeriv_colorarray")
outputs.append ("deriv_spline_v_float_vNoDeriv_colorarray.tif")
outputs.append ("deriv_spline_v_float_vNoDeriv_colorarrayDx.tif")
outputs.append ("deriv_spline_v_float_vNoDeriv_colorarrayDy.tif")

command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValOut deriv_spline_u_float_vNoDeriv_colorarray.tif -o DxOut deriv_spline_u_float_vNoDeriv_colorarrayDx.tif -o DyOut deriv_spline_u_float_vNoDeriv_colorarrayDy.tif test_deriv_spline_u_float_vNoDeriv_colorarray")
outputs.append ("deriv_spline_u_float_vNoDeriv_colorarray.tif")
outputs.append ("deriv_spline_u_float_vNoDeriv_colorarrayDx.tif")
outputs.append ("deriv_spline_u_float_vNoDeriv_colorarrayDy.tif")

command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o ValOut deriv_spline_c_float_vNoDeriv_colorarray.tif -o DxOut deriv_spline_c_float_vNoDeriv_colorarrayDx.tif -o DyOut deriv_spline_c_float_vNoDeriv_colorarrayDy.tif test_deriv_spline_c_float_vNoDeriv_colorarray")
outputs.append ("deriv_spline_c_float_vNoDeriv_colorarray.tif")
outputs.append ("deriv_spline_c_float_vNoDeriv_colorarrayDx.tif")
outputs.append ("deriv_spline_c_float_vNoDeriv_colorarrayDy.tif")

# expect a few LSB failures
failthresh = 0.008
failpercent = 3

