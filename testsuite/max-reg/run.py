#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# max(float,float) includes masking
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_max_u_float_u_float.tif test_max_u_float_u_float")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_max_u_float_v_float.tif test_max_u_float_v_float")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_max_v_float_u_float.tif test_max_v_float_u_float")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_max_v_float_v_float.tif test_max_v_float_v_float")
# Derivs includes masking
command += testshade("--vary_udxdy --vary_udxdy -t 1 -g 64 64 -od uint8 -o Cout out_max_v_dfloat_v_dfloat.tif test_max_v_dfloat_v_dfloat")

# max(int, int) (includes masking)
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_max_u_int_u_int.tif test_max_u_int_u_int")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_max_u_int_v_int.tif test_max_u_int_v_int")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_max_v_int_u_int.tif test_max_v_int_u_int")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_max_v_int_v_int.tif test_max_v_int_v_int")

# max(vec, vec) (including Masking)
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_max_u_vec_u_vec.tif test_max_u_vec_u_vec")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_max_u_vec_v_vec.tif test_max_u_vec_v_vec")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_max_v_vec_v_vec.tif test_max_v_vec_v_vec")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_max_v_vec_u_vec.tif test_max_v_vec_u_vec")
# Derivs includes masking
command += testshade("--vary_udxdy --vary_udxdy -t 1 -g 64 64 -od uint8 -o Cout out_max_v_dvec_v_dvec.tif test_max_v_dvec_v_dvec")

outputs = [ 
    "out_max_u_float_u_float.tif",
    "out_max_u_float_v_float.tif",
    "out_max_v_float_u_float.tif",
    "out_max_v_float_v_float.tif",
    "out_max_v_dfloat_v_dfloat.tif",
    "out_max_u_int_u_int.tif",
    "out_max_u_int_v_int.tif",
    "out_max_v_int_u_int.tif",
    "out_max_v_int_v_int.tif",
    "out_max_u_vec_u_vec.tif",
    "out_max_u_vec_v_vec.tif",
    "out_max_v_vec_v_vec.tif",
    "out_max_v_vec_u_vec.tif",
    "out_max_v_dvec_v_dvec.tif"
]


# expect a few LSB failures
failthresh = 0.008
failpercent = 3








