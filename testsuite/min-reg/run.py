#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# min(float,float) includes masking
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_min_u_float_u_float.tif test_min_u_float_u_float")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_min_u_float_v_float.tif test_min_u_float_v_float")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_min_v_float_u_float.tif test_min_v_float_u_float")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_min_v_float_v_float.tif test_min_v_float_v_float")
# Derivs includes masking
command += testshade("--vary_udxdy --vary_udxdy -t 1 -g 64 64 -od uint8 -o Cout out_min_v_dfloat_v_dfloat.tif test_min_v_dfloat_v_dfloat")

# min(int, int) (includes masking)
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_min_u_int_u_int.tif test_min_u_int_u_int")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_min_u_int_v_int.tif test_min_u_int_v_int")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_min_v_int_u_int.tif test_min_v_int_u_int")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_min_v_int_v_int.tif test_min_v_int_v_int")

# min(vec, vec) (including Masking)
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_min_u_vec_u_vec.tif test_min_u_vec_u_vec")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_min_u_vec_v_vec.tif test_min_u_vec_v_vec")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_min_v_vec_v_vec.tif test_min_v_vec_v_vec")
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_min_v_vec_u_vec.tif test_min_v_vec_u_vec")
# Derivs includes masking
command += testshade("--vary_udxdy --vary_udxdy -t 1 -g 64 64 -od uint8 -o Cout out_min_v_dvec_v_dvec.tif test_min_v_dvec_v_dvec")

outputs = [ 
    "out_min_u_float_u_float.tif",
    "out_min_u_float_v_float.tif",
    "out_min_v_float_u_float.tif",
    "out_min_v_float_v_float.tif",
    "out_min_v_dfloat_v_dfloat.tif",
    "out_min_u_int_u_int.tif",
    "out_min_u_int_v_int.tif",
    "out_min_v_int_u_int.tif",
    "out_min_v_int_v_int.tif",
    "out_min_u_vec_u_vec.tif",
    "out_min_u_vec_v_vec.tif",
    "out_min_v_vec_v_vec.tif",
    "out_min_v_vec_u_vec.tif",
    "out_min_v_dvec_v_dvec.tif",
]


# expect a few LSB failures
failthresh = 0.008
failpercent = 3











