#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


# mix float, float, float includes masking
command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_u_float_u_float_u_float.tif test_mix_u_float_u_float_u_float")
outputs.append ("mix_u_float_u_float_u_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_u_float_u_float_v_float.tif test_mix_u_float_u_float_v_float")
outputs.append ("mix_u_float_u_float_v_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_v_float_u_float_v_float.tif test_mix_v_float_u_float_v_float")
outputs.append ("mix_v_float_u_float_v_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_u_float_v_float_v_float.tif test_mix_u_float_v_float_v_float")
outputs.append ("mix_u_float_v_float_v_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_u_float_v_float_u_float.tif test_mix_u_float_v_float_u_float")
outputs.append ("mix_u_float_v_float_u_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_v_float_u_float_u_float.tif test_mix_v_float_u_float_u_float")
outputs.append ("mix_v_float_u_float_u_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_v_float_v_float_u_float.tif test_mix_v_float_v_float_u_float")
outputs.append ("mix_v_float_v_float_u_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_v_float_v_float_v_float.tif test_mix_v_float_v_float_v_float")
outputs.append ("mix_v_float_v_float_v_float.tif")

command += testshade("--vary_udxdy --vary_udxdy -t 1 -g 32 32 -od uint8 -o Cout mix_v_dfloat_v_dfloat_v_dfloat.tif test_mix_v_dfloat_v_dfloat_v_dfloat")
outputs.append ("mix_v_dfloat_v_dfloat_v_dfloat.tif")

command += testshade("--vary_udxdy --vary_udxdy -t 1 -g 32 32 -od uint8 -o Cout mix_v_dfloat_v_dfloat_c_float.tif test_mix_v_dfloat_v_dfloat_c_float")
outputs.append ("mix_v_dfloat_v_dfloat_c_float.tif")



# mix vector, vector, float includes masking
command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_u_vector_u_vector_u_float.tif test_mix_u_vector_u_vector_u_float")
outputs.append ("mix_u_vector_u_vector_u_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_u_vector_u_vector_v_float.tif test_mix_u_vector_u_vector_v_float")
outputs.append ("mix_u_vector_u_vector_v_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_u_vector_v_vector_u_float.tif test_mix_u_vector_v_vector_u_float")
outputs.append ("mix_u_vector_v_vector_u_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_u_vector_v_vector_v_float.tif test_mix_u_vector_v_vector_v_float")
outputs.append ("mix_u_vector_v_vector_v_float.tif")

command += testshade("--vary_udxdy --vary_udxdy -t 1 -g 32 32 -od uint8 -o Cout mix_v_dvector_v_dvector_c_float.tif test_mix_v_dvector_v_dvector_c_float")
outputs.append ("mix_v_dvector_v_dvector_c_float.tif")

command += testshade("--vary_udxdy --vary_udxdy -t 1 -g 32 32 -od uint8 -o Cout mix_v_dvector_v_dvector_v_dfloat.tif test_mix_v_dvector_v_dvector_v_dfloat")
outputs.append ("mix_v_dvector_v_dvector_v_dfloat.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_v_vector_u_vector_u_float.tif test_mix_v_vector_u_vector_u_float")
outputs.append ("mix_v_vector_u_vector_u_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_v_vector_u_vector_v_float.tif test_mix_v_vector_u_vector_v_float")
outputs.append ("mix_v_vector_u_vector_v_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_v_vector_v_vector_u_float.tif test_mix_v_vector_v_vector_u_float")
outputs.append ("mix_v_vector_v_vector_u_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_v_vector_v_vector_v_float.tif test_mix_v_vector_v_vector_v_float")
outputs.append ("mix_v_vector_v_vector_v_float.tif")



# mix vector, vector, vector includes masking
command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_u_vector_u_vector_u_vector.tif test_mix_u_vector_u_vector_u_vector")
outputs.append ("mix_u_vector_u_vector_u_vector.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_u_vector_u_vector_v_vector.tif test_mix_u_vector_u_vector_v_vector")
outputs.append ("mix_u_vector_u_vector_v_vector.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_u_vector_v_vector_u_vector.tif test_mix_u_vector_v_vector_u_vector")
outputs.append ("mix_u_vector_v_vector_u_vector.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_u_vector_v_vector_v_vector.tif test_mix_u_vector_v_vector_v_vector")
outputs.append ("mix_u_vector_v_vector_v_vector.tif")

command += testshade("--vary_udxdy --vary_udxdy -t 1 -g 32 32 -od uint8 -o Cout mix_v_dvector_v_dvector_c_vector.tif test_mix_v_dvector_v_dvector_c_vector")
outputs.append ("mix_v_dvector_v_dvector_c_vector.tif")

command += testshade("--vary_udxdy --vary_udxdy -t 1 -g 32 32 -od uint8 -o Cout mix_v_dvector_v_dvector_v_dvector.tif test_mix_v_dvector_v_dvector_v_dvector")
outputs.append ("mix_v_dvector_v_dvector_v_dvector.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_v_vector_u_vector_u_vector.tif test_mix_v_vector_u_vector_u_vector")
outputs.append ("mix_v_vector_u_vector_u_vector.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_v_vector_u_vector_v_vector.tif test_mix_v_vector_u_vector_v_vector")
outputs.append ("mix_v_vector_u_vector_v_vector.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_v_vector_v_vector_u_vector.tif test_mix_v_vector_v_vector_u_vector")
outputs.append ("mix_v_vector_v_vector_u_vector.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout mix_v_vector_v_vector_v_vector.tif test_mix_v_vector_v_vector_v_vector")
outputs.append ("mix_v_vector_v_vector_v_vector.tif")



# expect a few LSB failures
failthresh = 0.008
failpercent = 3

