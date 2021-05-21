#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# We will take the liberty of assuming point, vector, and normal all take 
# the identical code path

# compassign u index u float includes masking
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_compassign_u_index_u_float.tif test_compassign_u_index_u_float")

# compassign u index v float includes masking
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_compassign_u_index_v_float.tif test_compassign_u_index_v_float")

# compassign u index v dual float includes masking
command += testshade("-t 1 -g 64 64 --vary_pdxdy -od uint8 -o Cout out_compassign_u_index_v_dfloat.tif test_compassign_u_index_v_dfloat")

# compassign v index u float includes masking
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_compassign_v_index_u_float.tif test_compassign_v_index_u_float")

# compassign v index v float includes masking
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_compassign_v_index_v_float.tif test_compassign_v_index_v_float")

# compassign v index v dual float includes masking
command += testshade("-t 1 -g 64 64 --vary_pdxdy -od uint8 -o Cout out_compassign_v_index_v_dfloat.tif test_compassign_v_index_v_dfloat")

outputs = [ 
    "out_compassign_u_index_u_float.tif",
    "out_compassign_u_index_v_float.tif",
    "out_compassign_u_index_v_dfloat.tif",
    "out_compassign_v_index_u_float.tif",
    "out_compassign_v_index_v_float.tif",
    "out_compassign_v_index_v_dfloat.tif",
]

# expect a few LSB failures
failthresh = 0.008
failpercent = 3











