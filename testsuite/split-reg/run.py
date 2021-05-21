#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#Uniform string, uniform sep, uniform maxsplit
command += testshade("-t 1 -g 64 64 test_split_u_str_u_sep_u_max -od uint8 -o res split_uuu_out.tif -o calres msplit_uuu_out.tif")

#Uniform string, uniform sep, varying maxsplit  
command += testshade("-t 1 -g 64 64 test_split_u_str_u_sep_v_max -od uint8 -o res split_uuv_out.tif -o calres msplit_uuv_out.tif")

#Uniform string, varying sep, uniform maxsplit 
command += testshade("-t 1 -g 64 64 test_split_u_str_v_sep_u_max -od uint8 -o res split_uvu_out.tif -o calres msplit_uvu_out.tif")

#Uniform string, varying sep, varying maxsplit 
command += testshade("-t 1 -g 64 64 test_split_u_str_v_sep_v_max -od uint8 -o res split_uvv_out.tif -o calres msplit_uvv_out.tif")

#Varying string, default sep, default maxsplit 
command += testshade("-t 1 -g 64 64 test_split_v_str_default_sep_default_max -od uint8 -o res split_vdd_out.tif -o calres msplit_vdd_out.tif")

#Varying string, uniform sep, uniform maxsplit 
command += testshade("-t 1 -g 64 64 test_split_v_str_u_sep_u_max -od uint8 -o res split_vuu_out.tif -o calres msplit_vuu_out.tif")

#Varying string, uniform sep, varying maxsplit 
command += testshade("-t 1 -g 64 64 test_split_v_str_u_sep_v_max -od uint8 -o res split_vuv_out.tif -o calres msplit_vuv_out.tif")

#varying string, varying sep, uniform maxsplit 
command += testshade("-t 1 -g 64 64 test_split_v_str_v_sep_u_max -od uint8 -o res split_vvu_out.tif -o calres msplit_vvu_out.tif")

#varying string, varying sep, varying maxsplit 
command += testshade("-t 1 -g 64 64 test_split_v_str_v_sep_v_max  -od uint8 -o res split_vvv_out.tif -o calres msplit_vvv_out.tif")

#varying string, varying sep, varying maxsplit 
command += testshade("-t 1 -g 64 64 test_split_v_str_v_sep_v_max_ura  -od uint8 -o res split_vvv_ura_out.tif -o calres msplit_vvv_ura_out.tif")

outputs = [ 
    "split_uuu_out.tif",
    "msplit_uuu_out.tif",
    "split_uuv_out.tif",
    "msplit_uuv_out.tif",
    "split_uvu_out.tif",
    "msplit_uvu_out.tif",
    "split_uvv_out.tif",
    "msplit_uvv_out.tif",
    "split_vdd_out.tif",
    "msplit_vdd_out.tif",
    "split_vuu_out.tif",
    "msplit_vuu_out.tif",
    "split_vuv_out.tif",
    "msplit_vuv_out.tif",
    "split_vvu_out.tif",
    "msplit_vvu_out.tif",
    "split_vvv_out.tif",
    "msplit_vvv_out.tif",
    "split_vvv_ura_out.tif",
    "msplit_vvv_ura_out.tif"
]


# expect a few LSB failures
failthresh = 0.008
failpercent = 3











