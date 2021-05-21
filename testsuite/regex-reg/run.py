#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

######################
#Uniform result
##########################

#Uniform subject, uniform pattern#
command += testshade("-t 1 -g 64 64 -od uint8 u_subj_u_pattern -o cout uu_out.tif")

#Uniform subject, varying pattern#
command += testshade("-t 1 -g 64 64 -od uint8 u_subj_v_pattern -o cout uv_out.tif")

#Varying subject, uniform pattern#
command += testshade("-t 1 -g 64 64 -od uint8 v_subj_u_pattern -o cout vu_out.tif")

#Varying subject, varying pattern#
command += testshade("-t 1 -g 64 64 -od uint8 v_subj_v_pattern -o cout vv_out.tif")


##################
#Varying result
##################
#Uniform subject, uniform pattern#
command += testshade("-t 1 -g 64 64 -od uint8 u_subj_u_pattern_vr -o cout uu_vr_out.tif")

#Uniform subject, varying pattern#
command += testshade("-t 1 -g 64 64 -od uint8 u_subj_v_pattern_vr -o cout uv_vr_out.tif")

#Varying subject, uniform pattern#
command += testshade("-t 1 -g 64 64 -od uint8 v_subj_u_pattern_vr -o cout vu_vr_out.tif")

#Varying subject, varying pattern#
command += testshade("-t 1 -g 64 64 -od uint8 v_subj_v_pattern_vr -o cout vv_vr_out.tif")


##########################################
#Uniform result array
##########################################
command += testshade("-t 1 -g 64 64 -od uint8 u_subj_u_pattern_ura -o cout uu_ura_out.tif")
command += testshade("-t 1 -g 64 64 -od uint8 u_subj_v_pattern_ura -o cout uv_ura_out.tif")
command += testshade("-t 1 -g 64 64 -od uint8 v_subj_u_pattern_ura -o cout vu_ura_out.tif")
command += testshade("-t 1 -g 64 64 -od uint8 v_subj_v_pattern_ura -o cout vv_ura_out.tif")


##########################################
#Varying result array
##########################################
command += testshade("-t 1 -g 64 64 -od uint8 u_subj_u_pattern_vra -o cout uu_vra_out.tif")
command += testshade("-t 1 -g 64 64 -od uint8 u_subj_v_pattern_vra -o cout uv_vra_out.tif")
command += testshade("-t 1 -g 64 64 -od uint8 v_subj_u_pattern_vra -o cout vu_vra_out.tif")
command += testshade("-t 1 -g 64 64 -od uint8 v_subj_v_pattern_vra -o cout vv_vra_out.tif")


outputs = [ 
    "uu_out.tif",
    "uv_out.tif",
    "vu_out.tif",
    "vv_out.tif",
    "uu_vr_out.tif",
    "uv_vr_out.tif",
    "vu_vr_out.tif",
    "vv_vr_out.tif",
    "uu_ura_out.tif",
    "uv_ura_out.tif",
    "vu_ura_out.tif",
    "vv_ura_out.tif",
    "uu_vra_out.tif",
    "uv_vra_out.tif",
    "vu_vra_out.tif",
    "vv_vra_out.tif",
]

# expect a few LSB failures
failthresh = 0.008
failpercent = 3














