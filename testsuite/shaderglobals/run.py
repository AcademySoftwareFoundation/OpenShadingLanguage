#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


command = testshade("-g 64 64 --vary_pdxdy --vary_udxdy --vary_vdxdy -od uint8 -o out_P out_P.tif -o out_dPdx out_dPdx.tif -o out_dPdy out_dPdy.tif -o out_dPdy out_dPdz.tif -o out_I out_I.tif -o out_dIdx out_dIdx.tif -o out_dIdy out_dIdy.tif -o out_N out_N.tif -o out_Ng out_Ng.tif -o out_u out_u.tif -o out_dudx out_dudx.tif -o out_dudy out_dudy.tif -o out_v out_v.tif -o out_dvdx out_dvdx.tif -o out_dvdy out_dvdy.tif -o out_dPdu out_dPdu.tif -o out_dPdv out_dPdv.tif -o out_time out_time.tif -o out_dtime out_dtime.tif -o out_dPdtime out_dPdtime.tif -o out_Ps out_Ps.tif -o out_dPsdx out_dPsdx.tif -o out_dPsdy out_dPsdy.tif -o out_backfacing out_backfacing.tif -o out_surfacearea out_surfacearea.tif -o out_object2common_of_P out_object2common_of_P.tif -o out_shader2common_of_P out_shader2common_of_P.tif -o out_calculatenormal_fliphandedness out_calculatenormal_fliphandedness.tif -o out_rt_camera out_rt_camera.tif -o out_rt_shadow out_rt_shadow.tif -o out_rt_diffuse out_rt_diffuse.tif -o out_rt_glossy out_rt_glossy.tif -o out_rt_reflection out_rt_reflection.tif -o out_rt_refraction out_rt_refraction.tif test")
outputs = [ 
    "out.txt", 
    "out_dPdx.tif", 
    "out_dPdy.tif", 
    "out_dPdz.tif", 
    "out_I.tif", 
    "out_dIdx.tif", 
    "out_dIdy.tif", 
    "out_N.tif", 
    "out_Ng.tif", 
    "out_u.tif", 
    "out_dudx.tif", 
    "out_dudy.tif", 
    "out_v.tif", 
    "out_dvdx.tif", 
    "out_dvdy.tif", 
    "out_dPdu.tif", 
    "out_dPdv.tif", 
    "out_time.tif", 
    "out_dtime.tif", 
    "out_dPdtime.tif", 
    "out_Ps.tif", 
    "out_dPsdx.tif", 
    "out_dPsdy.tif",
    "out_backfacing.tif",
    "out_surfacearea.tif",
    "out_object2common_of_P.tif",
    "out_shader2common_of_P.tif",
    "out_calculatenormal_fliphandedness.tif",
    "out_rt_camera.tif",
    "out_rt_shadow.tif",
    "out_rt_diffuse.tif",
    "out_rt_glossy.tif",
    "out_rt_reflection.tif",
    "out_rt_refraction.tif"
]


# expect a few LSB failures
failthresh = 0.008
failpercent = 3

