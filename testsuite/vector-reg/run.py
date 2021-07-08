#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# distance with 2 points
command += testshade("--vary_udxdy --vary_vdxdy -g 32 32 distance_v_vector_u_vector "\
                     "-od uint8 -o ddistance ddistance_WdfWdvWv.tif "\
                     "-o dxdistance dxdistance_WdfWdvWv.tif -o dydistance dydistance_WdfWdvWv.tif "\
                     "-o mddistance mddistance_WdfWdvWv.tif "\
                     "-o mdxdistance mdxdistance_WdfWdvWv.tif -o mdydistance mdydistance_WdfWdvWv.tif")
outputs.append ("ddistance_WdfWdvWv.tif")
outputs.append ("dxdistance_WdfWdvWv.tif")
outputs.append ("dydistance_WdfWdvWv.tif")
outputs.append ("mddistance_WdfWdvWv.tif")
outputs.append ("mdxdistance_WdfWdvWv.tif")
outputs.append ("mdydistance_WdfWdvWv.tif")

command += testshade("--vary_udxdy --vary_vdxdy -g 32 32 distance_u_vector_v_vector "\
                     "-od uint8 -o ddistance ddistance_WdfWvWdv.tif "\
                     "-o dxdistance dxdistance_WdfWvWdv.tif -o dydistance dydistance_WdfWvWdv.tif "\
                     "-o mddistance mddistance_WdfWvWdv.tif "\
                     "-o mdxdistance mdxdistance_WdfWvWdv.tif -o mdydistance mdydistance_WdfWvWdv.tif")
outputs.append ("ddistance_WdfWvWdv.tif")
outputs.append ("dxdistance_WdfWvWdv.tif")
outputs.append ("dydistance_WdfWvWdv.tif")
outputs.append ("mddistance_WdfWvWdv.tif")
outputs.append ("mdxdistance_WdfWvWdv.tif")
outputs.append ("mdydistance_WdfWvWdv.tif")


command += testshade("--vary_udxdy --vary_vdxdy -g 32 32 distance_v_vector_v_vector "\
                     "-od uint8 -o ddistance ddistance_WdfWdvWdv.tif "\
                     "-o dxdistance dxdistance_WdfWdvWdv.tif -o dydistance dydistance_WdfWdvWdv.tif "\
                     "-o mddistance mddistance_WdfWdvWdv.tif "\
                     "-o mdxdistance mdxdistance_WdfWdvWdv.tif -o mdydistance mdydistance_WdfWdvWdv.tif") 
outputs.append ("ddistance_WdfWdvWdv.tif")
outputs.append ("dxdistance_WdfWdvWdv.tif")
outputs.append ("dydistance_WdfWdvWdv.tif")
outputs.append ("mddistance_WdfWdvWdv.tif")
outputs.append ("mdxdistance_WdfWdvWdv.tif")
outputs.append ("mdydistance_WdfWdvWdv.tif")

command += testshade("--vary_udxdy --vary_vdxdy -g 32 32 distance_u_vector_u_vector "\
                     "-od uint8 -o ddistance ddistance_WfWvWv.tif -o mddistance mddistance_WfWvWv.tif") 
outputs.append ("ddistance_WfWvWv.tif")
outputs.append ("mddistance_WfWvWv.tif")



# distance with 3 points
command += testshade("--vary_udxdy --vary_vdxdy -g 32 32 distance_v_vector_v_vector_v_vector "\
                     "-od uint8 -o ddistance ddistance_vvv.tif "\
                     "-o dxdistance dxdistance_vvv.tif -o dydistance dydistance_vvv.tif "\
                     "-o mddistance mddistance_vvv.tif "\
                     "-o mdxdistance mdxdistance_vvv.tif -o mdydistance mdydistance_vvv.tif") 
outputs.append ("ddistance_vvv.tif")
outputs.append ("dxdistance_vvv.tif")
outputs.append ("dydistance_vvv.tif")
outputs.append ("mddistance_vvv.tif")
outputs.append ("mdxdistance_vvv.tif")
outputs.append ("mdydistance_vvv.tif")

command += testshade("--vary_udxdy --vary_vdxdy -g 32 32 distance_v_vector_v_vector_u_vector "\
                     "-od uint8 -o ddistance ddistance_vvu.tif "\
                     "-o dxdistance dxdistance_vvu.tif -o dydistance dydistance_vvu.tif "\
                     "-o mddistance mddistance_vvu.tif "\
                     "-o mdxdistance mdxdistance_vvu.tif -o mdydistance mdydistance_vvu.tif") 
outputs.append ("ddistance_vvu.tif")
outputs.append ("dxdistance_vvu.tif")
outputs.append ("dydistance_vvu.tif")
outputs.append ("mddistance_vvu.tif")
outputs.append ("mdxdistance_vvu.tif")
outputs.append ("mdydistance_vvu.tif")

command += testshade("--vary_udxdy --vary_vdxdy -g 32 32 distance_v_vector_u_vector_v_vector "\
                     "-od uint8 -o ddistance ddistance_vuv.tif "\
                     "-o dxdistance dxdistance_vuv.tif -o dydistance dydistance_vuv.tif "\
                     "-o mddistance mddistance_vuv.tif "\
                     "-o mdxdistance mdxdistance_vuv.tif -o mdydistance mdydistance_vuv.tif") 
outputs.append ("ddistance_vuv.tif")
outputs.append ("dxdistance_vuv.tif")
outputs.append ("dydistance_vuv.tif")
outputs.append ("mddistance_vuv.tif")
outputs.append ("mdxdistance_vuv.tif")
outputs.append ("mdydistance_vuv.tif")

command += testshade("--vary_udxdy --vary_vdxdy -g 32 32 distance_u_vector_u_vector_u_vector "\
                     "-od uint8 -o ddistance ddistance_uuu.tif "\
                     "-o dxdistance dxdistance_uuu.tif -o dydistance dydistance_uuu.tif "\
                     "-o mddistance mddistance_uuu.tif "\
                     "-o mdxdistance mdxdistance_uuu.tif -o mdydistance mdydistance_uuu.tif") 
outputs.append ("ddistance_uuu.tif")
outputs.append ("dxdistance_uuu.tif")
outputs.append ("dydistance_uuu.tif")
outputs.append ("mddistance_uuu.tif")
outputs.append ("mdxdistance_uuu.tif")
outputs.append ("mdydistance_uuu.tif")

command += testshade("--vary_udxdy --vary_vdxdy -g 32 32 distance_u_vector_v_vector_u_vector "\
                     "-od uint8 -o ddistance ddistance_uvu.tif "\
                     "-o dxdistance dxdistance_uvu.tif -o dydistance dydistance_uvu.tif "\
                     "-o mddistance mddistance_uvu.tif "\
                     "-o mdxdistance mdxdistance_uvu.tif -o mdydistance mdydistance_uvu.tif") 
outputs.append ("ddistance_uvu.tif")
outputs.append ("dxdistance_uvu.tif")
outputs.append ("dydistance_uvu.tif")
outputs.append ("mddistance_uvu.tif")
outputs.append ("mdxdistance_uvu.tif")
outputs.append ("mdydistance_uvu.tif")

command += testshade("--vary_udxdy --vary_vdxdy -g 32 32 distance_u_vector_u_vector_v_vector "\
                     "-od uint8 -o ddistance ddistance_uuv.tif "\
                     "-o dxdistance dxdistance_uuv.tif -o dydistance dydistance_uuv.tif "\
                     "-o mddistance mddistance_uuv.tif "\
                     "-o mdxdistance mdxdistance_uuv.tif -o mdydistance mdydistance_uuv.tif") 
outputs.append ("ddistance_uuv.tif")
outputs.append ("dxdistance_uuv.tif")
outputs.append ("dydistance_uuv.tif")
outputs.append ("mddistance_uuv.tif")
outputs.append ("mdxdistance_uuv.tif")
outputs.append ("mdydistance_uuv.tif")





command += testshade("--vary_udxdy --vary_vdxdy --g 32 32 test_cross_v_dvector_u_vector "\
                     "-od uint8 -o dcross dcross_v_dvector_u_vector.tif "\
                     "-o dxcross dxcross_v_dvector_u_vector.tif -o dycross dycross_v_dvector_u_vector.tif "\
                     "-o mdcross mdcross_v_dvector_u_vector.tif "\
                     "-o mdxcross mdxcross_v_dvector_u_vector.tif -o mdycross mdycross_v_dvector_u_vector.tif") 
outputs.append ("dcross_v_dvector_u_vector.tif")
outputs.append ("dxcross_v_dvector_u_vector.tif")
outputs.append ("dycross_v_dvector_u_vector.tif")
outputs.append ("mdcross_v_dvector_u_vector.tif")
outputs.append ("mdxcross_v_dvector_u_vector.tif")
outputs.append ("mdycross_v_dvector_u_vector.tif")

command += testshade("--vary_udxdy --vary_vdxdy --g 32 32 test_cross_u_vector_v_dvector "\
                     "-od uint8 -o dcross dcross_u_vector_v_dvector.tif "\
                     "-o dxcross dxcross_u_vector_v_dvector.tif -o dycross dycross_u_vector_v_dvector.tif "\
                     "-o mdcross mdcross_u_vector_v_dvector.tif "\
                     "-o mdxcross mdxcross_u_vector_v_dvector.tif -o mdycross mdycross_u_vector_v_dvector.tif")
outputs.append ("dcross_u_vector_v_dvector.tif")
outputs.append ("dxcross_u_vector_v_dvector.tif")
outputs.append ("dycross_u_vector_v_dvector.tif")
outputs.append ("mdcross_u_vector_v_dvector.tif")
outputs.append ("mdxcross_u_vector_v_dvector.tif")
outputs.append ("mdycross_u_vector_v_dvector.tif")

command += testshade("--vary_udxdy --vary_vdxdy --g 32 32 test_cross_v_dvector_v_dvector "\
                     "-od uint8 -o dcross dcross_v_dvector_v_dvector.tif "\
                     "-o dxcross dxcross_v_dvector_v_dvector.tif -o dycross dycross_v_dvector_v_dvector.tif "\
                     "-o mdcross mdcross_v_dvector_v_dvector.tif "\
                     "-o mdxcross mdxcross_v_dvector_v_dvector.tif -o mdycross mdycross_v_dvector_v_dvector.tif")
outputs.append ("dcross_v_dvector_v_dvector.tif")
outputs.append ("dxcross_v_dvector_v_dvector.tif")
outputs.append ("dycross_v_dvector_v_dvector.tif")
outputs.append ("mdcross_v_dvector_v_dvector.tif")
outputs.append ("mdxcross_v_dvector_v_dvector.tif")
outputs.append ("mdycross_v_dvector_v_dvector.tif")

command += testshade("--vary_udxdy --vary_vdxdy --g 32 32 test_cross_v_vector_v_vector "\
                     "-od uint8 -o dcross dcross_v_vector_v_vector.tif -o mdcross mdcross_v_vector_v_vector.tif")
outputs.append ("dcross_v_vector_v_vector.tif")
outputs.append ("mdcross_v_vector_v_vector.tif")


command += testshade("--vary_udxdy --vary_vdxdy -g 32 32 dot_v_vector_u_vector "\
                     "-od uint8 -o ddot ddot_v_vector_u_vector.tif "\
                     "-o dxdot dxdot_v_vector_u_vector.tif -o dydot dydot_v_vector_u_vector.tif "\
                     "-o mddot mddot_v_vector_u_vector.tif "\
                     "-o mdxdot mdxdot_v_vector_u_vector.tif -o mdydot mdydot_v_vector_u_vector.tif")
outputs.append ("ddot_v_vector_u_vector.tif")
outputs.append ("dxdot_v_vector_u_vector.tif")
outputs.append ("dydot_v_vector_u_vector.tif")
outputs.append ("mddot_v_vector_u_vector.tif")
outputs.append ("mdxdot_v_vector_u_vector.tif")
outputs.append ("mdydot_v_vector_u_vector.tif")

command += testshade("--vary_udxdy --vary_vdxdy -g 32 32 dot_u_vector_v_vector "\
                     "-od uint8 -o ddot ddot_u_vector_v_vector.tif "\
                     "-o dxdot dxdot_u_vector_v_vector.tif -o dydot dydot_u_vector_v_vector.tif "\
                     "-o mddot mddot_u_vector_v_vector.tif "\
                     "-o mdxdot mdxdot_u_vector_v_vector.tif -o mdydot mdydot_u_vector_v_vector.tif")
outputs.append ("ddot_u_vector_v_vector.tif")
outputs.append ("dxdot_u_vector_v_vector.tif")
outputs.append ("dydot_u_vector_v_vector.tif")
outputs.append ("mddot_u_vector_v_vector.tif")
outputs.append ("mdxdot_u_vector_v_vector.tif")
outputs.append ("mdydot_u_vector_v_vector.tif")

command += testshade("--vary_udxdy --vary_vdxdy -g 32 32 dot_v_vector_v_vector -od uint8 -o ddot ddot_v_vector_v_vector.tif "\
                     "-o dxdot dxdot_v_vector_v_vector.tif -o dydot dydot_v_vector_v_vector.tif "\
                     "-o mddot mddot_v_vector_v_vector.tif "\
                     "-o mdxdot mdxdot_v_vector_v_vector.tif -o mdydot mdydot_v_vector_v_vector.tif") 
outputs.append ("ddot_v_vector_v_vector.tif")
outputs.append ("dxdot_v_vector_v_vector.tif")
outputs.append ("dydot_v_vector_v_vector.tif")
outputs.append ("mddot_v_vector_v_vector.tif")
outputs.append ("mdxdot_v_vector_v_vector.tif")
outputs.append ("mdydot_v_vector_v_vector.tif")

command += testshade("--vary_udxdy --vary_vdxdy -g 32 32 dot_u_vector_u_vector "\
                     "-od uint8 -o ddot ddot_u_vector_u_vector.tif -o mddot mddot_u_vector_u_vector.tif") 
outputs.append ("ddot_u_vector_u_vector.tif")
outputs.append ("mddot_u_vector_u_vector.tif")




# expect a few LSB failures
failthresh = 0.008
failpercent = 3

