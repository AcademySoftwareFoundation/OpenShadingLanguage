#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


command += testshade("-t 1 -g 32 32 -od uint8 a_v_b_u -o cout vu.tif -o mcout vu_m.tif")
outputs.append ("vu.tif")
outputs.append ("vu_m.tif")

command += testshade("-t 1 -g 32 32 -od uint8 a_u_b_v -o cout uv.tif -o mcout uv_m.tif")
outputs.append ("uv.tif")
outputs.append ("uv_m.tif")

command += testshade("-t 1 -g 32 32 -od uint8 a_v_b_v -o cout vv.tif -o mcout vv_m.tif")
outputs.append ("vv.tif")
outputs.append ("vv_m.tif")

command += testshade("-t 1 -g 32 32 -od uint8 a_u_b_u -o cout uu.tif -o mcout uu_m.tif")
outputs.append ("uu.tif")
outputs.append ("uu_m.tif")



command += testshade("-t 1 -g 32 32 -od uint8 a_vconditional_b_v -o cout vconditionalv.tif -o mcout vconditionalv_m.tif")
outputs.append ("vconditionalv.tif")
outputs.append ("vconditionalv_m.tif")

command += testshade("-t 1 -g 32 32 -od uint8 a_vconditional_b_u -o cout vconditionalu.tif -o mcout vconditionalu_m.tif")
outputs.append ("vconditionalu.tif")
outputs.append ("vconditionalu_m.tif")

command += testshade("-t 1 -g 32 32 -od uint8 a_vconditional_b_vconditional -o cout vconditionalvconditional.tif -o mcout vconditionalvconditional_m.tif")
outputs.append ("vconditionalvconditional.tif")
outputs.append ("vconditionalvconditional_m.tif")

command += testshade("-t 1 -g 32 32 -od uint8 a_v_b_vconditional -o cout vvconditional.tif -o mcout vvconditional_m.tif")
outputs.append ("vvconditional.tif")
outputs.append ("vvconditional_m.tif")


# expect a few LSB failures
failthresh = 0.008
failpercent = 3

