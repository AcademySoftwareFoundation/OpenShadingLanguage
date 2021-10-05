#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


###############################
# Uniform float, Uniform float
###############################
command += testshade("-t 1 -g 32 32 -od uint8 a_float_u_b_float_u  -o cout ufuf.tif -o mcout ufuf_m.tif")
outputs.append ("ufuf.tif")
outputs.append ("ufuf_m.tif")

###############################
# Varying float, Varying float
###############################
command += testshade("-t 1 -g 32 32 -od uint8 a_float_v_b_float_v  -o cout vfvf.tif -o mcout vfvf_m.tif")
outputs.append ("vfvf.tif")
outputs.append ("vfvf_m.tif")

###############################
# Uniform float, Varying float
###############################
command += testshade("-t 1 -g 32 32 -od uint8 a_float_u_b_float_v  -o cout ufvf.tif -o mcout ufvf_m.tif")
outputs.append ("ufvf.tif")
outputs.append ("ufvf_m.tif")

###############################
# Varying float, Uniform float
###############################
command += testshade("-t 1 -g 32 32 -od uint8 a_float_v_b_float_u  -o cout vfuf.tif -o mcout vfuf_m.tif")
outputs.append ("vfuf.tif")
outputs.append ("vfuf_m.tif")



###############################
# Uniform float, uniform int
###############################
command += testshade("-t 1 -g 32 32 -od uint8 a_float_u_b_int_u  -o cout ufui.tif -o mcout ufui_m.tif")
outputs.append ("ufui.tif")
outputs.append ("ufui_m.tif")

###############################
# Varying float, Varying int
###############################
command += testshade("-t 1 -g 32 32 -od uint8 a_float_v_b_int_v  -o cout vfvi.tif -o mcout vfvi_m.tif")
outputs.append ("vfvi.tif")
outputs.append ("vfvi_m.tif")


###############################
# Uniform float, Varying int
###############################
command += testshade("-t 1 -g 32 32 -od uint8 a_float_u_b_int_v  -o cout ufvi.tif -o mcout ufvi_m.tif")
outputs.append ("ufvi.tif")
outputs.append ("ufvi_m.tif")

###############################
# Varying float, Uniform int
###############################
command += testshade("-t 1 -g 32 32 -od uint8 a_float_v_b_int_u  -o cout vfui.tif -o mcout vfui_m.tif")
outputs.append ("vfui.tif")
outputs.append ("vfui_m.tif")



###############################
# Uniform int, Uniform float
###############################
command += testshade("-t 1 -g 32 32 -od uint8 a_int_u_b_float_u  -o cout uiuf.tif -o mcout uiuf_m.tif")
outputs.append ("uiuf.tif")
outputs.append ("uiuf_m.tif")



###############################
# Varying int, Varying float
###############################
command += testshade("-t 1 -g 32 32 -od uint8 a_int_v_b_float_v  -o cout vivf.tif -o mcout vivf_m.tif")
outputs.append ("vivf.tif")
outputs.append ("vivf_m.tif")


###############################
# Uniform int, Varying float
###############################
command += testshade("-t 1 -g 32 32 -od uint8 a_int_u_b_float_v  -o cout uivf.tif -o mcout uivf_m.tif")
outputs.append ("uivf.tif")
outputs.append ("uivf_m.tif")


###############################
# Varying int, uniform float
###############################
command += testshade("-t 1 -g 32 32 -od uint8 a_int_v_b_float_u  -o cout viuf.tif -o mcout viuf_m.tif")
outputs.append ("viuf.tif")
outputs.append ("viuf_m.tif")



###############################
# Uniform int, Uniform int
###############################
command += testshade("-t 1 -g 32 32 -od uint8 a_int_u_b_int_u  -o cout uiui.tif -o mcout uiui_m.tif")
outputs.append ("uiui.tif")
outputs.append ("uiui_m.tif")



###############################
# Varying int, Varying int
###############################
command += testshade("-t 1 -g 32 32 -od uint8 a_int_v_b_int_v  -o cout vivi.tif -o mcout vivi_m.tif")
outputs.append ("vivi.tif")
outputs.append ("vivi_m.tif")


###############################
# Uniform int, Varying int
###############################
command += testshade("-t 1 -g 32 32 -od uint8 a_int_u_b_int_v  -o cout uivi.tif -o mcout uivi_m.tif")
outputs.append ("uivi.tif")
outputs.append ("uivi_m.tif")

###############################
# Varying int, Uniform int
###############################
command += testshade("-t 1 -g 32 32 -od uint8 a_int_v_b_int_u  -o cout viui.tif -o mcout viui_m.tif")
outputs.append ("viui.tif")
outputs.append ("viui_m.tif")

# expect a few LSB failures
failthresh = 0.008
failpercent = 3

