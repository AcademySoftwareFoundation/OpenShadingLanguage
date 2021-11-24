#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# because the regression tests get run twice, we don't want to recreate
# these texture files because their DateTime attributes would be different
# between the BASELINE and REGRESSION runs possibly causing different
# results.  To avoid this we use the --no-clobber option to leave the 
# texture files created by BASELINE undisturbed
command += oiiotool("--pattern fill:topleft=0.125,0.25,0.5,0:topright=0.125,0.25,0.5,0.5:bottomleft=0.125,0.25,0.5,0.25:bottomright=0.125,0.25,0.5,0.75 64x64 4 -d half -o alpharamp.exr")

def run_test (suffix) :
    global command
    command += testshade("-t 1 -g 256 256 --vary_pdxdy --vary_udxdy --vary_vdxdy --center -od uint8 "
                         "-o out_alpha out_alpha_"+suffix+".tif "
                         "-o out_alpha_derivs out_alpha_derivs_"+suffix+".tif "
                         "-o out_blur out_blur_"+suffix+".tif "
                         "-o out_color out_color_"+suffix+".tif "
                         "-o out_dx out_dx_"+suffix+".tif "
                         "-o out_dy out_dy_"+suffix+".tif "
                         "-o out_errormsg out_errormsg_"+suffix+".tif "
                         "-o out_firstchannel out_firstchannel_"+suffix+".tif "
                         "-o out_missingalpha out_missingalpha_"+suffix+".tif "
                         "-o out_missing_color out_missing_color_"+suffix+".tif "
                         "-o out_simple out_simple_"+suffix+".tif "
                         "-o out_smallderivs out_smallderivs_"+suffix+".tif "
                         "-o out_time out_time_"+suffix+".tif "
                         "-o out_width out_width_"+suffix+".tif "
                         "-o out_widthderivs out_widthderivs_"+suffix+".tif "
                         "-o out_wrap out_wrap_"+suffix+".tif "
                         "test_texture3d_opts_"+suffix)
    global outputs     
    outputs.append ("out_alpha_"+suffix+".tif")
    outputs.append ("out_alpha_derivs_"+suffix+".tif")
    outputs.append ("out_blur_"+suffix+".tif")
    outputs.append ("out_color_"+suffix+".tif")
    outputs.append ("out_dx_"+suffix+".tif")
    outputs.append ("out_dy_"+suffix+".tif")
    outputs.append ("out_errormsg_"+suffix+".tif")
    outputs.append ("out_firstchannel_"+suffix+".tif")
    outputs.append ("out_missingalpha_"+suffix+".tif")
    outputs.append ("out_missing_color_"+suffix+".tif")
    outputs.append ("out_simple_"+suffix+".tif")
    outputs.append ("out_smallderivs_"+suffix+".tif")
    outputs.append ("out_time_"+suffix+".tif")
    outputs.append ("out_width_"+suffix+".tif")
    outputs.append ("out_widthderivs_"+suffix+".tif")
    outputs.append ("out_wrap_"+suffix+".tif")
    
    return
    
run_test ("c_c")
run_test ("c_u")
run_test ("c_v")

run_test ("v_c")
run_test ("v_u")
run_test ("v_v")

# expect a few LSB failures
failthresh = 0.008
failpercent = 3

