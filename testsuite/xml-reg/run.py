#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# because the regression tests get run twice, we don't want to recreate
# these texture files because their DateTime attributes would be different
# between the BASELINE and REGRESSION runs possibly causing different
# results.  To avoid this we use the --no-clobber option to leave the 
# texture files created by BASELINE undisturbed
command += oiiotool ("-pattern fill:topleft=0,0,0:topright=1,0,0:bottomleft=0,1,0:bottomright=1,1,1 256x128 3 -d uint8 -oenv ramp.env")

def run_test (suffix) :
    global command
    command += testshade("-t 1 -g 256 256 --center -od uint8 "
                         "-o out_nomatchId out_nomatchId_"+suffix+".tif "
                         "-o out_camerapackId out_camerapackId_"+suffix+".tif "
                         "-o out_imageId out_imageId_"+suffix+".tif "
                         "-o out_cameraId out_cameraId_"+suffix+".tif "
                         "-o out_nocameraId out_nocameraId_"+suffix+".tif "
                         "-o out_foundName out_foundName_"+suffix+".tif "
                         "-o out_name out_name_"+suffix+".tif "
                         "-o out_found2sides out_found2sides_"+suffix+".tif "
                         "-o out_2sides out_2sides_"+suffix+".tif "
                         "-o out_xformId out_xformId_"+suffix+".tif "
                         "-o out_foundMat out_foundMat_"+suffix+".tif "
                         "-o out_mat out_mat_"+suffix+".tif "
                         "-o out_foundChannel out_foundChannel_"+suffix+".tif "
                         "-o out_channel out_channel_"+suffix+".tif "
                         "-o out_foundFilter out_foundFilter_"+suffix+".tif "
                         "-o out_filter out_filter_"+suffix+".tif "
                         "test_xml_"+suffix)
    global outputs     
    outputs.append ("out_nomatchId_"+suffix+".tif")
    outputs.append ("out_camerapackId_"+suffix+".tif")
    outputs.append ("out_imageId_"+suffix+".tif")
    outputs.append ("out_cameraId_"+suffix+".tif")
    outputs.append ("out_nocameraId_"+suffix+".tif")
    outputs.append ("out_foundName_"+suffix+".tif")
    outputs.append ("out_name_"+suffix+".tif")
    outputs.append ("out_found2sides_"+suffix+".tif")
    outputs.append ("out_2sides_"+suffix+".tif")
    outputs.append ("out_xformId_"+suffix+".tif")
    outputs.append ("out_foundMat_"+suffix+".tif")
    outputs.append ("out_mat_"+suffix+".tif")
    outputs.append ("out_foundChannel_"+suffix+".tif")
    outputs.append ("out_channel_"+suffix+".tif")
    outputs.append ("out_foundFilter_"+suffix+".tif")
    outputs.append ("out_filter_"+suffix+".tif")
    
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

