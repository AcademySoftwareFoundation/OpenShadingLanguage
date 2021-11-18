#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# because the regression tests get run twice, we don't want to recreate
# these texture files because their DateTime attributes would be different
# between the BASELINE and REGRESSION runs possibly causing different
# results.  To avoid this we use the --no-clobber option to leave the 
# texture files created by BASELINE undisturbed
command += oiiotool ("--no-clobber -q -pattern constant:color=.5,.1,.1 128x128 3 -d uint8 -otex file.1001.tx")
command += oiiotool ("--no-clobber -q -pattern constant:color=.1,.5,.1 256x256 3 -d uint8 -otex file.1002.tx")
command += oiiotool ("--no-clobber -q -pattern constant:color=.1,.1,.5 512x512 3 -d uint8 -otex file.1011.tx")

command += oiiotool ("--no-clobber -q -pattern constant:color=.25,.35,.45 512x512 3 -d uint8 -otex fileB.1001.tx")
command += oiiotool ("--no-clobber -q -pattern constant:color=.95,.75,.65 256x256 3 -d uint8 -otex fileB.1002.tx")
command += oiiotool ("--no-clobber -q -pattern constant:color=.125,.35,.5 128x128 3 -d uint8 -otex fileB.1011.tx")

command += oiiotool ("--no-clobber -q -pattern constant:color=.45,.25,.85 256x256 3 -d uint8 -otex fileC.1001.tx")
command += oiiotool ("--no-clobber -q -pattern constant:color=.15,.55,.75 128x128 3 -d uint8 -otex fileC.1002.tx")
command += oiiotool ("--no-clobber -q -pattern constant:color=.75,.15,.35 512x512 3 -d uint8 -otex fileC.1011.tx")

command += oiiotool ("--no-clobber -q -pattern constant:color=.45,.33,.2 128x128 3 -d uint8 -otex fileD.1001.tx")
command += oiiotool ("--no-clobber -q -pattern constant:color=.33,.66,.8 512x512 3 -d uint8 -otex fileD.1002.tx")
command += oiiotool ("--no-clobber -q -pattern constant:color=.8,.6,.2 256x256 3 -d uint8 -otex fileD.1011.tx")

def run_test (suffix) :
    global command
    command += testshade("-t 1 -g 32 32 --center -scaleuv 2 2 -od uint8 "
                         "-o out_resolution out_resolution_"+suffix+".tif "
                         "-o out_channels out_channels_"+suffix+".tif "
                         "-o out_texturetype out_texturetype_"+suffix+".tif "
                         "-o out_textureformat out_textureformat_"+suffix+".tif "
                         "-o out_datawin out_datawin_"+suffix+".tif "
                         "-o out_dispwin out_dispwin_"+suffix+".tif "
                         "-o out_datetime out_datetime_"+suffix+".tif "
                         "-o out_avgcolor out_avgcolor_"+suffix+".tif "
                         "-o out_avgalpha out_avgalpha_"+suffix+".tif "
                         "-o out_constcolor out_constcolor_"+suffix+".tif "
                         "-o out_constalpha out_constalpha_"+suffix+".tif "
                         "-o out_unfoundinfo out_unfoundinfo_"+suffix+".tif "
                         "-o out_unfoundfile out_unfoundfile_"+suffix+".tif "
                         "-o out_skipcondition out_skipcondition_"+suffix+".tif "
                         "-o out_exists out_exists_"+suffix+".tif "
                         "-o out_not_exists out_not_exists_"+suffix+".tif "
                         "test_gettextureinfo_udim_"+suffix)
    global outputs     
    outputs.append ("out_resolution_"+suffix+".tif")
    outputs.append ("out_channels_"+suffix+".tif")
    outputs.append ("out_texturetype_"+suffix+".tif")
    outputs.append ("out_textureformat_"+suffix+".tif")
    outputs.append ("out_datawin_"+suffix+".tif")
    outputs.append ("out_dispwin_"+suffix+".tif")
    outputs.append ("out_datetime_"+suffix+".tif")
    outputs.append ("out_avgcolor_"+suffix+".tif")
    outputs.append ("out_avgalpha_"+suffix+".tif")
    outputs.append ("out_constcolor_"+suffix+".tif")
    outputs.append ("out_constalpha_"+suffix+".tif")
    outputs.append ("out_unfoundinfo_"+suffix+".tif")
    outputs.append ("out_unfoundfile_"+suffix+".tif")
    outputs.append ("out_skipcondition_"+suffix+".tif")
    outputs.append ("out_exists_"+suffix+".tif")
    outputs.append ("out_not_exists_"+suffix+".tif")
    
    return

run_test ("c_c_c_const_st")
run_test ("c_c_c_uniform_st")

run_test ("u_u_u_const_st")
run_test ("u_u_u_uniform_st")

run_test ("v_v_v_const_st")
run_test ("v_v_v_uniform_st")
    
run_test ("c_c_c")
run_test ("u_c_c")
run_test ("v_c_c")

run_test ("c_u_c")
run_test ("u_u_c")
run_test ("v_u_c")

run_test ("c_v_c")
run_test ("u_v_c")
run_test ("v_v_c")

run_test ("c_c_u")
run_test ("u_c_u")
run_test ("v_c_u")

run_test ("c_u_u")
run_test ("u_u_u")
run_test ("v_u_u")

run_test ("c_v_u")
run_test ("u_v_u")
run_test ("v_v_u")

run_test ("c_c_v")
run_test ("u_c_v")
run_test ("v_c_v")

run_test ("c_u_v")
run_test ("u_u_v")
run_test ("v_u_v")

run_test ("c_v_v")
run_test ("u_v_v")
run_test ("v_v_v")

# expect a few LSB failures
failthresh = 0.008
failpercent = 3

