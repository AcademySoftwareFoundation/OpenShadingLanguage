#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

def run_varying_option_test (option) :
    global command
    command += testshade("-g 512 512 -center"+
                         " -layer src vary_"+option+" -layer dst test_gabor_options"+
                         " --connect src "+option+" dst "+option+ 
                         " -od uint8 -o Cout out_v_"+option+".tif")
    global outputs     
    outputs.append ("out_v_"+option+".tif")
    return

run_varying_option_test ("anisotropic")
run_varying_option_test ("bandwidth")
run_varying_option_test ("direction")
run_varying_option_test ("do_filter")
run_varying_option_test ("impulses")

def run_fixed_and_varying_option_test (fixedoption, fixedval, varyoption) :
    global command
    command += testshade("-g 512 512 -center"+
                         " -layer src vary_"+varyoption+
                         " --param "+fixedoption+" "+fixedval+
                         " -layer dst test_gabor_options"+
                         " --connect src "+varyoption+" dst "+varyoption+ 
                         " -od uint8 -o Cout out_"+fixedoption+"_"+fixedval+"_v_"+varyoption+".tif")
    global outputs     
    outputs.append ("out_"+fixedoption+"_"+fixedval+"_v_"+varyoption+".tif")
    return
    
run_fixed_and_varying_option_test ("do_filter", "0", "anisotropic")
run_fixed_and_varying_option_test ("anisotropic", "0", "direction")
run_fixed_and_varying_option_test ("anisotropic", "1", "direction")
run_fixed_and_varying_option_test ("anisotropic", "2", "direction")
run_fixed_and_varying_option_test ("bandwidth", "10", "impulses")

def run_varying_option2_test (option1, option2) :
    global command
    command += testshade("-g 512 512 -center"+
                         " -layer src1 vary_"+option1+
                         " -layer src2 vary_"+option2+
                         " -layer dst test_gabor_options"+
                         " --connect src1 "+option1+" dst "+option1+ 
                         " --connect src2 "+option2+" dst "+option2+ 
                         " -od uint8 -o Cout out_v_"+option1+"_v_"+option2+".tif")
    global outputs     
    outputs.append ("out_v_"+option1+"_v_"+option2+".tif")
    return

run_varying_option2_test ("anisotropic","bandwidth")
run_varying_option2_test ("bandwidth","direction")
run_varying_option2_test ("direction","do_filter")
run_varying_option2_test ("do_filter","impulses")

# vary all options
command += testshade("-g 512 512 -center"+
                     " -layer src1 vary_anisotropic"+
                     " -layer src2 vary_bandwidth"+
                     " -layer src3 vary_direction"+
                     " -layer src4 vary_do_filter"+
                     " -layer src5 vary_impulses"+
                     " -layer dst test_gabor_options"+
                     " --connect src1 anisotropic dst anisotropic"+ 
                     " --connect src2 bandwidth dst bandwidth"+ 
                     " --connect src3 direction dst direction"+ 
                     " --connect src4 do_filter dst do_filter"+ 
                     " --connect src5 impulses dst impulses"+ 
                     " -od uint8 -o Cout out_v_all_options.tif")
outputs.append ("out_v_all_options.tif")
    
# expect a few LSB failures
failthresh = 0.008
failpercent = 3

