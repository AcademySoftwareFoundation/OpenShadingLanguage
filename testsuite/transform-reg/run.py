#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


def run_2space_tests (triptype) :
    global command
    command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_transform_v_fromspace_v_tospace_u_"+triptype+".tif test_transform_v_fromspace_v_tospace_u_"+triptype+"")
    command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_transform_v_fromspace_v_tospace_v_"+triptype+".tif test_transform_v_fromspace_v_tospace_v_"+triptype+"")
    command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 32 32 -od uint8 -o Cout out_transform_v_fromspace_v_tospace_v_d"+triptype+".tif test_transform_v_fromspace_v_tospace_v_d"+triptype+"")
    global outputs     
    outputs.append ("out_transform_v_fromspace_v_tospace_u_"+triptype+".tif")
    
    outputs.append ("out_transform_v_fromspace_v_tospace_v_"+triptype+".tif")
    outputs.append ("out_transform_v_fromspace_v_tospace_v_d"+triptype+".tif")
    
    def run_fromspace_tospace_tests (fromspace, tospace) :
        global command
        command += testshade("-t 1 -g 32 32 -param fromspace " + fromspace + " -param tospace " + tospace + " -od uint8 -o Cout out_transform_" + fromspace + "_fromspace_" + tospace + "_tospace_u_"+triptype+".tif test_transform_u_fromspace_u_tospace_u_"+triptype+"")
        command += testshade("-t 1 -g 32 32 -param fromspace " + fromspace + " -param tospace " + tospace + " -od uint8 -o Cout out_transform_" + fromspace + "_fromspace_" + tospace + "_tospace_v_"+triptype+".tif test_transform_u_fromspace_u_tospace_v_"+triptype+"")
        command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 32 32 -param fromspace $1 -param tospace " + tospace + " -od uint8 -o Cout out_transform_" + fromspace + "_fromspace_" + tospace + "_tospace_v_d"+triptype+".tif test_transform_u_fromspace_u_tospace_v_d"+triptype+"")
    
        global outputs     
        outputs.append ("out_transform_" + fromspace + "_fromspace_" + tospace + "_tospace_u_"+triptype+".tif")
        outputs.append ("out_transform_" + fromspace + "_fromspace_" + tospace + "_tospace_v_"+triptype+".tif")
        outputs.append ("out_transform_" + fromspace + "_fromspace_" + tospace + "_tospace_v_d"+triptype+".tif")
        return
    
    def run_u_fromspace_tests (space) :
        run_fromspace_tospace_tests(space, "common")
        run_fromspace_tospace_tests(space, "object")
        run_fromspace_tospace_tests(space, "shader")
        run_fromspace_tospace_tests(space, "world")
        run_fromspace_tospace_tests(space, "camera")
    
        global command
        command += testshade("-t 1 -g 32 32 -param fromspace " + space + " -od uint8 -o Cout out_transform_" + space + "_fromspace_v_tospace_u_"+triptype+".tif test_transform_u_fromspace_v_tospace_u_"+triptype+"")
        command += testshade("-t 1 -g 32 32 -param fromspace " + space + " -od uint8 -o Cout out_transform_" + space + "_fromspace_v_tospace_v_"+triptype+".tif test_transform_u_fromspace_v_tospace_v_"+triptype+"")     
        command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 32 32 -param fromspace " + space + " -od uint8 -o Cout out_transform_" + space + "_fromspace_v_tospace_v_d"+triptype+".tif test_transform_u_fromspace_v_tospace_v_d"+triptype+"")
        
        global outputs     
        outputs.append ("out_transform_" + space + "_fromspace_v_tospace_u_"+triptype+".tif")
        outputs.append ("out_transform_" + space + "_fromspace_v_tospace_v_"+triptype+".tif")
        outputs.append ("out_transform_" + space + "_fromspace_v_tospace_v_d"+triptype+".tif")
        return
    
    run_u_fromspace_tests("common")
    run_u_fromspace_tests("object")
    run_u_fromspace_tests("shader")
    run_u_fromspace_tests("world")
    run_u_fromspace_tests("camera")
    
    def run_v_fromspace_tospace_tests (space) :
        global command
        command += testshade("-t 1 -g 32 32 -param tospace " + space + " -od uint8 -o Cout out_transform_v_fromspace_" + space + "_tospace_u_"+triptype+".tif test_transform_v_fromspace_u_tospace_u_"+triptype+"")
        command += testshade("-t 1 -g 32 32 -param tospace " + space + " -od uint8 -o Cout out_transform_v_fromspace_" + space + "_tospace_v_"+triptype+".tif test_transform_v_fromspace_u_tospace_v_"+triptype+"")     
        command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 32 32 -param tospace " + space + " -od uint8 -o Cout out_transform_v_fromspace_" + space + "_tospace_v_d"+triptype+".tif test_transform_v_fromspace_u_tospace_v_d"+triptype+"")
        
        global outputs     
        outputs.append ("out_transform_v_fromspace_" + space + "_tospace_u_"+triptype+".tif")
        outputs.append ("out_transform_v_fromspace_" + space + "_tospace_v_"+triptype+".tif")
        outputs.append ("out_transform_v_fromspace_" + space + "_tospace_v_d"+triptype+".tif")
        return
    
    run_v_fromspace_tospace_tests("common")
    run_v_fromspace_tospace_tests("object")
    run_v_fromspace_tospace_tests("shader")
    run_v_fromspace_tospace_tests("world")
    run_v_fromspace_tospace_tests("camera")
    return


run_2space_tests ("normal")
run_2space_tests ("point")
run_2space_tests ("vector")

def run_matrix_tests (triptype) :
    global command
    command += testshade("--center -t 1 -g 32 32 -od uint8 -o Cout out_transform_u_matrix_u_"+triptype+".tif test_transform_u_matrix_u_"+triptype+"")
    command += testshade("--center -t 1 -g 32 32 -od uint8 -o Cout out_transform_u_matrix_v_"+triptype+".tif test_transform_u_matrix_v_"+triptype+"")     
    command += testshade("--center --vary_udxdy --vary_vdxdy -t 1 -g 32 32 -od uint8 -o Cout out_transform_u_matrix_v_d"+triptype+".tif test_transform_u_matrix_v_d"+triptype+"")

    command += testshade("--center -t 1 -g 32 32 -od uint8 -o Cout out_transform_v_matrix_u_"+triptype+".tif test_transform_v_matrix_u_"+triptype+"")
    command += testshade("--center -t 1 -g 32 32 -od uint8 -o Cout out_transform_v_matrix_v_"+triptype+".tif test_transform_v_matrix_v_"+triptype+"")
    command += testshade("--center --vary_udxdy --vary_vdxdy -t 1 -g 32 32 -od uint8 -o Cout out_transform_v_matrix_v_d"+triptype+".tif test_transform_v_matrix_v_d"+triptype+"")
    
    command += testshade("--center -t 1 -g 32 32 -od uint8 -o Cout out_transform_u_matrix_affine_u_"+triptype+".tif test_transform_u_matrix_affine_u_"+triptype+"")
    command += testshade("--center -t 1 -g 32 32 -od uint8 -o Cout out_transform_u_matrix_affine_v_"+triptype+".tif test_transform_u_matrix_affine_v_"+triptype+"")
    command += testshade("--center --vary_udxdy --vary_vdxdy -t 1 -g 32 32 -od uint8 -o Cout out_transform_u_matrix_affine_v_d"+triptype+".tif test_transform_u_matrix_affine_v_d"+triptype+"")
    
    command += testshade("--center -t 1 -g 32 32 -od uint8 -o Cout out_transform_v_matrix_affine_u_"+triptype+".tif test_transform_v_matrix_affine_u_"+triptype+"")
    command += testshade("--center -t 1 -g 32 32 -od uint8 -o Cout out_transform_v_matrix_affine_v_"+triptype+".tif test_transform_v_matrix_affine_v_"+triptype+"")
    command += testshade("--center --vary_udxdy --vary_vdxdy -t 1 -g 32 32 -od uint8 -o Cout out_transform_v_matrix_affine_v_d"+triptype+".tif test_transform_v_matrix_affine_v_d"+triptype+"")
    
    global outputs     
    outputs.append ("out_transform_u_matrix_u_"+triptype+".tif")
    outputs.append ("out_transform_u_matrix_v_"+triptype+".tif")
    outputs.append ("out_transform_u_matrix_v_d"+triptype+".tif")
    
    outputs.append ("out_transform_v_matrix_u_"+triptype+".tif")
    outputs.append ("out_transform_v_matrix_v_"+triptype+".tif")
    outputs.append ("out_transform_v_matrix_v_d"+triptype+".tif")
    
    outputs.append ("out_transform_u_matrix_affine_u_"+triptype+".tif")
    outputs.append ("out_transform_u_matrix_affine_v_"+triptype+".tif")
    outputs.append ("out_transform_u_matrix_affine_v_d"+triptype+".tif")

    outputs.append ("out_transform_v_matrix_affine_u_"+triptype+".tif")
    outputs.append ("out_transform_v_matrix_affine_v_"+triptype+".tif")
    outputs.append ("out_transform_v_matrix_affine_v_d"+triptype+".tif")
    return

run_matrix_tests("normal")
run_matrix_tests("point")
run_matrix_tests("vector")

def run_1space_tests (triptype) :
    def run_tospace_tests (space) :
        global command
        command += testshade("-t 1 -g 32 32 -param tospace "+space+" -od uint8 -o Cout out_transform_"+space+"_tospace_u_"+triptype+".tif test_transform_u_tospace_u_"+triptype+"")
        command += testshade("-t 1 -g 32 32 -param tospace "+space+" -od uint8 -o Cout out_transform_"+space+"_tospace_v_"+triptype+".tif test_transform_u_tospace_v_"+triptype+"")
        command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 32 32 -param tospace "+space+" -od uint8 -o Cout out_transform_"+space+"_tospace_v_d"+triptype+".tif test_transform_u_tospace_v_d"+triptype+"")
        
        global outputs     
        outputs.append ("out_transform_"+space+"_tospace_u_"+triptype+".tif")
        outputs.append ("out_transform_"+space+"_tospace_v_"+triptype+".tif")
        outputs.append ("out_transform_"+space+"_tospace_v_d"+triptype+".tif")
        return
    run_tospace_tests("common")
    run_tospace_tests("object")
    run_tospace_tests("shader")
    run_tospace_tests("world")
    run_tospace_tests("camera")
    
    global command
    command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_transform_v_tospace_u_"+triptype+".tif test_transform_v_tospace_u_"+triptype+"")
    command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_transform_v_tospace_v_"+triptype+".tif test_transform_v_tospace_v_"+triptype+"")
    command += testshade("--vary_udxdy --vary_vdxdy -t 1 -g 32 32 -od uint8 -o Cout out_transform_v_tospace_v_d"+triptype+".tif test_transform_v_tospace_v_d"+triptype+"")
    
    global outputs     
    outputs.append ("out_transform_v_tospace_u_"+triptype+".tif")
    outputs.append ("out_transform_v_tospace_v_"+triptype+".tif")
    outputs.append ("out_transform_v_tospace_v_d"+triptype+".tif")
    return

run_1space_tests("normal")
run_1space_tests("point")
run_1space_tests("vector")
    
# expect a few LSB failures
failthresh = 0.008
failpercent = 3

