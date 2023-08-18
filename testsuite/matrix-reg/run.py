#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_matrix_u_determinant.tif test_matrix_u_determinant")
command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_matrix_v_determinant.tif test_matrix_v_determinant")
outputs.append ("out_matrix_u_determinant.tif")
outputs.append ("out_matrix_v_determinant.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_matrix_u_transpose.tif test_matrix_u_transpose")
command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_matrix_v_transpose.tif test_matrix_v_transpose")
outputs.append ("out_matrix_u_transpose.tif")
outputs.append ("out_matrix_v_transpose.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_matrix_16x_u_float.tif test_matrix_16x_u_float")
command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_matrix_16x_v_float.tif test_matrix_16x_v_float")
command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_matrix_u_float.tif test_matrix_u_float")
command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_matrix_v_float.tif test_matrix_v_float")
outputs.append ("out_matrix_16x_u_float.tif")
outputs.append ("out_matrix_16x_v_float.tif")
outputs.append ("out_matrix_u_float.tif")
outputs.append ("out_matrix_v_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_matrix_v_fromspace_16x_u_float.tif test_matrix_v_fromspace_16x_u_float")
command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_matrix_v_fromspace_16x_v_float.tif test_matrix_v_fromspace_16x_v_float")
outputs.append ("out_matrix_v_fromspace_16x_u_float.tif")
outputs.append ("out_matrix_v_fromspace_16x_v_float.tif")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_matrix_v_fromspace_u_float.tif test_matrix_v_fromspace_u_float")
command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_matrix_v_fromspace_v_float.tif test_matrix_v_fromspace_v_float")
outputs.append ("out_matrix_v_fromspace_u_float.tif")
outputs.append ("out_matrix_v_fromspace_v_float.tif")


command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_matrix_v_fromspace_v_tospace.tif test_matrix_v_fromspace_v_tospace")
outputs.append ("out_matrix_v_fromspace_v_tospace.tif")


def run_space_tests (space) :
    global command
    global outputs     
    
    command += testshade("-t 1 -g 32 32 -param fromspace "+space+" -od uint8 -o Cout out_matrix_"+space+"_fromspace_u_float.tif test_matrix_u_fromspace_u_float")
    command += testshade("-t 1 -g 32 32 -param fromspace "+space+" -od uint8 -o Cout out_matrix_"+space+"_fromspace_v_float.tif test_matrix_u_fromspace_v_float")
    command += testshade("-t 1 -g 32 32 -param fromspace "+space+" -od uint8 -o Cout out_matrix_"+space+"_fromspace_16x_u_float.tif test_matrix_u_fromspace_16x_u_float")
    command += testshade("-t 1 -g 32 32 -param fromspace "+space+" -od uint8 -o Cout out_matrix_"+space+"_fromspace_16x_v_float.tif test_matrix_u_fromspace_16x_v_float")    
    outputs.append ("out_matrix_"+space+"_fromspace_u_float.tif")
    outputs.append ("out_matrix_"+space+"_fromspace_v_float.tif")
    outputs.append ("out_matrix_"+space+"_fromspace_16x_u_float.tif")
    outputs.append ("out_matrix_"+space+"_fromspace_16x_v_float.tif")
    
    command += testshade("-t 1 -g 32 32 -param tospace "+space+" -od uint8 -o Cout out_matrix_v_fromspace_"+space+"_tospace.tif test_matrix_v_fromspace_u_tospace")
    command += testshade("-t 1 -g 32 32 -param fromspace "+space+" -od uint8 -o Cout out_matrix_"+space+"_fromspace_v_tospace.tif test_matrix_u_fromspace_v_tospace")    
    outputs.append ("out_matrix_v_fromspace_"+space+"_tospace.tif")
    outputs.append ("out_matrix_"+space+"_fromspace_v_tospace.tif")
    
    command += testshade("-t 1 -g 32 32 -param fromspace "+space+" -od uint8 -o Cout out_getmatrix_"+space+"_fromspace_v_tospace.tif test_getmatrix_u_fromspace_v_tospace")
    command += testshade("-t 1 -g 32 32 -param tospace "+space+" -od uint8 -o Cout out_getmatrix_v_fromspace_"+space+"_tospace.tif test_getmatrix_v_fromspace_u_tospace")
    outputs.append ("out_getmatrix_"+space+"_fromspace_v_tospace.tif")
    outputs.append ("out_getmatrix_v_fromspace_"+space+"_tospace.tif")

    def run_from_to_test (fromspace, tospace) :
        global command
        command += testshade("-t 1 -g 32 32 -param fromspace "+fromspace+" -param tospace "+tospace+" -od uint8 -o Cout out_matrix_"+fromspace+"_fromspace_"+tospace+"_tospace.tif test_matrix_u_fromspace_u_tospace")
        command += testshade("-t 1 -g 32 32 -param fromspace "+fromspace+" -param tospace "+tospace+" -od uint8 -o Cout out_getmatrix_"+fromspace+"_fromspace_"+tospace+"_tospace.tif test_getmatrix_u_fromspace_u_tospace")
        
        global outputs     
        outputs.append ("out_matrix_"+fromspace+"_fromspace_"+tospace+"_tospace.tif")
        outputs.append ("out_getmatrix_"+fromspace+"_fromspace_"+tospace+"_tospace.tif")
        return

    run_from_to_test(space, "common")
    run_from_to_test(space, "object")
    run_from_to_test(space, "shader")
    #run_from_to_test(space, "world")
    #run_from_to_test(space, "camera")
    #run_from_to_test(space, "spam")
    return
    
run_space_tests("common")
run_space_tests("object")
run_space_tests("shader")
#run_space_tests("world")
#run_space_tests("camera")
#run_space_tests("spam")

command += testshade("-t 1 -g 32 32 -od uint8 -o Cout out_getmatrix_v_fromspace_v_tospace.tif test_getmatrix_v_fromspace_v_tospace")
outputs.append ("out_getmatrix_v_fromspace_v_tospace.tif")



# expect a few LSB failures
failthresh = 0.008
failpercent = 3

