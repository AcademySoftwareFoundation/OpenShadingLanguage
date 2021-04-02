#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# sqrt
command += testshade("-g 64 64 -od uint8 -o out_float out_float_sqrt.tif -o out_color out_color_sqrt.tif -o out_point out_point_sqrt.tif -o out_vector out_vector_sqrt.tif -o out_normal out_normal_sqrt.tif test_sqrt")
command += testshade("-g 64 64 --vary_pdxdy --param derivX 1 -od uint8 -o out_float out_float_Dx_sqrt.tif -o out_color out_color_Dx_sqrt.tif -o out_point out_point_Dx_sqrt.tif -o out_vector out_vector_Dx_sqrt.tif -o out_normal out_normal_Dx_sqrt.tif test_sqrt")
command += testshade("-g 64 64 --vary_pdxdy --param derivY 1 -od uint8 -o out_float out_float_Dy_sqrt.tif -o out_color out_color_Dy_sqrt.tif -o out_point out_point_Dy_sqrt.tif -o out_vector out_vector_Dy_sqrt.tif -o out_normal out_normal_Dy_sqrt.tif test_sqrt")

# inversesqrt
command += testshade("-g 64 64 -od uint8 -o out_float out_float_inversesqrt.tif -o out_color out_color_inversesqrt.tif -o out_point out_point_inversesqrt.tif -o out_vector out_vector_inversesqrt.tif -o out_normal out_normal_inversesqrt.tif test_inversesqrt")
command += testshade("-g 64 64 --vary_pdxdy --param derivX 1 --param derivScale -1.0 -od uint8 -o out_float out_float_Dx_inversesqrt.tif -o out_color out_color_Dx_inversesqrt.tif -o out_point out_point_Dx_inversesqrt.tif -o out_vector out_vector_Dx_inversesqrt.tif -o out_normal out_normal_Dx_inversesqrt.tif test_inversesqrt")
command += testshade("-g 64 64 --vary_pdxdy --param derivY 1 --param derivScale -1.0 -od uint8 -o out_float out_float_Dy_inversesqrt.tif -o out_color out_color_Dy_inversesqrt.tif -o out_point out_point_Dy_inversesqrt.tif -o out_vector out_vector_Dy_inversesqrt.tif -o out_normal out_normal_Dy_inversesqrt.tif test_inversesqrt")

# abs
command += testshade("-g 64 64 -od uint8 -o out_float out_float_abs.tif -o out_color out_color_abs.tif -o out_point out_point_abs.tif -o out_vector out_vector_abs.tif -o out_normal out_normal_abs.tif test_abs")
command += testshade("-g 64 64 --vary_pdxdy --param derivX 1 --param derivScale -1.0 -od uint8 -o out_float out_float_Dx_abs.tif -o out_color out_color_Dx_abs.tif -o out_point out_point_Dx_abs.tif -o out_vector out_vector_Dx_abs.tif -o out_normal out_normal_Dx_abs.tif test_abs")
command += testshade("-g 64 64 --vary_pdxdy --param derivY 1 --param derivScale -1.0 -od uint8 -o out_float out_float_Dy_abs.tif -o out_color out_color_Dy_abs.tif -o out_point out_point_Dy_abs.tif -o out_vector out_vector_Dy_abs.tif -o out_normal out_normal_Dy_abs.tif test_abs")

# abs int
command += testshade("-g 64 64 -od uint8 -o out_int out_int_abs.tif test_abs_int")

# floor
command += testshade("-g 64 64 -od uint8 -o out_float out_float_floor.tif -o out_color out_color_floor.tif -o out_point out_point_floor.tif -o out_vector out_vector_floor.tif -o out_normal out_normal_floor.tif test_floor")
command += testshade("-g 64 64 --vary_pdxdy --param derivX 1 -od uint8 -o out_float out_float_Dx_floor.tif -o out_color out_color_Dx_floor.tif -o out_point out_point_Dx_floor.tif -o out_vector out_vector_Dx_floor.tif -o out_normal out_normal_Dx_floor.tif test_floor")
command += testshade("-g 64 64 --vary_pdxdy --param derivY 1 -od uint8 -o out_float out_float_Dy_floor.tif -o out_color out_color_Dy_floor.tif -o out_point out_point_Dy_floor.tif -o out_vector out_vector_Dy_floor.tif -o out_normal out_normal_Dy_floor.tif test_floor")

# ceil
command += testshade("-g 64 64 -od uint8 -o out_float out_float_ceil.tif -o out_color out_color_ceil.tif -o out_point out_point_ceil.tif -o out_vector out_vector_ceil.tif -o out_normal out_normal_ceil.tif test_ceil")
command += testshade("-g 64 64 --vary_pdxdy --param derivX 1 -od uint8 -o out_float out_float_Dx_ceil.tif -o out_color out_color_Dx_ceil.tif -o out_point out_point_Dx_ceil.tif -o out_vector out_vector_Dx_ceil.tif -o out_normal out_normal_Dx_ceil.tif test_ceil")
command += testshade("-g 64 64 --vary_pdxdy --param derivY 1 -od uint8 -o out_float out_float_Dy_ceil.tif -o out_color out_color_Dy_ceil.tif -o out_point out_point_Dy_ceil.tif -o out_vector out_vector_Dy_ceil.tif -o out_normal out_normal_Dy_ceil.tif test_ceil")

# trunc
command += testshade("-g 64 64 -od uint8 -o out_float out_float_trunc.tif -o out_color out_color_trunc.tif -o out_point out_point_trunc.tif -o out_vector out_vector_trunc.tif -o out_normal out_normal_trunc.tif test_trunc")
command += testshade("-g 64 64 --vary_pdxdy --param derivX 1 -od uint8 -o out_float out_float_Dx_trunc.tif -o out_color out_color_Dx_trunc.tif -o out_point out_point_Dx_trunc.tif -o out_vector out_vector_Dx_trunc.tif -o out_normal out_normal_Dx_trunc.tif test_trunc")
command += testshade("-g 64 64 --vary_pdxdy --param derivY 1 -od uint8 -o out_float out_float_Dy_trunc.tif -o out_color out_color_Dy_trunc.tif -o out_point out_point_Dy_trunc.tif -o out_vector out_vector_Dy_trunc.tif -o out_normal out_normal_Dy_trunc.tif test_trunc")

# round
command += testshade("-g 64 64 -od uint8 -o out_float out_float_round.tif -o out_color out_color_round.tif -o out_point out_point_round.tif -o out_vector out_vector_round.tif -o out_normal out_normal_round.tif test_round")
command += testshade("-g 64 64 --vary_pdxdy --param derivX 1 -od uint8 -o out_float out_float_Dx_round.tif -o out_color out_color_Dx_round.tif -o out_point out_point_Dx_round.tif -o out_vector out_vector_Dx_round.tif -o out_normal out_normal_Dx_round.tif test_round")
command += testshade("-g 64 64 --vary_pdxdy --param derivY 1 -od uint8 -o out_float out_float_Dy_round.tif -o out_color out_color_Dy_round.tif -o out_point out_point_Dy_round.tif -o out_vector out_vector_Dy_round.tif -o out_normal out_normal_Dy_round.tif test_round")

# fmod
command += testshade("-g 64 64 -od uint8 -o out_float out_float_fmod.tif -o out_color out_color_fmod.tif -o out_point out_point_fmod.tif -o out_vector out_vector_fmod.tif -o out_normal out_normal_fmod.tif test_fmod")
command += testshade("-g 64 64 --vary_pdxdy --param derivX 1 -od uint8 -o out_float out_float_Dx_fmod.tif -o out_color out_color_Dx_fmod.tif -o out_point out_point_Dx_fmod.tif -o out_vector out_vector_Dx_fmod.tif -o out_normal out_normal_Dx_fmod.tif test_fmod")
command += testshade("-g 64 64 --vary_pdxdy --param derivY 1 -od uint8 -o out_float out_float_Dy_fmod.tif -o out_color out_color_Dy_fmod.tif -o out_point out_point_Dy_fmod.tif -o out_vector out_vector_Dy_fmod.tif -o out_normal out_normal_Dy_fmod.tif test_fmod")

# step
command += testshade("-g 64 64 -od uint8 -o out_float out_float_step.tif -o out_color out_color_step.tif -o out_point out_point_step.tif -o out_vector out_vector_step.tif -o out_normal out_normal_step.tif test_step")
command += testshade("-g 64 64 --vary_pdxdy --param derivX 1 -od uint8 -o out_float out_float_Dx_step.tif -o out_color out_color_Dx_step.tif -o out_point out_point_Dx_step.tif -o out_vector out_vector_Dx_step.tif -o out_normal out_normal_Dx_step.tif test_step")
command += testshade("-g 64 64 --vary_pdxdy --param derivY 1 -od uint8 -o out_float out_float_Dy_step.tif -o out_color out_color_Dy_step.tif -o out_point out_point_Dy_step.tif -o out_vector out_vector_Dy_step.tif -o out_normal out_normal_Dy_step.tif test_step")

# add
command += testshade("-g 64 64 -od uint8 -o out_float out_float_add.tif -o out_color out_color_add.tif -o out_point out_point_add.tif -o out_vector out_vector_add.tif -o out_normal out_normal_add.tif test_add")
command += testshade("-g 64 64 --vary_pdxdy --param derivX 1 -od uint8 -o out_float out_float_Dx_add.tif -o out_color out_color_Dx_add.tif -o out_point out_point_Dx_add.tif -o out_vector out_vector_Dx_add.tif -o out_normal out_normal_Dx_add.tif test_add")
command += testshade("-g 64 64 --vary_pdxdy --param derivY 1 -od uint8 -o out_float out_float_Dy_add.tif -o out_color out_color_Dy_add.tif -o out_point out_point_Dy_add.tif -o out_vector out_vector_Dy_add.tif -o out_normal out_normal_Dy_add.tif test_add")

# sub
command += testshade("-g 64 64 -od uint8 -o out_float out_float_sub.tif -o out_color out_color_sub.tif -o out_point out_point_sub.tif -o out_vector out_vector_sub.tif -o out_normal out_normal_sub.tif test_sub")
command += testshade("-g 64 64 --vary_pdxdy --param derivX 1 -od uint8 -o out_float out_float_Dx_sub.tif -o out_color out_color_Dx_sub.tif -o out_point out_point_Dx_sub.tif -o out_vector out_vector_Dx_sub.tif -o out_normal out_normal_Dx_sub.tif test_sub")
command += testshade("-g 64 64 --vary_pdxdy --param derivY 1 -od uint8 -o out_float out_float_Dy_sub.tif -o out_color out_color_Dy_sub.tif -o out_point out_point_Dy_sub.tif -o out_vector out_vector_Dy_sub.tif -o out_normal out_normal_Dy_sub.tif test_sub")

# mul
command += testshade("-g 64 64 -od uint8 -o out_float out_float_mul.tif -o out_color out_color_mul.tif -o out_point out_point_mul.tif -o out_vector out_vector_mul.tif -o out_normal out_normal_mul.tif test_mul")
command += testshade("-g 64 64 --vary_pdxdy --param derivX 1 -od uint8 -o out_float out_float_Dx_mul.tif -o out_color out_color_Dx_mul.tif -o out_point out_point_Dx_mul.tif -o out_vector out_vector_Dx_mul.tif -o out_normal out_normal_Dx_mul.tif test_mul")
command += testshade("-g 64 64 --vary_pdxdy --param derivY 1 -od uint8 -o out_float out_float_Dy_mul.tif -o out_color out_color_Dy_mul.tif -o out_point out_point_Dy_mul.tif -o out_vector out_vector_Dy_mul.tif -o out_normal out_normal_Dy_mul.tif test_mul")

# div
command += testshade("-g 64 64 -od uint8 -o out_float out_float_div.tif -o out_color out_color_div.tif -o out_point out_point_div.tif -o out_vector out_vector_div.tif -o out_normal out_normal_div.tif test_div")
command += testshade("-g 64 64 --vary_pdxdy --param derivX 1 -od uint8 -o out_float out_float_Dx_div.tif -o out_color out_color_Dx_div.tif -o out_point out_point_Dx_div.tif -o out_vector out_vector_Dx_div.tif -o out_normal out_normal_Dx_div.tif test_div")
command += testshade("-g 64 64 --vary_pdxdy --param derivY 1 -od uint8 -o out_float out_float_Dy_div.tif -o out_color out_color_Dy_div.tif -o out_point out_point_Dy_div.tif -o out_vector out_vector_Dy_div.tif -o out_normal out_normal_Dy_div.tif test_div")


outputs = [ 
    "out_float_sqrt.tif",
    "out_color_sqrt.tif",
    "out_point_sqrt.tif",
    "out_vector_sqrt.tif",
    "out_normal_sqrt.tif",

    "out_float_Dx_sqrt.tif",
    "out_color_Dx_sqrt.tif",
    "out_point_Dx_sqrt.tif",
    "out_vector_Dx_sqrt.tif",
    "out_normal_Dx_sqrt.tif",

    "out_float_Dy_sqrt.tif",
    "out_color_Dy_sqrt.tif",
    "out_point_Dy_sqrt.tif",
    "out_vector_Dy_sqrt.tif",
    "out_normal_Dy_sqrt.tif",
    
    "out_float_inversesqrt.tif",
    "out_color_inversesqrt.tif",
    "out_point_inversesqrt.tif",
    "out_vector_inversesqrt.tif",
    "out_normal_inversesqrt.tif",

    "out_float_Dx_inversesqrt.tif",
    "out_color_Dx_inversesqrt.tif",
    "out_point_Dx_inversesqrt.tif",
    "out_vector_Dx_inversesqrt.tif",
    "out_normal_Dx_inversesqrt.tif",

    "out_float_Dy_inversesqrt.tif",
    "out_color_Dy_inversesqrt.tif",
    "out_point_Dy_inversesqrt.tif",
    "out_vector_Dy_inversesqrt.tif",
    "out_normal_Dy_inversesqrt.tif",
    
    "out_float_abs.tif",
    "out_color_abs.tif",
    "out_point_abs.tif",
    "out_vector_abs.tif",
    "out_normal_abs.tif",

    "out_float_Dx_abs.tif",
    "out_color_Dx_abs.tif",
    "out_point_Dx_abs.tif",
    "out_vector_Dx_abs.tif",
    "out_normal_Dx_abs.tif",

    "out_float_Dy_abs.tif",
    "out_color_Dy_abs.tif",
    "out_point_Dy_abs.tif",
    "out_vector_Dy_abs.tif",
    "out_normal_Dy_abs.tif",
    
    "out_int_abs.tif",        
    
    
    "out_float_floor.tif",
    "out_color_floor.tif",
    "out_point_floor.tif",
    "out_vector_floor.tif",
    "out_normal_floor.tif",

    "out_float_Dx_floor.tif",
    "out_color_Dx_floor.tif",
    "out_point_Dx_floor.tif",
    "out_vector_Dx_floor.tif",
    "out_normal_Dx_floor.tif",

    "out_float_Dy_floor.tif",
    "out_color_Dy_floor.tif",
    "out_point_Dy_floor.tif",
    "out_vector_Dy_floor.tif",
    "out_normal_Dy_floor.tif",
    
    "out_float_ceil.tif",
    "out_color_ceil.tif",
    "out_point_ceil.tif",
    "out_vector_ceil.tif",
    "out_normal_ceil.tif",

    "out_float_Dx_ceil.tif",
    "out_color_Dx_ceil.tif",
    "out_point_Dx_ceil.tif",
    "out_vector_Dx_ceil.tif",
    "out_normal_Dx_ceil.tif",

    "out_float_Dy_ceil.tif",
    "out_color_Dy_ceil.tif",
    "out_point_Dy_ceil.tif",
    "out_vector_Dy_ceil.tif",
    "out_normal_Dy_ceil.tif",
    
    
    "out_float_trunc.tif",
    "out_color_trunc.tif",
    "out_point_trunc.tif",
    "out_vector_trunc.tif",
    "out_normal_trunc.tif",

    "out_float_Dx_trunc.tif",
    "out_color_Dx_trunc.tif",
    "out_point_Dx_trunc.tif",
    "out_vector_Dx_trunc.tif",
    "out_normal_Dx_trunc.tif",

    "out_float_Dy_trunc.tif",
    "out_color_Dy_trunc.tif",
    "out_point_Dy_trunc.tif",
    "out_vector_Dy_trunc.tif",
    "out_normal_Dy_trunc.tif",
    
    "out_float_round.tif",
    "out_color_round.tif",
    "out_point_round.tif",
    "out_vector_round.tif",
    "out_normal_round.tif",

    "out_float_Dx_round.tif",
    "out_color_Dx_round.tif",
    "out_point_Dx_round.tif",
    "out_vector_Dx_round.tif",
    "out_normal_Dx_round.tif",

    "out_float_Dy_round.tif",
    "out_color_Dy_round.tif",
    "out_point_Dy_round.tif",
    "out_vector_Dy_round.tif",
    "out_normal_Dy_round.tif",
    
    
    "out_float_fmod.tif",
    "out_color_fmod.tif",
    "out_point_fmod.tif",
    "out_vector_fmod.tif",
    "out_normal_fmod.tif",

    "out_float_Dx_fmod.tif",
    "out_color_Dx_fmod.tif",
    "out_point_Dx_fmod.tif",
    "out_vector_Dx_fmod.tif",
    "out_normal_Dx_fmod.tif",

    "out_float_Dy_fmod.tif",
    "out_color_Dy_fmod.tif",
    "out_point_Dy_fmod.tif",
    "out_vector_Dy_fmod.tif",
    "out_normal_Dy_fmod.tif",      

    "out_float_step.tif",
    "out_color_step.tif",
    "out_point_step.tif",
    "out_vector_step.tif",
    "out_normal_step.tif",

    "out_float_Dx_step.tif",
    "out_color_Dx_step.tif",
    "out_point_Dx_step.tif",
    "out_vector_Dx_step.tif",
    "out_normal_Dx_step.tif",

    "out_float_Dy_step.tif",
    "out_color_Dy_step.tif",
    "out_point_Dy_step.tif",
    "out_vector_Dy_step.tif",
    "out_normal_Dy_step.tif",      

    "out_float_add.tif",
    "out_color_add.tif",
    "out_point_add.tif",
    "out_vector_add.tif",
    "out_normal_add.tif",

    "out_float_Dx_add.tif",
    "out_color_Dx_add.tif",
    "out_point_Dx_add.tif",
    "out_vector_Dx_add.tif",
    "out_normal_Dx_add.tif",

    "out_float_Dy_add.tif",
    "out_color_Dy_add.tif",
    "out_point_Dy_add.tif",
    "out_vector_Dy_add.tif",
    "out_normal_Dy_add.tif",
    
    "out_float_sub.tif",
    "out_color_sub.tif",
    "out_point_sub.tif",
    "out_vector_sub.tif",
    "out_normal_sub.tif",

    "out_float_Dx_sub.tif",
    "out_color_Dx_sub.tif",
    "out_point_Dx_sub.tif",
    "out_vector_Dx_sub.tif",
    "out_normal_Dx_sub.tif",

    "out_float_Dy_sub.tif",
    "out_color_Dy_sub.tif",
    "out_point_Dy_sub.tif",
    "out_vector_Dy_sub.tif",
    "out_normal_Dy_sub.tif",            
                  
    "out_float_mul.tif",
    "out_color_mul.tif",
    "out_point_mul.tif",
    "out_vector_mul.tif",
    "out_normal_mul.tif",

    "out_float_Dx_mul.tif",
    "out_color_Dx_mul.tif",
    "out_point_Dx_mul.tif",
    "out_vector_Dx_mul.tif",
    "out_normal_Dx_mul.tif",

    "out_float_Dy_mul.tif",
    "out_color_Dy_mul.tif",
    "out_point_Dy_mul.tif",
    "out_vector_Dy_mul.tif",
    "out_normal_Dy_mul.tif",              
                  
    "out_float_div.tif",
    "out_color_div.tif",
    "out_point_div.tif",
    "out_vector_div.tif",
    "out_normal_div.tif",

    "out_float_Dx_div.tif",
    "out_color_Dx_div.tif",
    "out_point_Dx_div.tif",
    "out_vector_Dx_div.tif",
    "out_normal_Dx_div.tif",

    "out_float_Dy_div.tif",
    "out_color_Dy_div.tif",
    "out_point_Dy_div.tif",
    "out_vector_Dy_div.tif",
    "out_normal_Dy_div.tif"              
                  
]




# expect a few LSB failures
failthresh = 0.008
failpercent = 3
