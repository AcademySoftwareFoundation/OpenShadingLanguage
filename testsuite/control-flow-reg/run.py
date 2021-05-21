#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade("-t 1 -g 8 8 exit_in_initops_of_unlockedgeom -od uint8 -o outputVal exit_in_initops_of_unlockedgeom.tif")
command += testshade("-t 1 -g 8 8 exit_in_initops_of_unlockedgeom_useparam_in_uniform_then -od uint8 -o outputVal exit_in_initops_of_unlockedgeom_useparam_in_uniform_then.tif")
command += testshade("-t 1 -g 8 8 exit_in_initops_of_unlockedgeom_useparam_in_varying_then -od uint8 -o outputVal exit_in_initops_of_unlockedgeom_useparam_in_varying_then.tif")
command += testshade("-t 1 -g 8 8 exit_in_initops_of_unlockedgeom_notfound -od uint8 -o outputVal exit_in_initops_of_unlockedgeom_notfound.tif")
command += testshade("-t 1 -g 8 8 exit_in_varying_then_of_initops_of_unlockedgeom_notfound_useparam_in_varying_then -od uint8 -o outputVal exit_in_varying_then_of_initops_of_unlockedgeom_notfound_useparam_in_varying_then.tif")
command += testshade("-t 1 -g 8 8 exit_in_varying_then_of_initops_of_unlockedgeom_notfound_useparam_in_uniform_then -od uint8 -o outputVal exit_in_varying_then_of_initops_of_unlockedgeom_notfound_useparam_in_uniform_then.tif")
command += testshade("-t 1 -g 8 8 exit_in_varying_then_of_initops_of_unlockedgeom_notfound -od uint8 -o outputVal exit_in_varying_then_of_initops_of_unlockedgeom_notfound.tif")
command += testshade("-t 1 -g 8 8 exit_in_varying_then_of_func -od uint8 -o c exit_in_varying_then_of_func.tif")
command += testshade("-t 1 -g 8 8 exit_return_in_varying_thens_of_func_in_uniform_dowhile -od uint8 -o c exit_return_in_varying_thens_of_func_in_uniform_dowhile.tif")
command += testshade("-t 1 -g 8 8 exit_in_varying_then -od uint8 -o c exit_in_varying_then.tif")
command += testshade("-t 1 -g 8 8 exit_in_varying_then_skips_else -od uint8 -o c exit_in_varying_then_skips_else.tif")
command += testshade("-t 1 -g 8 8 exit_in_varying_then_of_initfunc_skips_loop -od uint8 -o c exit_in_varying_then_of_initfunc_skips_loop.tif")
command += testshade("-t 1 -g 8 8 exit_in_varying_then_of_initfunc -od uint8 -o c exit_in_varying_then_of_initfunc.tif")
command += testshade("-t 1 -g 8 8 exit_in_varying_then_of_input_initfunc -od uint8 -o c exit_in_varying_then_of_input_initfunc.tif")

command += testshade("-t 1 -g 8 8 return_in_shader_scope -od uint8 -o c return_in_shader_scope.tif")
command += testshade("-t 1 -g 8 8 return_in_uniform_then -od uint8 -o c return_in_uniform_then.tif")
command += testshade("-t 1 -g 8 8 return_in_varying_then -od uint8 -o c return_in_varying_then.tif")
command += testshade("-t 1 -g 8 8 return_in_uniform_then_in_varying_then -od uint8 -o c return_in_uniform_then_in_varying_then.tif")
command += testshade("-t 1 -g 8 8 return_in_uniform_else -od uint8 -o c return_in_uniform_else.tif")
command += testshade("-t 1 -g 8 8 return_in_varying_else -od uint8 -o c return_in_varying_else.tif")
command += testshade("-t 1 -g 8 8 return_in_varying_then_in_func -od uint8 -o c return_in_varying_then_in_func.tif")
command += testshade("-t 1 -g 8 8 return_in_then_of_nested_funcs -od uint8 -o c return_in_then_of_nested_funcs.tif")
command += testshade("-t 1 -g 8 8 return_in_varying_then_b4_loop -od uint8 -o c return_in_varying_then_b4_loop.tif")
command += testshade("-t 1 -g 8 8 return_in_varying_then_of_uniform_loop -od uint8 -o c return_in_varying_then_of_uniform_loop.tif")
command += testshade("-t 1 -g 8 8 return_in_uniform_thens_in_varying_then_in_uniform_loop -od uint8 -o c return_in_uniform_thens_in_varying_then_in_uniform_loop.tif")
command += testshade("-t 1 -g 8 8 return_in_varying_then_of_initops_of_unlockedgeom -od uint8 -o outputVal return_in_varying_then_of_initops_of_unlockedgeom.tif")
command += testshade("-t 1 -g 8 8 return_in_varying_then_of_initops_of_unlockedgeom_notfound -od uint8 -o outputVal return_in_varying_then_of_initops_of_unlockedgeom_notfound.tif")
command += testshade("-t 1 -g 8 8 return_exit_in_varying_then_of_nested_funcs -od uint8 -o c return_exit_in_varying_then_of_nested_funcs.tif")

command += testshade("-t 1 -g 8 8 varying_dowhile -od uint8 -o c varying_dowhile.tif")
command += testshade("-t 1 -g 8 8 return_in_uniform_then_of_varying_dowhile -od uint8 -o c return_in_uniform_then_of_varying_dowhile.tif")
command += testshade("-t 1 -g 8 8 return_in_varying_then_of_varying_dowhile -od uint8 -o c return_in_varying_then_of_varying_dowhile.tif")
command += testshade("-t 1 -g 8 8 exit_in_uniform_then_of_varying_dowhile -od uint8 -o c exit_in_uniform_then_of_varying_dowhile.tif")
command += testshade("-t 1 -g 8 8 exit_in_varying_then_of_varying_dowhile -od uint8 -o c exit_in_varying_then_of_varying_dowhile.tif")

command += testshade("-t 1 -g 8 8 return_in_uniform_then_of_varying_loop -od uint8 -o c return_in_uniform_then_of_varying_loop.tif")
command += testshade("-t 1 -g 8 8 return_in_varying_then_of_varying_loop -od uint8 -o c return_in_varying_then_of_varying_loop.tif")
command += testshade("-t 1 -g 8 8 exit_in_uniform_then_of_varying_loop -od uint8 -o c exit_in_uniform_then_of_varying_loop.tif")
command += testshade("-t 1 -g 8 8 exit_in_varying_then_of_varying_loop -od uint8 -o c exit_in_varying_then_of_varying_loop.tif")

command += testshade("-t 1 -g 8 8 break_return_in_uniform_then_in_uniform_loop_in_varying_then -od uint8 -o c break_return_in_uniform_then_in_uniform_loop_in_varying_then.tif")
command += testshade("-t 1 -g 8 8 break_in_uniform_then_return_in_varying_then_in_uniform_loop_in_varying_then -od uint8 -o c break_in_uniform_then_return_in_varying_then_in_uniform_loop_in_varying_then.tif")
command += testshade("-t 1 -g 8 8 break_in_varying_then_return_in_uniform_then_in_uniform_loop_in_varying_then -od uint8 -o c break_in_varying_then_return_in_uniform_then_in_uniform_loop_in_varying_then.tif")
command += testshade("-t 1 -g 8 8 break_return_in_varying_then_in_uniform_loop_in_varying_then -od uint8 -o c break_return_in_varying_then_in_uniform_loop_in_varying_then.tif")

command += testshade("-t 1 -g 8 8 break_exit_in_uniform_then_in_uniform_loop_in_varying_then -od uint8 -o c break_exit_in_uniform_then_in_uniform_loop_in_varying_then.tif")
command += testshade("-t 1 -g 8 8 break_in_uniform_then_exit_in_varying_then_in_uniform_loop_in_varying_then -od uint8 -o c break_in_uniform_then_exit_in_varying_then_in_uniform_loop_in_varying_then.tif")
command += testshade("-t 1 -g 8 8 break_in_varying_then_exit_in_uniform_then_in_uniform_loop_in_varying_then -od uint8 -o c break_in_varying_then_exit_in_uniform_then_in_uniform_loop_in_varying_then.tif")
command += testshade("-t 1 -g 8 8 break_exit_in_varying_then_in_uniform_loop_in_varying_then -od uint8 -o c break_exit_in_varying_then_in_uniform_loop_in_varying_then.tif")


command += testshade("-t 1 -g 8 8 continue_return_in_uniform_then_in_uniform_loop_in_varying_then -od uint8 -o c continue_return_in_uniform_then_in_uniform_loop_in_varying_then.tif")
command += testshade("-t 1 -g 8 8 continue_in_uniform_then_return_in_varying_then_in_uniform_loop_in_varying_then -od uint8 -o c continue_in_uniform_then_return_in_varying_then_in_uniform_loop_in_varying_then.tif")
command += testshade("-t 1 -g 8 8 continue_in_varying_then_return_in_uniform_then_in_uniform_loop_in_varying_then -od uint8 -o c continue_in_varying_then_return_in_uniform_then_in_uniform_loop_in_varying_then.tif")
command += testshade("-t 1 -g 8 8 continue_return_in_varying_then_in_uniform_loop_in_varying_then -od uint8 -o c continue_return_in_varying_then_in_uniform_loop_in_varying_then.tif")

command += testshade("-t 1 -g 8 8 continue_exit_in_uniform_then_in_uniform_loop_in_varying_then -od uint8 -o c continue_exit_in_uniform_then_in_uniform_loop_in_varying_then.tif")
command += testshade("-t 1 -g 8 8 continue_in_uniform_then_exit_in_varying_then_in_uniform_loop_in_varying_then -od uint8 -o c continue_in_uniform_then_exit_in_varying_then_in_uniform_loop_in_varying_then.tif")
command += testshade("-t 1 -g 8 8 continue_in_varying_then_exit_in_uniform_then_in_uniform_loop_in_varying_then -od uint8 -o c continue_in_varying_then_exit_in_uniform_then_in_uniform_loop_in_varying_then.tif")
command += testshade("-t 1 -g 8 8 continue_exit_in_varying_then_in_uniform_loop_in_varying_then -od uint8 -o c continue_exit_in_varying_then_in_uniform_loop_in_varying_then.tif")

command += testshade("-t 1 -g 8 8 -layer a layer_a --layer b layer_b_exit_in_varying_then --connect a f_out b f_in --connect a c_out b c_in -od uint8 -o c_outb a_2_b_exit_in_varying_then.tif")
command += testshade("-t 1 -g 8 8 -layer a layer_a_return_exit_in_varying_then --layer b layer_b --connect a f_out b f_in --connect a c_out b c_in -od uint8 -o c_outb a_return_exit_in_varying_then_2_b.tif")
command += testshade("-t 1 -g 8 8 -layer c layer_c --layer d layer_d_return_in_unvisited_then --connect c f_out d f_in --connect c c_out d c_in -od uint8 -o c_outb c_2_d_return_in_unvisited_then.tif")
command += testshade("-t 1 -g 8 8 -layer e layer_e --layer f layer_f_return_in_varying_then_in_uniform_then_in_varying_loop --connect e f_out f f_in --connect e c_out f c_in -od uint8 -o c_outb e_2_layer_f_return_in_varying_then_in_uniform_then_in_varying_loop.tif")

outputs = [ 
    "exit_in_initops_of_unlockedgeom.tif",
    "exit_in_initops_of_unlockedgeom_useparam_in_uniform_then.tif",
    "exit_in_initops_of_unlockedgeom_useparam_in_varying_then.tif",
    "exit_in_initops_of_unlockedgeom_notfound.tif",
    "exit_in_varying_then_of_initops_of_unlockedgeom_notfound_useparam_in_varying_then.tif",
    "exit_in_varying_then_of_initops_of_unlockedgeom_notfound_useparam_in_uniform_then.tif",
    "exit_in_varying_then_of_initops_of_unlockedgeom_notfound.tif",
    "exit_in_varying_then_of_func.tif",
    "exit_in_varying_then.tif",
    "exit_in_varying_then_skips_else.tif",
    "exit_in_varying_then_of_initfunc_skips_loop.tif",
    "exit_in_varying_then_of_initfunc.tif",
    "exit_in_varying_then_of_input_initfunc.tif",
    "exit_return_in_varying_thens_of_func_in_uniform_dowhile.tif",
    
    "return_in_shader_scope.tif",
    "return_in_uniform_then.tif",
    "return_in_varying_then.tif",
    "return_in_uniform_then_in_varying_then.tif",
    "return_in_uniform_else.tif",
    "return_in_varying_else.tif",
    "return_in_varying_then_in_func.tif",
    "return_in_then_of_nested_funcs.tif",
    "return_in_varying_then_b4_loop.tif",
    "return_in_varying_then_of_uniform_loop.tif",
    "return_in_uniform_thens_in_varying_then_in_uniform_loop.tif",
    "return_in_varying_then_of_initops_of_unlockedgeom.tif",
    "return_in_varying_then_of_initops_of_unlockedgeom_notfound.tif",
    "return_exit_in_varying_then_of_nested_funcs.tif",
    
    "varying_dowhile.tif",
    "return_in_uniform_then_of_varying_dowhile.tif",
    "return_in_varying_then_of_varying_dowhile.tif",
    "exit_in_uniform_then_of_varying_dowhile.tif",
    "exit_in_varying_then_of_varying_dowhile.tif",
    
    "return_in_uniform_then_of_varying_loop.tif",
    "return_in_varying_then_of_varying_loop.tif",    

    "exit_in_uniform_then_of_varying_loop.tif",
    "exit_in_varying_then_of_varying_loop.tif",    
    
    "break_return_in_uniform_then_in_uniform_loop_in_varying_then.tif",
    "break_in_uniform_then_return_in_varying_then_in_uniform_loop_in_varying_then.tif",
    "break_in_varying_then_return_in_uniform_then_in_uniform_loop_in_varying_then.tif",
    "break_return_in_varying_then_in_uniform_loop_in_varying_then.tif",

    "break_exit_in_uniform_then_in_uniform_loop_in_varying_then.tif",
    "break_in_uniform_then_exit_in_varying_then_in_uniform_loop_in_varying_then.tif",
    "break_in_varying_then_exit_in_uniform_then_in_uniform_loop_in_varying_then.tif",
    "break_exit_in_varying_then_in_uniform_loop_in_varying_then.tif",

    "continue_return_in_uniform_then_in_uniform_loop_in_varying_then.tif",
    "continue_in_uniform_then_return_in_varying_then_in_uniform_loop_in_varying_then.tif",
    "continue_in_varying_then_return_in_uniform_then_in_uniform_loop_in_varying_then.tif",
    "continue_return_in_varying_then_in_uniform_loop_in_varying_then.tif",

    "continue_exit_in_uniform_then_in_uniform_loop_in_varying_then.tif",
    "continue_in_uniform_then_exit_in_varying_then_in_uniform_loop_in_varying_then.tif",
    "continue_in_varying_then_exit_in_uniform_then_in_uniform_loop_in_varying_then.tif",
    "continue_exit_in_varying_then_in_uniform_loop_in_varying_then.tif",
    
    "a_2_b_exit_in_varying_then.tif",
    "a_return_exit_in_varying_then_2_b.tif",
    "c_2_d_return_in_unvisited_then.tif",
    "e_2_layer_f_return_in_varying_then_in_uniform_then_in_varying_loop.tif",
    
    "out.txt"
]




# expect a few LSB failures
#failthresh = 0.008
#failpercent = 3
