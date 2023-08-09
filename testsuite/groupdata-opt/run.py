#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

shader_commands = " ".join([
    "-shader input_shader input_layer",
    "-shader main_shader main_layer",
    "-connect input_layer Val_Out main_layer Val_In",
])
for opt in [0, 1]:
    # Assert that groupdata is sized differently based on opt_groupdata param
    command += testshade("{} --options opt_groupdata={} --print-groupdata".format(shader_commands, opt))

# Assert that opt_groupdata correctly skips connected output parameters
# Other skip conditions (renderer outputs, closures) are covered by existing tests
shader_commands = " ".join([
    "-shader input_shader input_layer",
    "-shader connected_output_shader test_layer",
    "-shader main_shader main_layer",
    "-connect input_layer Val_Out test_layer Val_Out",
    "-connect test_layer Val_Out main_layer Val_In",
])
command += testshade("{} --print-groupdata".format(shader_commands))
