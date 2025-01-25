#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

shader_commands = " ".join([
    "-shader trace_shader trace_layer",
    "-shader main_shader main_layer",
    "-connect trace_layer x main_layer x",
])

# Run once with default (lazytrace=1), and once explicitly disabled
command += testshade(shader_commands)
command += testshade("{} --options \"lazytrace=0\"".format(shader_commands))
