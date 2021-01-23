#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command = "echo Without repeated errors:>> out.txt 2>&1 ;\n"
command += testshade("-g 2 2 test")

command += "echo With repeated errors:>> out.txt 2>&1 ;\n"
command += testshade("--options error_repeats=1 -g 2 2 test")
