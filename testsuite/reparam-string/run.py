#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade(" ".join([
    "--layer lay0",
    "--param:type=string:interactive=1 test_string 'initial value'",
    "test --iters 2",
    "--reparam:type=string:interactive=1 lay0 test_string 'updated value'",
]))

outputs = [ "out.txt" ]

