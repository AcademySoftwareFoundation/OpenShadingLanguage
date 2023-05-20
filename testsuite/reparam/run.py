#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade ("--layer lay0 --param:type=float:interactive=1 f 1 --param:type=float:interactive=1 user 2 --param:type=float:interactive=1 third 3 test --iters 2 --reparam:type=float:interactive=1 lay0 f 10.0 --reparam:type=float:interactive=1 lay0 third 30.0")

outputs = [ "out.txt" ]

