#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade("--layer alayer a --layer blayer b --connect alayer st_out.s blayer x.s --connect alayer st_out.t blayer x.t --connect alayer st_out blayer y --connect alayer r blayer r")
