#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade("-g 2 2 -layer alayer a -layer blayer b --layer clayer c --connect alayer f_out clayer f_in --connect alayer c_out clayer c_in --connect blayer out clayer unused")
