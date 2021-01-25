#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade ("-g 1000 64 -od half -o Cout out.exr test")
outputs += [ "out.exr" ]

# Allow some per-platform numerical slop
failthresh = 0.004
failpercent = 0.05
