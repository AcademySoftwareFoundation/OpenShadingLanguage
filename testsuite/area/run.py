#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command = testshade("-g 64 64 --vary_pdxdy --vary_udxdy --vary_vdxdy -od float -o Aout out.tif test")
outputs = [ "out.txt", "out.tif" ]

# expect a few LSB failures
failthresh = 0.008
failpercent = 3
