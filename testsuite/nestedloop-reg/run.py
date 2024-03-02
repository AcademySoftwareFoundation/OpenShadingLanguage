#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


###############################
# 
###############################
command += testshade("-t 1 -g 32 32 -od uint8 test  -o cout out.tif ")
outputs.append ("out.tif")

# expect a few LSB failures
failthresh = 0.008
failpercent = 3

