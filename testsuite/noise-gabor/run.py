#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

command = oslc("../common/shaders/testnoise.osl")
command += testshade ("-g 512 512 -od uint8 -o Cout out.tif -param noisename gabor testnoise")
outputs = [ "out.txt", "out.tif" ]
# expect a few LSB failures
failthresh = 0.004
failpercent = 0.05
