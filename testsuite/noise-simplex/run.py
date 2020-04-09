#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

command = oslc("../common/shaders/testnoise.osl")
command += testshade ("-g 512 512 -od uint8 -o Cout out.tif -param noisename simplex testnoise")
command += testshade ("-g 512 512 -od uint8 -o Cout uout.tif -param noisename usimplex -param offset 0.0 -param scale 1.0 testnoise")
outputs = [ "out.txt", "out.tif", "uout.tif" ]
# expect some LSB failures on this test
failthresh = 0.004
failpercent = 0.05
