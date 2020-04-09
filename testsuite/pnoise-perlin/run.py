#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

command = oslc("../common/shaders/testpnoise.osl")
command += testshade("-g 512 512 -od uint8 -o Cout out.tif -param noisename perlin testpnoise")
command += testshade("-g 512 512 -od uint8 -o Cout uout.tif -param noisename uperlin -param offset 0.0 -param scale 1.0 testpnoise")
outputs = [ "out.txt", "out.tif", "uout.tif" ]
