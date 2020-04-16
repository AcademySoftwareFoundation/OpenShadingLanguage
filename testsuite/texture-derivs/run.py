#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

command += testshade("-g 128 128 --center -od uint8 -o Cout out.tif -o dx dx.tif -o dy dy.tif -param scale 128.0 -param filename data/ramp.exr test")
outputs = [ "out.txt", "out.tif", "dx.tif", "dy.tif" ]
