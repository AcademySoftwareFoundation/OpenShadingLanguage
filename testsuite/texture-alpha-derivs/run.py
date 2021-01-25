#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Set up a texture that is a gradient image with a distinct alpha gradient
# of 0.5 from left to right and 0.25 right to left, and no change in value
# for R,G,B.
command += oiiotool("-pattern fill:topleft=0.125,0.25,0.5,0:topright=0.125,0.25,0.5,0.5:bottomleft=0.125,0.25,0.5,0.25:bottomright=0.125,0.25,0.5,0.75 64x64 4 -d half -o alpharamp.exr")

command += testshade("-g 64 64 --center -od uint8 -o Cout out.tif test")
outputs = [ "out.txt", "out.tif" ]
