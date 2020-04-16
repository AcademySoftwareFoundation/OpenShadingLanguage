#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

# Simple test on a grid texture
command = testshade("-g 1 1 test")

# Construct a test specifically for odd data and pixel windows
command += oiiotool("--pattern checker 100x50+10+20 3 --fullsize 300x200+0+0 -o win.exr")

command += testshade("-g 1 1 --param filename win.exr --param date 0 test")

# test that constant color can be detected
command += oiiotool ("--pattern constant:color=0.1,0.5,0.1,1 128x128 4 -d uint8 -o green.tif")
command += maketx ("green.tif -o green.tx")
command += testshade("-g 1 1 -param filename green.tx --param date 0 test")
