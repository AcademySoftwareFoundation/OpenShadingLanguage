#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += oiiotool ("-pattern constant:color=.5,.1,.1 128x128 3 -d uint8 -otex file.1001.tx")
command += oiiotool ("-pattern constant:color=.1,.5,.1 256x256 3 -d uint8 -otex file.1002.tx")
command += oiiotool ("-pattern constant:color=.1,.1,.5 512x512 3 -d uint8 -otex file.1011.tx")

command += testshade("-g 128 128 --center -scaleuv 2 2 -od uint8 -o Cout out.tif test")
outputs = [ "out.txt", "out.tif" ]
