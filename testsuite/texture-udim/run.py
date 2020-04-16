#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

#command += oiiotool ("-pattern constant:color=.5,.1,.1 256x256 3 -text:size=50:x=75:y=140 1001 -d uint8 -otex file.1001.tx")
command += oiiotool ("-pattern constant:color=.1,.5,.1 256x256 3 -text:size=50:x=75:y=140 1002 -d uint8 -otex file.1002.tx")
command += oiiotool ("-pattern constant:color=.1,.1,.5 256x256 3 -text:size=50:x=75:y=140 1011 -d uint8 -otex file.1011.tx")
#command += oiiotool ("-pattern constant:color=.1,.5,.5 256x256 3 -text:size=50:x=75:y=140 1012 -d uint8 -otex file.1012.tx")

# Purposely only create two of the four UDIM textures

command += testshade("-g 128 128 --center -scaleuv 2 2 -od uint8 -o Cout out.tif test")
outputs = [ "out.txt", "out.tif" ]
