#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

command += testshade ('-v -g 64 64 -od uint8 -o result out.tif -expr "result=color(u,v,0)"')
outputs = [ "out.txt", "out.tif" ]
