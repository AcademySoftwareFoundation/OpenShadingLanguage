#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

command += testshade("-g 128 128 -layer A attribs -param:type=string filename " +
                     "../common/textures/grid.tx -layer root test -connect A value " +
                     "root swidth -connect A value root twidth -od uint8 -o Cout out.tif")
outputs = [ "out.txt", "out.tif" ]

