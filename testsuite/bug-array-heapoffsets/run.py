#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

command = testshade("-g 64 64 --layer A textureRGB --layer B endRGB --connect A Cout B Cin  -od uint8 -o Cfinal out.tif")
outputs = [ "out.tif" ]
