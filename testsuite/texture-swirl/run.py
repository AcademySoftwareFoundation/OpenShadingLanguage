#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

command += testshade("-g 512 512 --center --param swirl 2.0 -od uint8 -o Cout out.tif swirl")
outputs = [ "out.txt", "out.tif" ]
