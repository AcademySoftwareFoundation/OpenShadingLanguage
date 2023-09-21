#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# NOTE: goal is to have non-full batch, thus the resulting (width*height%8 != 0)
#       to verify partial user data from the BatchedRendererServices are handled
#       correctly as some data lanes will be populated by userdata and others
#       populated by default value specified in the shader
command += testshade("-g 11 12 --center -options lazy_userdata=0 -od uint8 -o Cout out.lazy_userdata_OFF.tif test")
command += testshade("-g 11 12 --center -options lazy_userdata=1 -od uint8 -o Cout out.lazy_userdata_ON.tif test")
outputs = [ "out.txt", "out.lazy_userdata_OFF.tif", "out.lazy_userdata_ON.tif" ]