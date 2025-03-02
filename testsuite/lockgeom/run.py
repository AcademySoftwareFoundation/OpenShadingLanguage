#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# The tests are similar but have different default values of face_idx 
# to make sure batched analysis doesn't force interpolated integer 
# argements to be boolean.
command = testshade("-t 1 --res 256 256 -od uint8 -o dst out0.tif test0")
command += testshade("-t 1 --res 256 256 -od uint8 -o dst out1.tif test1")
command += testshade("-t 1 --res 256 256 -od uint8 -o dst out2.tif test2")
command += testshade("-t 1 --res 256 256 -od uint8 -o dst out3.tif test3")
outputs.append ("out0.tif")
outputs.append ("out1.tif")
outputs.append ("out2.tif")
outputs.append ("out3.tif")
