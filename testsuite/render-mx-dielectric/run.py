#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

failthresh = 0.03   # allow a little more LSB noise between platforms
hardfail = 0.05
failpercent = 1
outputs = [ "out.exr" ]
command = testrender("-v -r 320 240 -aa 16 scene.xml out.exr")
