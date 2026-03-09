#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

failpercent = 1
failthresh = 0.05   # allow a little more LSB noise between platforms
hardfail = 0.05

outputs = [ "out.exr" ]
command = testrender("-r 320 240 -aa 8 scene.xml out.exr")
