#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

failthresh = 0.01   # allow a little more LSB noise between platforms
hardfail = 0.025
outputs = [ "out.exr" ]
command = testrender("-r 384 64 -aa 16 scene.xml out.exr")
