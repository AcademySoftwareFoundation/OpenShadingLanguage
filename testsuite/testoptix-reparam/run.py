#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

failthresh = 0.03   # allow a little more LSB noise between platforms
failpercent = .5
outputs  = [ "out.exr" ]
command  = testrender("-optix -res 320 240 scene.xml out.exr")
