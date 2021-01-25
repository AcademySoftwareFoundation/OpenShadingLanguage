#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

failthresh = 0.005
failpercent = 0.1
outputs = [ "out.exr" ]
command = testrender("-r 320 240 -aa 8 scene.xml out.exr")
