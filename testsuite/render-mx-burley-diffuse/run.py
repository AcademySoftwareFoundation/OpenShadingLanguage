#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

failthresh = 0.01
failpercent = 1
hardfail = 0.016

outputs = [ "out.exr" ]
command = testrender("-r 320 240 -aa 4 scene.xml out.exr")
