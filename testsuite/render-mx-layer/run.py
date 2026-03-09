#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

failthresh = 0.03   # allow a little more LSB noise between platforms
hardfail = 0.05
idiff_program = "idiff"

outputs = [ "out.exr" ]
command = testrender("-r 320 240 -aa 6 scene.xml out.exr")
