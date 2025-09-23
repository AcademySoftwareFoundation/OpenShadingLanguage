#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

failthresh = 0.01
failpercent = 1
allowfailures = 5
idiff_program = "idiff"

outputs = [ "out.exr" ]
command = testrender("-v -r 320 240 -aa 16 scene.xml out.exr")
