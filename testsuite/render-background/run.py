#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

failthresh = 0.01
failpercent = 1
hardfail = 0.11
allowfailures = 5
idiff_program = "idiff"

outputs = [ "out.exr" ]
command = testrender("-r 160 120 -aa 4 --options statistics:level=1 scene.xml out.exr")
