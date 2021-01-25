#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

outputs = [ "out.exr" ]
command = testrender("-r 320 240 -aa 4 veach.xml out.exr")
