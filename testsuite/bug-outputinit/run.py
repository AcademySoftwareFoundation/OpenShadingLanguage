#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# See the test.osl source for explanation of this test

command = testshade("-g 16 16 test -od uint8 -o Cout out.tif")
outputs = [ "out.tif" ]
