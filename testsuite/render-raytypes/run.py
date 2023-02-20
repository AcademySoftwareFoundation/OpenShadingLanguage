#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# This scene tests the raytype() function by making a shader that appears
# differently to camera rays vs indirect rays.

failthresh = 0.01
failpercent = 1
hardfail = 0.025

outputs = [ "out.exr" ]
command = testrender("-v -r 100 75 -aa 2 scene.xml out.exr")
