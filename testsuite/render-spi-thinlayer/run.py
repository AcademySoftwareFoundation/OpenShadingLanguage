#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

failthresh = 0.02
failrelative = 0.01
failpercent = 1

# Bright pixels (env seen through near-specular glass) vary by a few
# percent across platforms/dependencies. Forgive up to allowfailures such
# pixels, but keep hardfail low enough to catch localized regressions.
hardfail = 0.1
allowfailures = 10
idiff_program = "idiff"

outputs = [ "out.exr" ]
command = testrender("-r 160 120 -aa 16 scene.xml out.exr")
