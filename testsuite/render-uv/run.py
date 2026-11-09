#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

failthresh = 0.013
failpercent = 1

# A few isolated pixels (fireflies) differ across platforms/GPU. Forgive
# that many via idiff's allowfailures, keeping hardfail strict so
# localized regressions are still caught.
hardfail = 0.025
allowfailures = 3
idiff_program = "idiff"

outputs = [ "out.exr" ]
command = testrender("-r 160 120 -aa 16 scene.xml out.exr")
