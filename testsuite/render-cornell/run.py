#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

failthresh = 0.01
failpercent = 1

# A few isolated pixels (fireflies) differ across platforms/GPU. Forgive
# that many via idiff's allowfailures, keeping hardfail strict so
# localized regressions are still caught.
hardfail = 0.012
allowfailures = 2
idiff_program = "idiff"

outputs = [ "out.exr" ]
command = testrender("-r 128 128 -aa 4 --llvm_opt 12 cornell.xml out.exr")

# Note: we pick this test arbitrarily as the one to verify llvm_opt=12 works
