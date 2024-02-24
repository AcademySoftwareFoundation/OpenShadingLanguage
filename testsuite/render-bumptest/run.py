#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

failthresh = 0.01
failpercent = 0.5
hardfail = 0.035

outputs = [ "out.exr" ]
command = testrender("-r 256 256 -aa 4 --llvm_opt 13 bumptest.xml out.exr")

# Note: we pick this test arbitrarily as the one to verify llvm_opt=13 works
