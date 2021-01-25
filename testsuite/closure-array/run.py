#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade ("-layer alayer a -layer dlayer d --layer clayer c --layer blayer b  --connect alayer output_closure clayer in0 --connect dlayer output_closure clayer in1 --connect clayer output_closures blayer input_closures ")
