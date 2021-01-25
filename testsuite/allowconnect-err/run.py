#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

failureok = True    # Expect an error
command += testshade("-layer alayer a --layer blayer b --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in")
outputs = [ "out.txt" ]
