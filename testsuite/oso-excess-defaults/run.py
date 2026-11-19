#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Test that loading an oso file whose params or consts list more initial
# values than the declared type holds is rejected with an error rather than
# writing out of bounds. The "valid" shader exercises legal cases (scalar,
# aggregate, unsized array, fixed array) that must still load.

for s in [ "valid",
           "param_int", "param_float_many", "param_array", "param_string",
           "const_int", "const_float", "const_string" ] :
    command += testshade("-g 1 1 data/" + s)
