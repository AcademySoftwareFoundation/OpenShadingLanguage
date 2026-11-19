#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Test that loading an oso file with a %structfields hint on a symbol that
# is not a struct is rejected with an error, rather than dereferencing a
# null struct description. Found by fuzzing. The "valid" shader has a
# legal struct parameter and must still load.

for s in [ "valid", "structfields_on_float" ] :
    command += testshade("-g 1 1 data/" + s)
