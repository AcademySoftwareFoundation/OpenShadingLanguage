#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Test that loading an oso file with an %argderivs hint that appears before
# any op (where there is no op for it to apply to) does not crash. Found by
# fuzzing. Both shaders must load and run.

for s in [ "valid", "argderivs_before_code" ] :
    command += testshade("-g 1 1 data/" + s)
