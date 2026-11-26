#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Test that loading an oso file containing an op with the wrong number of
# args is rejected with an error, rather than later reading nonexistent
# args during constant folding or code generation. Found by fuzzing. Covers
# too few and too many args for a fixed-arity op, too many for an op with a
# range, and too few for a variadic op. The "valid" shader must still load
# and run.

for s in [ "valid", "too_few_args", "too_many_args", "range_too_many",
           "variadic_too_few" ] :
    command += testshade("-g 1 1 data/" + s)
