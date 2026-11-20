#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Test that loading an oso file containing an op whose jump target is out
# of range (past the end of the code, or not after the op itself) is
# rejected with an error, rather than later indexing op-sized arrays out of
# bounds during runtime optimization. Found by fuzzing. The "valid" shader
# has legal forward jump targets (including one to the final op) and
# must still load and run.

for s in [ "valid", "jump_past_end", "jump_backward", "jump_to_self" ] :
    command += testshade("-g 1 1 data/" + s)
