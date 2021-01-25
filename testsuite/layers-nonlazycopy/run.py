#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Example that failed with a prior bug.
# The test case is that layer a modifies Ci so is unconditional, but also
# its output CCout is connected to layer b's CCin. There was a bug where
# the earlier layer was run and its values copied before (duh, not after)
# b's params would be initialized, thus overwriting the copied values.


command += testshade("-layer alayer a -layer blayer b --connect alayer CCout blayer CCin")

