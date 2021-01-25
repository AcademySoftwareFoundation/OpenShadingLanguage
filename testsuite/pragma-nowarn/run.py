#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# This shader would ordinarily issue a warning.
# With -Werror, it should be upgraded to an error.
oslcargs = "-Werror"

# BUT... the shader carefully uses #pragma nowarn to disable the warning.
# Which should cause the test to pass.

command = testshade("test")

