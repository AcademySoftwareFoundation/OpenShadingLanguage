#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

# This shader would ordinarily issue a warning.
# With -Werror, it should be upgraded to an error.

failureok = 1     # this test is expected to have oslc errors
oslcargs = "-Werror"

# No need, the shader in this dir are always compiled
#command = oslc("test.osl")

