#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade("-g 2 2 -debug printbackfacing")
command += testshade("-g 2 2 -debug printcalculatenormal")

# debug output is very verbose, use regexp to filter down to
# only the line we're interested in testing
filter_re = "Need.*globals.*"
