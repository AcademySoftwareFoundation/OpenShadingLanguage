#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

# Use -MMD instead of -MD to cause it to not list stdosl.h, which with its
# absolute path is very hard to have match across CI platforms and test
# exactly.

# Test deps to default test.d
command = oslc ("-q -MMD test.osl")

# Test deps to custom file location
command += oslc ("-q -MMD -MFmydep.d test.osl")

# Test deps to stdout
command += oslc ("-MM test.osl")

# Test deps with custom target
command += oslc ("-q -MMD -MF mycustom.d -MT customtarget test.osl")

outputs = [ "test.d", "mydep.d", "mycustom.d", "out.txt" ]

