#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

# Here we have a parameter 'a' that's a color, and we are passing just
# a float, and a parameter 'p' that's a point but we're passing a vector.
# Thexe used to be errors, but now we want to accept it.
# Test to make sure it works.

command = testshade("--param a 10.0 -param:type=vector p 42,21,7 test")
