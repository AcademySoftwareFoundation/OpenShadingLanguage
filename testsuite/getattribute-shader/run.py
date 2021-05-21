#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Simple test on a grid texture
command = testshade("-t 1 -groupname Beatles -layer Cake test")
command += testshade("-t 1 -g 2 2 -groupname Beatles -layer Cake test_v_name")
