#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command = testshade("--res 128 128 -layer testLayer test --layer constantLayer constant --connect testLayer result constantLayer x")
