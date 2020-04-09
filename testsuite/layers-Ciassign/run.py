#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

command += testshade("--options lazyglobals=0 -layer alayer a --layer blayer b --connect alayer out blayer in")
