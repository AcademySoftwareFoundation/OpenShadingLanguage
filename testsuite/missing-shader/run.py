#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

failureok = True    # Expect an error
command = testshade("-g 2 2 --layer lay1 foo --layer lay2 bar --connect lay1 x lay2 y")
