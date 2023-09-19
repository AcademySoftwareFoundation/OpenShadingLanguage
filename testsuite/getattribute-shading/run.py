#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Simple test on a grid texture
command = testshade("-t 1 -g 2 2 --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1 test")
