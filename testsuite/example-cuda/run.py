#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


command += run_app("cmake --config Release data >> build.txt 2>&1", silent=True)
command += run_app("cmake --build . >> build.txt 2>&1", silent=True)
command += run_app("bin/example-cuda test_add >> out.txt")
