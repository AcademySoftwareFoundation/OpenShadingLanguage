#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


outputs = [ "out.exr" ]
command += run_app("cmake -DCMAKE_BUILD_TYPE=Release data >> build.txt 2>&1", silent=True)
command += run_app("cmake --build . >> build.txt 2>&1", silent=True)
command += run_app("bin/tinyrender >> out.txt")
