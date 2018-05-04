#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage


command += run_app ("cmake --config Release data >> build.txt", silent=True)
command += run_app ("cmake --build . >> build.txt", silent=True)
command += run_app ("bin/osldeformer >> out.txt")
