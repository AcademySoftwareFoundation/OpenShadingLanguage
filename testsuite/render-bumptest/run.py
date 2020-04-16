#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

outputs = [ "out.exr" ]
command = testrender("-r 256 256 -aa 4 bumptest.xml out.exr")
