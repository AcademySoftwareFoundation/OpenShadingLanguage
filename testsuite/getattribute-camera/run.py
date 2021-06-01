# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage
#!/usr/bin/env python 

# Simple test on a grid texture
command = testshade("-t 1 -g 1 1 test")
command += testshade("-t 1 -g 2 2 test_v_name")
