# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

from __future__ import absolute_import

import os

if os.path.isfile("out_fileprint.txt") :
    os.remove ("out_fileprint.txt")

command = testshade("test_fileprint")

outputs = [ "out_fileprint.txt" ]

