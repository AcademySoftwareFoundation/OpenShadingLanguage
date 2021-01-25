#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += maketx("../common/textures/mandrill.tif --wrap clamp -o mandrill.tx")
command += testshade("-g 64 64 --center -od uint8 -o Cblack black.tif -o Cclamp clamp.tif -o Cperiodic periodic.tif -o Cmirror mirror.tif -o Cdefault default.tif test")

outputs = [ "out.txt", "black.tif", "clamp.tif", "periodic.tif", "mirror.tif",
            "default.tif" ]
