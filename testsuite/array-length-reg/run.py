#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade("-t 1 -g 64 64 -od uint8 v_float -o cout vfloat.tif -o mcout mvfloat.tif")
command += testshade("-t 1 -g 64 64 -od uint8 u_float -o cout ufloat.tif -o mcout mufloat.tif")
command += testshade("-t 1 -g 64 64 -od uint8 v_vector -o cout vvector.tif -o mcout mvvector.tif")
command += testshade("-t 1 -g 64 64 -od uint8 u_vector -o cout uvector.tif -o mcout muvector.tif")
command += testshade("-t 1 -g 64 64 -od uint8 v_string -o cout vstring.tif -o mcout mvstring.tif")
command += testshade("-t 1 -g 64 64 -od uint8 u_string -o cout ustring.tif -o mcout mustring.tif")

outputs = [ 
    "vfloat.tif",
    "mvfloat.tif",
    "ufloat.tif",
    "mufloat.tif",
    "vvector.tif",
    "mvvector.tif",
    "uvector.tif",
    "muvector.tif",
    "vstring.tif",
    "mvstring.tif",
    "ustring.tif",
    "mustring.tif"
]


# expect a few LSB failures
failthresh = 0.008
failpercent = 3









