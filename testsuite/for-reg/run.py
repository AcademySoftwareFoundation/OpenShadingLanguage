#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


command += testshade("-t 1 --center -g 256 256 cyclic_read_in_loop_v1 -od uint8 -o rgb scyclic_read_in_loop_v1.tif")
command += testshade("-t 1 --center -g 256 256 cyclic_read_in_loop_v2 -od uint8 -o rgb scyclic_read_in_loop_v2.tif")
command += testshade("-t 1 --center -g 256 256 cyclic_read_in_loop_v3 -od uint8 -o rgb scyclic_read_in_loop_v3.tif")
command += testshade("-t 1 --center -g 256 256 cyclic_read_in_loop_v4 -od uint8 -o rgb scyclic_read_in_loop_v4.tif")
command += testshade("-t 1 --center -g 256 256 cyclic_read_in_loop_v5 -od uint8 -o rgb scyclic_read_in_loop_v5.tif")


outputs = [ 
    "scyclic_read_in_loop_v1.tif",
    "scyclic_read_in_loop_v2.tif",
    "scyclic_read_in_loop_v3.tif",
    "scyclic_read_in_loop_v4.tif",
    "scyclic_read_in_loop_v5.tif",
]


# expect a few LSB failures
failthresh = 0.008
failpercent = 3
