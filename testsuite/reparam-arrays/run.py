#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade ("-g 128 128 --layer testlay -param:interactive=1 scale 5.0 test -iters 2 -reparam testlay scale 15.0 -od uint8 -o Cout out.tif")

command += testshade ("-g 128 128 --layer testlay -param:type=float[2] scale 5.0,2.0 test_array -od uint8 -o Cout out_array_iter0.tif")
command += testshade ("-g 128 128 --layer testlay -param:type=float[2]:interactive=1 scale 5.0,2.0 test_array -iters 2 -reparam:type=float[2] testlay scale 15.0,30.0 -od uint8 -o Cout out_array_iter1.tif")

command += testshade ("-g 128 128 --layer testlay -param:type=color[2] colors 5,5,5,0.5,0.5,0.5 test_colors -od uint8 -o Cout out_colors_iter0.tif")
command += testshade ("-g 128 128 --layer testlay -param:type=color[2]:interactive=1 colors 5,5,5,0.5,0.5,0.5 test_colors -iters 2 -reparam:type=color[2] testlay colors 20,20,20,1.0,0.75,0.25 -od uint8 -o Cout out_colors_iter1.tif")

outputs = [ "out.txt", "out.tif", "out_array_iter0.tif", "out_array_iter1.tif", "out_colors_iter0.tif", "out_colors_iter1.tif" ]

# expect a few LSB failures
failthresh = 0.004
failpercent = 0.05
