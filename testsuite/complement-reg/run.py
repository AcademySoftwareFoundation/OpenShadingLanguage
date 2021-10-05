#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


command += testshade("-t 1 -g 32 32 -od uint8 v_complement -o cout v_comp.tif -o mcout mv_comp.tif")
outputs.append ("v_comp.tif")
outputs.append ("mv_comp.tif")


command += testshade("-t 1 -g 32 32 -od uint8 u_complement -o cout u_comp.tif -o mcout mu_comp.tif")
outputs.append ("u_comp.tif")
outputs.append ("mu_comp.tif")

# expect a few LSB failures
failthresh = 0.008
failpercent = 3

