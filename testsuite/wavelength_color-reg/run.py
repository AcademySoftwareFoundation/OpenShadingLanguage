#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


command += testshade("-g 200 200 wavelength_v_lambda -od uint8 -o Cout v_lambda.tif")
outputs.append ("v_lambda.tif")

command += testshade("-g 200 200 wavelength_u_lambda -od uint8 -o Cout u_lambda.tif")
outputs.append ("u_lambda.tif")

command += testshade("-g 200 200 wavelength_u_lambda_masked -od uint8 -o Cout u_lambda_masked.tif")
outputs.append ("u_lambda_masked.tif")

command += testshade("-g 200 200 wavelength_v_lambda_masked -od uint8 -o Cout v_lambda_masked.tif")
outputs.append ("v_lambda_masked.tif")

# expect a few LSB failures
failthresh = 0.008
failpercent = 3

