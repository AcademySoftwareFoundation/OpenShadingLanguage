#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#NOTE:  its really difficult to test isnan(value) == true, 
# as its unclear to actually generate a NAN in OSL because 
# most functions are "safe" and check/avoid NAN's
  
command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_ieee_fp_div.tif test_ieee_fp_div")
outputs.append ("out_ieee_fp_div.tif")

command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_ieee_fp_sqrt.tif test_ieee_fp_sqrt")
outputs.append ("out_ieee_fp_sqrt.tif")

command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_ieee_fp_log.tif test_ieee_fp_log")
outputs.append ("out_ieee_fp_log.tif")

command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_ieee_fp_log2.tif test_ieee_fp_log2")
outputs.append ("out_ieee_fp_log2.tif")

command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_ieee_fp_log10.tif test_ieee_fp_log10")
outputs.append ("out_ieee_fp_log10.tif")

command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_ieee_fp_logbase.tif test_ieee_fp_logbase")
outputs.append ("out_ieee_fp_logbase.tif")

command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_ieee_fp_logb.tif test_ieee_fp_logb")
outputs.append ("out_ieee_fp_logb.tif")

command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_ieee_fp_asin.tif test_ieee_fp_asin")
outputs.append ("out_ieee_fp_asin.tif")

command += testshade("-t 1 -g 64 64 -od uint8 -o Cout out_ieee_fp_acos.tif test_ieee_fp_acos")
outputs.append ("out_ieee_fp_acos.tif")



# expect a few LSB failures
failthresh = 0.008
failpercent = 3

