#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage



command += testshade("-g 16 16 -od uint8 -o Cout out0.tif wrcloud")
command += testshade("-g 16 16 -od uint8 -o Cout out0_transpose.tif wrcloud_transpose")
command += testshade("-g 16 16 -od uint8 -o Cout out0_varying_filename.tif wrcloud_varying_filename")

command += testshade("-g 256 256 -param radius 0.01 -od uint8 -o Cout out1.tif rdcloud")
command += testshade("-g 256 256 -param radius 0.01 -param filename cloud_masked_1.geo -od uint8 -o Cout out1_masked_1.tif rdcloud")
command += testshade("-g 256 256 -param radius 0.01 -param filename cloud_masked_2.geo -od uint8 -o Cout out1_masked_2.tif rdcloud")

command += testshade("--vary_pdxdy -g 256 256 -t 1 -param radius 0.01 -od uint8 -o Cout out_zero_derivs.tif rdcloud_zero_derivs")
command += testshade("--vary_pdxdy -g 256 256 -t 1 -param radius 0.01 -od uint8 -o Cout out_zero_derivs_search_only.tif rdcloud_zero_derivs_search_only")
command += testshade("--vary_pdxdy -g 256 256 -t 1 -param radius 0.01 -od uint8 -o Cout out_search_only.tif rdcloud_search_only")

command += testshade("-g 256 256 -param radius 0.1 -od uint8 -o Cout out2.tif rdcloud")

command += testshade("--vary_pdxdy -g 256 256 -t 1 -param radius 0.01 -od uint8 -o Cout out_rdcloud_varying_filename.tif rdcloud_varying_filename")
command += testshade("--center --vary_pdxdy -g 256 256 -t 1 -param radius 0.1 -od uint8 -o Cout out_rdcloud_varying_maxpoint.tif rdcloud_varying_maxpoint")
command += testshade("--center --vary_pdxdy -g 256 256 -t 1 -param radius 0.1 -od uint8 -o Cout out_rdcloud_varying_sort.tif rdcloud_varying_sort")

command += testshade("--vary_pdxdy -g 256 256 -t 1 -param radius 0.01 -od uint8 -o Cout out_rdcloud_get_varying_filename.tif rdcloud_get_varying_filename")

command += testshade("--center --vary_pdxdy -g 256 256 -t 1 -param radius 0.1 -od uint8 -o Cout out_rdcloud_varying.tif rdcloud_varying")
command += testshade("--center --vary_pdxdy -g 256 256 -t 1 -param radius 0.1 -od uint8 -o Cout out_rdcloud_varying_no_index.tif rdcloud_varying_no_index")
command += testshade("--center --vary_pdxdy -g 256 256 -t 1 -param radius 0.1 -od uint8 -o Cout out_rdcloud_varying_mismatch.tif rdcloud_varying_mismatch")

outputs = [ "out0.tif" ]
outputs += [ "out0_transpose.tif" ]
outputs += [ "out0_varying_filename.tif" ]

outputs += [ "out1.tif" ]
outputs += [ "out1_masked_1.tif" ]
outputs += [ "out1_masked_2.tif" ]
outputs += [ "out_zero_derivs.tif" ]
outputs += [ "out_search_only.tif" ]
outputs += [ "out_zero_derivs_search_only.tif" ]

outputs += [ "out2.tif" ]

outputs += [ "out_rdcloud_varying_filename.tif" ]
outputs += [ "out_rdcloud_varying_maxpoint.tif" ]
outputs += [ "out_rdcloud_varying_sort.tif" ]
outputs += [ "out_rdcloud_varying.tif" ]
outputs += [ "out_rdcloud_varying_no_index.tif" ]
outputs += [ "out_rdcloud_varying_mismatch.tif" ]

outputs += [ "out_rdcloud_get_varying_filename.tif" ]


# expect a few LSB failures
failthresh = 0.008
failpercent = 3
