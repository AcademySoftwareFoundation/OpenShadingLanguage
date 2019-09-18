#!/usr/bin/env python

failthresh = 0.03   # allow a little more LSB noise between platforms
failpercent = .5
outputs  = [ "out.exr", "test_microfacet_dist.exr", "test_texture.exr", "test_spline.exr", "out.txt" ]
command  = testrender("-optix -res 320 240 scene.xml out.exr")
command += testrender("-optix -res 320 240 test_microfacet_dist.xml test_microfacet_dist.exr")
command += testrender("-optix -res 1 1 test_print.xml dummy.exr")
command += testrender("-optix -res 1 1 test_compare.xml dummy.exr")
command += testrender("-optix -res 1 1 test_assign.xml dummy.exr")
command += testrender("-optix -res 1 1 test_assign_02.xml dummy.exr")
command += testrender("-optix -res 1 1 test_str_ops.xml dummy.exr")
command += testrender("-optix -res 1 1 test_userdata_string.xml dummy.exr")

command += testshade("-optix -res 256 256 test_spline -o Cout test_spline.exr")
command += testshade("-optix -res 512 512 test_texture -o Cout test_texture.exr")
