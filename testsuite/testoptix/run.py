#!/usr/bin/env python

failthresh = 0.03   # allow a little more LSB noise between platforms
failpercent = .5
outputs  = [ "out.exr", "test_texture.exr", "out.txt" ]
command  = testoptix("-res 320 240 scene.xml out.exr")
command += testoptix("-res 1 1 test_print.xml dummy.exr")
command += testoptix("-res 1 1 test_compare.xml dummy.exr")
command += testoptix("-res 1 1 test_assign.xml dummy.exr")
command += testoptix("-res 1 1 test_assign_02.xml dummy.exr")
command += testoptix("-res 1 1 test_str_ops.xml dummy.exr")
command += testoptix("-res 1 1 test_userdata_string.xml dummy.exr")
command += testoptix("-res 512 512 test_texture.xml test_texture.exr")
