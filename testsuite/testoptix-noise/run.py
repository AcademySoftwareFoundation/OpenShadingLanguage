#!/usr/bin/env python

failthresh = 0.03   # allow a little more LSB noise between platforms
failpercent = .5
outputs = [ "out.exr" ]
command = testoptix("-r 320 240 scene.xml out.exr")
