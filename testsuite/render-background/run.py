#!/usr/bin/python 

failthresh = 0.005   # allow a little more LSB noise between platforms
failpercent = 1
outputs = [ "out.exr" ]
command = testrender("-r 320 240 -aa 4 scene.xml out.exr")
