#!/usr/bin/python 

failthresh = 0.0055   # allow a little more LSB noise between platforms
outputs = [ "out.exr" ]
command = testrender("-r 320 240 -aa 4 scene.xml out.exr")
