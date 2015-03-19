#!/usr/bin/env python

failthresh = 0.005   # allow a little more LSB noise between platforms
outputs = [ "out.exr" ]
command = testrender("-r 256 256 -aa 4 cornell.xml out.exr")
