#!/usr/bin/env python

failthresh = 0.005
failpercent = 0.1
outputs = [ "out.exr" ]
command = testrender("-r 320 240 -aa 8 scene.xml out.exr")
