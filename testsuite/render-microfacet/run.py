#!/usr/bin/env python

failthresh = 0.01
failpercent = 0.01
hardfail = 0.026
outputs = [ "out.exr" ]
command = testrender("-r 320 240 -aa 8 scene.xml out.exr")
