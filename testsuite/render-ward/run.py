#!/usr/bin/env python

failthresh = 0.006
failpercent = 0.35
outputs = [ "out.exr" ]
command = testrender("-r 320 240 -aa 4 scene.xml out.exr")
