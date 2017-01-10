#!/usr/bin/env python

command += testshade ("-g 128 128 --layer testlay -param:lockgeom=0 scale 5.0 test -iters 2 -reparam testlay scale 15.0 -od uint8 -o Cout out.tif")
outputs = [ "out.txt", "out.tif" ]
# expect a few LSB failures
failthresh = 0.004
failpercent = 0.05
