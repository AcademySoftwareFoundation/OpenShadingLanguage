#!/usr/bin/env python

command += testshade ("-g 1000 64 -od half -o Cout out.exr test")
outputs += [ "out.exr" ]

# Allow some per-platform numerical slop
failthresh = 0.004
failpercent = 0.05
