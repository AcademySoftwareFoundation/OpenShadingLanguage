#!/usr/bin/env python

command += testshade("-g 1 1 --center -od uint8 -o Cout out.tif test")
outputs = [ "out.txt", "out.tif" ]
