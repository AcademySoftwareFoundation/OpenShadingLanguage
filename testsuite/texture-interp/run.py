#!/usr/bin/env python

command += testshade("-g 256 256 --center -od uint8 -o Cout out.tif test")
outputs = [ "out.txt", "out.tif" ]
