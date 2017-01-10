#!/usr/bin/env python

command = testshade("-g 256 64 -od uint8 -o Cout out.tif test")
outputs = [ "out.tif", "out.txt" ]
