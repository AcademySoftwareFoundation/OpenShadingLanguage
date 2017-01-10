#!/usr/bin/env python

command = testshade("-g 512 512 -od uint8 -o Cout out.tif test")
outputs = [ "out.txt", "out.tif" ]
