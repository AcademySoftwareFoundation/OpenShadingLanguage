#!/usr/bin/python 

command = testshade("-g 64 64 -od uint8 -o Cout out.tif test")
outputs = [ "out.txt", "out.tif" ]

# expect a few LSB failures
failthresh = 0.008
failpercent = 3
