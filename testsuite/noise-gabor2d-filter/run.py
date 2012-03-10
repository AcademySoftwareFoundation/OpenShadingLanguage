#!/usr/bin/python 

command += testshade("-g 512 512 -od uint8 -o Cout out.tif test")
outputs = [ "out.txt", "out.tif" ]
# expect a few LSB failures
failthresh = 0.004
failpercent = 0.05
