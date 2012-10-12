#!/usr/bin/python 

command += testshade("-g 16 16 -od uint8 -o Cout out0.tif wrcloud")
command += testshade("-g 256 256 -fparam radius 0.01 -od uint8 -o Cout out1.tif rdcloud")
command += testshade("-g 256 256 -fparam radius 0.1 -od uint8 -o Cout out2.tif rdcloud")
outputs = [ "out0.tif", "out1.tif", "out2.tif" ]
