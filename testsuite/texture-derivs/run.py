#!/usr/bin/env python

command += testshade("-g 128 128 --center -od uint8 -o Cout out.tif -o dx dx.tif -o dy dy.tif -param scale 128.0 test")
outputs = [ "out.txt", "out.tif", "dx.tif", "dy.tif" ]
