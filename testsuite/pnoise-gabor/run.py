#!/usr/bin/env python

command = oslc("../common/shaders/testpnoise.osl")
command += testshade("-g 512 512 -od uint8 -o Cout out.tif -param noisename gabor testpnoise")
outputs = [ "out.txt", "out.tif" ]
