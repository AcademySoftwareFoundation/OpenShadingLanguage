#!/usr/bin/python 

command = oslc("../common/shaders/testpnoise.osl")
command += testshade("-g 512 512 -od uint8 -o Cout out.tif -sparam noisename gabor testpnoise")
outputs = [ "out.txt", "out.tif" ]
