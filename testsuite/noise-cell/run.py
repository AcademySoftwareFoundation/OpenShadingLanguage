#!/usr/bin/env python

command = oslc("../common/shaders/testnoise.osl")
command += testshade ("-g 512 512 -od uint8 -o Cout out.tif -param noisename cell -param offset 0.0 -param scale 1.0 testnoise")
outputs = [ "out.txt", "out.tif" ]
