#!/usr/bin/python 

command = oslc("../common/shaders/testnoise.osl")
command += testshade ("-g 512 512 -od uint8 -o Cout out.tif -sparam noisename cell -fparam offset 0 -fparam scale 1 testnoise")
outputs = [ "out.txt", "out.tif" ]
