#!/usr/bin/python 

command = oslc("../common/shaders/testnoise.osl")
command += testshade ("-g 512 512 -od uint8 -o Cout out.tif -sparam noisename usimplex -fparam offset 0 -fparam scale 1 testnoise")
outputs = [ "out.txt", "out.tif" ]
# expect some LSB failures on this test
failthresh = 0.004
failpercent = 0.05
