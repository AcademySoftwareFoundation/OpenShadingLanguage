#!/usr/bin/python 

command += testshade("-g 64 64 --center -od uint8 -o Cblack black.tif -o Cclamp clamp.tif -o Cperiodic periodic.tif -o Cmirror mirror.tif test")

outputs = [ "out.txt", "black.tif", "clamp.tif", "periodic.tif", "mirror.tif" ]
