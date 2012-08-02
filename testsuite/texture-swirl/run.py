#!/usr/bin/python 

command += testshade("-g 512 512 --center --fparam swirl 2 -od uint8 -o Cout out.tif swirl")
outputs = [ "out.txt", "out.tif" ]
