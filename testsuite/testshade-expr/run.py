#!/usr/bin/env python

command += testshade ("-v -g 64 64 -od uint8 -o result out.tif -expr 'result=color(u,v,0)'")
outputs = [ "out.txt", "out.tif" ]
