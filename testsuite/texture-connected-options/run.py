#!/usr/bin/env python

command += testshade("-g 128 128 -layer A attribs -param:type=string filename " +
                     "../common/textures/grid.tx -layer root test -connect A value " +
                     "root swidth -connect A value root twidth -od uint8 -o Cout out.tif")
outputs = [ "out.txt", "out.tif" ]

