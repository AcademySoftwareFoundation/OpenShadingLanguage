#!/usr/bin/env python

command = testshade("-g 64 64 --layer A textureRGB --layer B endRGB --connect A Cout B Cin  -od uint8 -o Cfinal out.tif")
outputs = [ "out.tif" ]
