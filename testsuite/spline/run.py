#!/usr/bin/env python

command += testshade("-g 256 256 -od uint8 -o Cspline color.tif -o DxCspline dcolor.tif -o Fspline float.tif -o DxFspline dfloat.tif -o NumKnots numknots.tif test")

command += testshade("-g 256 256 -od uint8 -o Cspline constcolor.tif constspline")

outputs = [ "out.txt", "color.tif", "dcolor.tif", "float.tif", "dfloat.tif", "numknots.tif", "constcolor.tif" ]
