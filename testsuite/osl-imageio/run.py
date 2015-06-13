#!/usr/bin/env python

# Various tests of our osl plugin for OIIO

# Test straightforward oso execution
command += oiiotool ('"ramp.oso?RES=64" -d uint8 -o ramp-oso-default.tif')

# Test parameter override
command += oiiotool ('"ramp.oso?RES=64&color topright=0,0,1" -d uint8 -o ramp-oso-blue.tif')

# Test tiles
command += oiiotool ('"ramp.oso?RES=64&color bottomright=0,1,1&TILE=32x32" -d uint8 -o ramp-oso-tiles.tif')


# Test mip and also oslbody
command += oiiotool ('"result=sin(40*s)/2+0.5.oslbody?RES=256x256&MIP=1" -selectmip 2 -d uint8 -o wave-mip.tif')


outputs = [ "out.txt",
            "ramp-oso-default.tif",
            "ramp-oso-blue.tif",
            "ramp-oso-tiles.tif",
            "wave-mip.tif",
          ]

