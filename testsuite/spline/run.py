#!/usr/bin/python 

import os
import sys

path = ""
command = ""
if len(sys.argv) > 2 :
    os.chdir (sys.argv[1])
    path = sys.argv[2] + "/"

# A command to run
command = path + "oslc/oslc test.osl > out.txt"
command = command + "; " + path + "testshade/testshade -g 256 256 -od uint8 -o Cspline color.tif -o DxCspline dcolor.tif -o Fspline float.tif -o DxFspline dfloat.tif test >> out.txt"

# Outputs to check against references
outputs = [ "out.txt", "color.tif", "dcolor.tif", "float.tif", "dfloat.tif" ]

# Files that need to be cleaned up, IN ADDITION to outputs
cleanfiles = [ ]


# boilerplate
sys.path = [".."] + sys.path
import runtest
ret = runtest.runtest (command, outputs, cleanfiles)
sys.exit (ret)
