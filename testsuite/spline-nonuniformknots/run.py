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
command = command + "; " + path + "testshade/testshade -g 256 64 test -od uint8 -o Cout out.tif  >> out.txt"

# Outputs to check against references
outputs = [ "out.tif" ]

# Files that need to be cleaned up, IN ADDITION to outputs
cleanfiles = [ "out.txt" ]


# boilerplate
sys.path = [".."] + sys.path
import runtest
ret = runtest.runtest (command, outputs, cleanfiles)
sys.exit (ret)
