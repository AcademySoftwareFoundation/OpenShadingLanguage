#!/usr/bin/python 

import os
import sys

path = ""
command = ""
if len(sys.argv) > 2 :
    os.chdir (sys.argv[1])
    path = sys.argv[2] + "/"

# A command to run
command = path + "oslc/oslc ../common/shaders/testpnoise.osl > out.txt"
command = command + "; " + path + "testshade/testshade -g 512 512 -od uint8 "
command = command + "-o Cout out.tif "
command = command + "-sparam noisename cell -fparam offset 0 -fparam scale 1 testpnoise >> out.txt"

# Outputs to check against references
outputs = [ "out.txt", "out.tif" ]

# Files that need to be cleaned up, IN ADDITION to outputs
cleanfiles = [ ]


# boilerplate
sys.path = [".."] + sys.path
import runtest
ret = runtest.runtest (command, outputs, cleanfiles, failthresh=0.004, failpercent=.05)
sys.exit (ret)
