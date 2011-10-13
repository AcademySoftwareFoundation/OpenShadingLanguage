#!/usr/bin/python 

import os
import sys

path = ""
command = ""
if len(sys.argv) > 2 :
    os.chdir (sys.argv[1])
    path = sys.argv[2] + "/"

# A command to run
command = path + "oslc/oslc a.osl > out.txt"
command = command + "; " + path + "oslc/oslc b.osl >> out.txt"
command = command + "; " + path + "testshade/testshade --layer alayer a --layer blayer b --connect alayer st_out.s blayer x.s --connect alayer st_out.t blayer x.t --connect alayer st_out blayer y --connect alayer r blayer r >> out.txt"

# Outputs to check against references
outputs = [ "out.txt" ]

# Files that need to be cleaned up, IN ADDITION to outputs
cleanfiles = [ ]


# boilerplate
sys.path = [".."] + sys.path
import runtest
ret = runtest.runtest (command, outputs, cleanfiles)
sys.exit (ret)
