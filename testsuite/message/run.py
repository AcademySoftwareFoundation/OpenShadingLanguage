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
command = command + "; " + path + "testshade/testshade -layer alayer a --layer blayer b --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in --connect alayer dummy blayer dummy  >> out.txt 2>&1"

# Outputs to check against references
outputs = [ "out.txt" ]

# Files that need to be cleaned up, IN ADDITION to outputs
cleanfiles = [ "a.oso", "b.oso" ]


# boilerplate
sys.path = [".."] + sys.path
import runtest
ret = runtest.runtest (command, outputs, cleanfiles)
sys.exit (ret)
