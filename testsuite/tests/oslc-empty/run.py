#!/usr/bin/python 

# A command to run
command = "oslc -d test.osl > out.txt"

# Outputs to check against references
outputs = [ "out.txt" ]

# Files that need to be cleaned up, IN ADDITION to outputs
cleanfiles = [ ]


# boilerplate
import sys
sys.path = ["../.."] + sys.path
import runtest
ret = runtest.runtest (command, outputs, cleanfiles)
exit (ret)
