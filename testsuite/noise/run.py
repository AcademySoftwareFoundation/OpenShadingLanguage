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
command = command + "; " + path + "testshade/testshade -g 512 512 "
command = command + "-o Cout_f1 test_f1.tif "
command = command + "-o Cout_f2 test_f2.tif "
command = command + "-o Cout_f3 test_f3.tif "
command = command + "-o Cout_f4 test_f4.tif "
command = command + "-o Cout_c1 test_c1.tif "
command = command + "-o Cout_c2 test_c2.tif "
command = command + "-o Cout_c3 test_c3.tif "
command = command + "-o Cout_c4 test_c4.tif "
command = command + "test >> out.txt"

# Outputs to check against references
outputs = [ "out.txt", "test_f1.tif", "test_f2.tif", "test_f3.tif", "test_f4.tif", "test_c1.tif", "test_c2.tif", "test_c3.tif", "test_c4.tif" ]

# Files that need to be cleaned up, IN ADDITION to outputs
cleanfiles = [ ]


# boilerplate
sys.path = [".."] + sys.path
import runtest
ret = runtest.runtest (command, outputs, cleanfiles)
sys.exit (ret)
