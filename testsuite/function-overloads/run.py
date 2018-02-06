#!/usr/bin/env python

realruntest = runtest

def runtest (command, *args, **kwargs) :
    passed = True
    for arg in  ("-DORDER_1 ", ""):
        command =  oslc(arg + "test.osl")
        command += testshade("-g 1 1 test")
        if realruntest(command, *args, **kwargs):
            passed = False
    return not passed

command = ""

