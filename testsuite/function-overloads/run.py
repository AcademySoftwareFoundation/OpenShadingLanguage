#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

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

