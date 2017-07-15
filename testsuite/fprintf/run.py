#!/usr/bin/env python

import os

if os.path.isfile("data.txt") :
    os.remove ("data.txt")

command = testshade("test")

outputs = [ "data.txt" ]
