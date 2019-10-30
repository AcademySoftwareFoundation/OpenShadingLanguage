#!/usr/bin/env python

from __future__ import absolute_import

import os

if os.path.isfile("data.txt") :
    os.remove ("data.txt")

command = testshade("test")

outputs = [ "data.txt" ]
