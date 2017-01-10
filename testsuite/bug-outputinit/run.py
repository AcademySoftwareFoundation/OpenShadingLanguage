#!/usr/bin/env python

# See the test.osl source for explanation of this test

command = testshade("-g 16 16 test -od uint8 -o Cout out.tif")
outputs = [ "out.tif" ]
