#!/usr/bin/python 

outputs = [ "out.exr" ]
command = testrender("-r 256 256 -aa 4 bumptest.xml out.exr")
