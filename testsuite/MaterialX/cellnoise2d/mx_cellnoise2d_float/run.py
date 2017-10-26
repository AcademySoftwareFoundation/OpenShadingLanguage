#!/usr/bin/env python
import os
loc = os.environ["OSLHOME"] + os.environ["MATERIALX_OSOS"]+ "/"

shader = loc + os.path.basename(os.getcwd())

command = testshade("-g 512 512 -od uint8 -o out out.tif " + shader)
outputs = [ "out.txt", "out.tif" ]