#!/usr/bin/env python
import os
loc = os.environ["OSLHOME"] + os.environ["MATERIALX_OSOS"]+ "/"
command = testshade("--print -param in.x -0.1 -param in.y -0.2 -param in.z -0.3 -param in.w 0.5 %smx_absval_vector4  -o out mx.exr") %loc
command = None