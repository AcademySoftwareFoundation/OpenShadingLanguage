#!/usr/bin/env python
import os
loc = os.environ["OSLHOME"] + os.environ["MATERIALX_OSOS"]+ "/"
command = testshade("--print -param in.r -0.1 -param in.a -0.2   %smx_absval_color2  -o out mx.exr") %loc
command = None