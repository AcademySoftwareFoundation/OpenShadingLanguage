#!/usr/bin/env python
import os
loc = os.environ["OSLHOME"] + os.environ["MATERIALX_OSOS"]+ "/"
command = testshade("--print -param in.rgb -0.1,-0.2,0.4 -param in.a -0.5  %smx_absval_color4 -o out mx.exr") %loc
command = None