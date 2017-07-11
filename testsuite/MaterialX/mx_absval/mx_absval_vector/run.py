#!/usr/bin/env python
import os
loc = os.environ["OSLHOME"] + os.environ["MATERIALX_OSOS"]+ "/"
command = testshade("--print -param in -0.1,-0.2,0.3 %smx_absval_vector  -o out mx.exr") %loc
