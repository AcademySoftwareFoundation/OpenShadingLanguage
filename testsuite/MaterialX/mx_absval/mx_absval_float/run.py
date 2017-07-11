#!/usr/bin/env python
import os
loc = os.environ["OSLHOME"] + os.environ["MATERIALX_OSOS"]+ "/"
command = testshade("--print -param in -1.0 %smx_absval_float  -o out mx.exr") %loc
