#!/usr/bin/env python
import os
loc = os.environ["OSLHOME"] + os.environ["MATERIALX_OSOS"]+ "/"
command = testshade("--print -param:type=float in.x -0.1 -param:type=float in.y -0.2 %smx_absval_vector2  -o out mx.exr") %loc
command = None