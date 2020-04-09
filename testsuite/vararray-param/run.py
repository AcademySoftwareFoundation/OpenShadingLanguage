# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage
#!/usr/bin/env python 

#command += testshade("-layer alayer a --layer blayer b --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in")
command += testshade("-param:type=float[5] a 1.1,1.2,1.3,1.4,1.5 test")
