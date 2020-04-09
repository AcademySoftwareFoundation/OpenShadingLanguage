# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage
#!/usr/bin/env python 

command += testshade("-g 64 64 -layer alayer a --layer blayer b "
                     "--connect alayer f_out blayer f_in "
                     " --connect alayer c_out blayer c_in "
                     "-od uint8 -o alayer.c_out out.tif")

outputs = [ "out.tif" ]
