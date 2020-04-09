# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage
#!/usr/bin/env python 

command += testshade("-g 256 256 -od uint8 -o Cout out.tif test")
outputs = [ "out.tif" ]
