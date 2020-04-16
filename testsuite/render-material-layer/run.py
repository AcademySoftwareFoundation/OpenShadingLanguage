# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage
#!/usr/bin/python 

outputs = [ "out.exr" ]
command = testrender("-r 320 240 -aa 4 material-layer.xml out.exr")
