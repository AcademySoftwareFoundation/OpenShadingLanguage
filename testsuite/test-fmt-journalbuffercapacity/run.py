# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command = "echo test warning when journal buffer capacity is inadequate>> out.txt 2>&1 ;\n"
command += testshade("--jbufferMB 3 -t 1 -g 64 64 -od uint8 -o Cout test_jbc.tif test_jbc")

