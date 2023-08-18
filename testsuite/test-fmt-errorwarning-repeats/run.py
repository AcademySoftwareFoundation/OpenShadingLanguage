# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command = "echo errors and warnings with repeats:>> out.txt 2>&1 ;\n"
command += testshade("--options error_repeats=1 -t 1 -g 2 2 -od uint8 -o Cout test_errorwarning.tif test_errorwarning")

