# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command = "echo incorrect noisetype:>> out.txt 2>&1 ;\n"
command += testshade("-t 1 -g 2 2 -od uint8 -o Cout test_noise.tif test_noise")

