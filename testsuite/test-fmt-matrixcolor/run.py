# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command = "echo incorrect color space and matrix transform names:>> out.txt 2>&1 ;\n"
command += testshade("-t 1 -g 2 2 -od uint8 -o Cout_color matrixcolor1.tif -o Cout_color1 matrixcolor2.tif -o Cout_matrix matrixcolor3.tif -o Cout_matrix1 matrixcolor4.tif test_matrixcolor")
command += "echo const-foldable version of incorrect color space and matrix transform names:>> out.txt 2>&1 ;\n"
command += testshade("-t 1 -g 2 2 -od uint8 -o Cout_color const_matrixcolor1.tif -o Cout_color1 const_matrixcolor2.tif -o Cout_matrix const_matrixcolor3.tif -o Cout_matrix1 matrixcolor4.tif test_const_matrixcolor")
command += "echo test_2wrongspaces.osl:>> out.txt 2>&1 ;\n"
command += testshade("-t 1 -g 2 2 -od uint8 -o Cout_color matrixcolor1wrong.tif -o Cout_matrix matrixcolor2wrong.tif test_2wrongspaces")

