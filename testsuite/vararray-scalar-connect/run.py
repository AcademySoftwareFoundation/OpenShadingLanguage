# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage
#!/usr/bin/env python 


command += testshade("-param:type=float[4] fin 21,22,23,24 -layer u1 upstream \
   -layer u2 upstream \
   -param:type=float[6] a 41,42,43,44,45,46 \
   -layer t test \
   -connect u1 fout4 t a[0] \
   -connect u2 fout3 t a[2] \
   -connect u1 fout[1] t a[3] \
   -connect u2 fout[0] t a[4]")
