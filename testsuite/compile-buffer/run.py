# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage
#!/usr/bin/env python 

# Don't have it compile the .osl to .oso -- the whole point is to verify
# that we're doing it without file I/O
compile_osl_files = False

command = testshade("--inbuffer test")
