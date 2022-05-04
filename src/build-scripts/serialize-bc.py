# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Turn an LLVM-compiled bitfile into a C++ source file where the compiled
# bitcode is in a huge array.

from __future__ import print_function, absolute_import

import sys

in_name = sys.argv[1]
out_name = sys.argv[2]
prefix = sys.argv[3]
f_in = open(in_name, 'rb')
f_out = open(out_name, 'w')
f_out.write('#include <cstddef>\n')
f_out.write('unsigned char ' + prefix + '_block[] = {\n')
f_in.read
if (sys.version_info > (3, 0)):
    for c in f_in.read():
        f_out.write(hex(c) + ',\n')
else:
    for c in f_in.read():
        f_out.write('0x{},\n'.format(c.encode('hex')))
f_out.write('0x00 };\n')
f_out.write('int {}_size = sizeof({}_block)-1;\n'.format(prefix, prefix))
