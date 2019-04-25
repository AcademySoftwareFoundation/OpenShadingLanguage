# Turn an LLVM-compiled bitfile into a C++ source file where the compiled
# bitcode is in a huge array.

import sys
in_name = sys.argv[1]
out_name = sys.argv[2]
prefix = sys.argv[3]
f_in = open(in_name, 'rb')
f_out = open(out_name, 'w')
f_out.write('#include <cstddef>\n')
f_out.write('unsigned char %s_block[] = {\n' % prefix)
f_in.read
for c in f_in.read():
    f_out.write('0x%s,\n' % c.encode('hex'))
f_out.write('0x00 };\n')
f_out.write('int %s_size = sizeof(%s_block)-1;\n' % (prefix, prefix))
