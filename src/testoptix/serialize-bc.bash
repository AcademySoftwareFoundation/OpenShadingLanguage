#!/bin/bash

# Turn an LLVM-compiled bitfile into a C++ source file where the compiled
# bitcode is in a huge array.

in=$1
out=$2

echo "#include <cstddef>" > $out
echo "unsigned char rend_llvm_compiled_ops_block[] = {" >> $out
hexdump -v -e '"" /1 "0x%02x" ",\n"' $in >> $out
echo "0x00 };" >> $out
echo "size_t rend_llvm_compiled_ops_size = sizeof(rend_llvm_compiled_ops_block)-1;" >> $out

