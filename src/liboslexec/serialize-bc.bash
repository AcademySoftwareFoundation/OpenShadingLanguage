#!/bin/bash

# Turn an LLVM-compiled bitfile into a C++ source file where the compiled
# bitcode is in a huge array.

in=$1
out=$2
prefix=$3

echo "#include <cstddef>" > $out
echo "unsigned char " ${prefix}"_block[] = {" >> $out
hexdump -v -e '"" /1 "0x%02x" ",\n"' $in >> $out
echo "0x00 };" >> $out
echo "size_t " ${prefix}"_size = sizeof("${prefix}"_block)-1;" >> $out

