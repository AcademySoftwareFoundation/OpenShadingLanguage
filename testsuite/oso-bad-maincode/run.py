#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Test that loading an oso file whose main code section ends up empty is
# rejected with an error, rather than later using the start of main code
# (one past the last op) as an op index. Found by fuzzing. Two ways to get
# there: a file that ends right after the `code ___main___` marker, and a
# stray byte >= 0x80 in the code, which the lexer once turned into a
# negative token that the parser took as end of input. The "valid" shader
# must still load and run.

# Make the high-byte case from the valid shader, keeping non-ASCII bytes
# out of the repo.
with open("data/valid.oso", "rb") as f :
    oso = f.read()
oso = oso.replace(b"shader valid", b"shader high_byte")
oso = oso.replace(b"code ___main___\n", b"code ___main___\n\xe3\xdf}\n")
with open("high_byte.oso", "wb") as f :
    f.write(oso)

for s in [ "data/valid", "data/empty_main", "high_byte" ] :
    command += testshade("-g 1 1 " + s)
