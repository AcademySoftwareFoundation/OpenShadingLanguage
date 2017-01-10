#!/usr/bin/env python 

# Don't have it compile the .osl to .oso -- the whole point is to verify
# that we're doing it without file I/O
compile_osl_files = False

command = testshade("--inbuffer test")
