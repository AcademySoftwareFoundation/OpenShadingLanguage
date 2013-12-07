#!/usr/bin/python 

# Simple test on a grid texture
command = testshade("-g 1 1 test")

# Construct a test specifically for odd data and pixel windows
command += oiiotool("--pattern checker 100x50+10+20 3 --fullsize 300x200+0+0 -o win.exr")
command += testshade("-g 1 1 --param filename win.exr --param date 0 test")
