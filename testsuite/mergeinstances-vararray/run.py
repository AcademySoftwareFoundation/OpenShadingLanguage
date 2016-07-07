#!/usr/bin/env python

# This is a regression test for a bug in which instance merging was
# done incorrectly for shaders that differed only in the values given
# to parameters which were arrays of unspecified length.

command = testshade("--group data/shadergroup")
