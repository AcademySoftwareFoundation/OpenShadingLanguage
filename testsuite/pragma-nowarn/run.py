#!/usr/bin/env python

# This shader would ordinarily issue a warning.
# With -Werror, it should be upgraded to an error.
oslcargs = "-Werror"

# BUT... the shader carefully uses #pragma nowarn to disable the warning.
# Which should cause the test to pass.

command = testshade("test")

