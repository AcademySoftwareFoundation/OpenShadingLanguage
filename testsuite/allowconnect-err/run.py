#!/usr/bin/env python

failureok = True    # Expect an error
command += testshade("-layer alayer a --layer blayer b --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in")
outputs = [ "out.txt" ]
