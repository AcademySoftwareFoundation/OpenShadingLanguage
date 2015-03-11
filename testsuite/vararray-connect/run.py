#!/usr/bin/env python 

#command += testshade("-layer alayer a --layer blayer b --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in")
command += testshade("-layer u upstream -layer t test -connect u fout t a")
