#!/usr/bin/env python

failureok = True    # Expect an error
command = testshade("-g 2 2 --layer lay1 foo --layer lay2 bar --connect lay1 x lay2 y")
