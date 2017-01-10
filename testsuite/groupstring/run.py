#!/usr/bin/env python

command += testshade('-v --oslquery -group "' +
                         'shader a alayer, \n' +
                         'shader b blayer, \n' +
                         'connect alayer.f_out blayer.f_in, \n' +
                         'connect alayer.c_out blayer.c_in "')
