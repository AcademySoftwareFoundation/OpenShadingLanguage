#!/usr/bin/env python

command += testshade('-v --oslquery -group "' +
                         'shader a alayer, ' +
                         'shader b blayer, ' +
                         'connect alayer.f_out blayer.f_in, ' +
                         'connect alayer.c_out blayer.c_in"')
