#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

command += testshade('-v --oslquery -group "' +
                         'shader a alayer, ' +
                         'shader b blayer, ' +
                         'connect alayer.f_out blayer.f_in, ' +
                         'connect alayer.c_out blayer.c_in"')
