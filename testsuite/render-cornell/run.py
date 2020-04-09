#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

failthresh = max (failthresh, 0.005)   # allow a little more LSB noise between platforms
outputs = [ "out.exr" ]
command = testrender("-r 256 256 -aa 4 cornell.xml out.exr")
