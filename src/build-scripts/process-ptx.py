#!/usr/bin/env python
# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Open a PTX file and:
#  * add the `.visible` directive to all functions
#  * add the `.visible` directive to global/constant variables

from __future__ import print_function, absolute_import

import re
import sys

in_name = sys.argv[1]
out_name = sys.argv[2]
ptx = ""
with open(in_name, 'r') as ptx_in:
    ptx = ptx_in.read()
ptx = re.sub(r'(?m)^(\.const .align)', ".visible .const .align", ptx)
ptx = re.sub(r'(?m)^(\.global .align)', ".visible .global .align", ptx)
ptx = re.sub(r'(?m)^(\.func)', ".visible .func", ptx)
with open(out_name, 'w') as ptx_out:
    ptx_out.write(ptx)
