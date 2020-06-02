#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

# Test the 'lazylayer' option.
# We have two layers, but the first one is never connected to the second.
# Normal lazy evaluation means that layer should not run at all.
#
# But the 'a' layer contains two error() calls: one that should be optimized
# away, and one that does not.
#
# When lazyerror=1 (default), the first layer should be skipped, including
# encountering the error call, since its outputs are not used. But when
# lazyerror=0, the error statement that isn't optimized away should still be
# called and a warning should be issued about errors that were not optimized
# away.

command += "; echo 'lazyerror=1:' >> out.txt ; "
command += testshade("-res 2 1 -options lazyerror=1,opt_warnings=1 -O2 -layer alayer a -layer blayer b")
command += "; echo 'lazyerror=0:' >> out.txt ; "
command += testshade("-res 2 1 -options lazyerror=0,opt_warnings=1 -O2 -layer alayer a -layer blayer b")

