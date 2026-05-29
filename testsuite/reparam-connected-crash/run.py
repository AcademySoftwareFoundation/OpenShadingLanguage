#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Regression test: downstream input param is both connected (from upstream) and
# marked interactive with an instance value.
#
# Bug: OSL give precedence to the interactive trait instead of connected. This
# leads to an attempted write of the interactive buffer. However, the offset
# is calculated incorrectly and the write occurs at interactive_buffer[-1].
#
# Correct output: in = 0.5 (u at the default shade point, from the connection)
# Buggy output:   in = <garbage> (corrupted memory) or crash

command += testshade(
    "-layer upstream_layer upstream "
    "-layer downstream_layer -param:interactive=1 in 9.0 downstream "
    "-connect upstream_layer out downstream_layer in "
)

outputs = ["out.txt"]
