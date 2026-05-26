#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Test that per-group compile stats are recorded and reported in getstats().
#
# "complex" group: layer_a (1 texture op) -> layer_b (2 noise ops)
#   Expected: active_layers=2, network_depth=2, texture_ops=1, noise_ops=2
#
# With statistics:level=1, getstats() should emit min/max/median and a
# ranked list for each metric.

command = testshade(
    "--options statistics:level=1"
    " --groupname complex"
    " --shader layer_a la"
    " --shader layer_b lb"
    " --connect la Cout lb Cin"
    " -o Cout null"
)

command += testshade(
    "--print-group-stats"
    " --groupname complex"
    " --shader layer_a la"
    " --shader layer_b lb"
    " --connect la Cout lb Cin"
    " -o Cout null"
)

# Filter to only the new per-group ranked stats lines and getattribute
# stat key output; everything else is machine- or build-specific.
# Note: runtest uses re.match() (anchored at line start), so prefix with .*
filter_re = r".*(Shader compilation stats|Active layers|Network depth|Texture ops|Noise ops|Top shader groups|stat:)"
