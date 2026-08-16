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

# A single testshade run only ever builds one group, so use testrender for a
# scene with three groups of differing complexity. This exercises the parts
# of the ranking that one group cannot: ordering, the group-name-ascending
# tie-break, and exclusion of zero-valued groups from the ranked list while
# they still count toward min/max/median.
#
#   heavy: layer_a -> layer_b -> mtl   3 layers, depth 3, 1 texture, 2 noise
#   mid:              layer_b -> mtl   2 layers, depth 2, 0 texture, 2 noise
#   light:            simple  -> mtl   2 layers, depth 2, 0 texture, 0 noise
# --runstats prints getstats() while the groups are still alive (the
# statistics:level option instead reports at shading system teardown, by
# which time the renderer has released most of its groups).
command += testrender(
    "-r 32 32 -aa 1 --runstats --print-group-stats scene.xml out.exr"
)

# Filter to only the per-group ranked stats lines and getattribute stat key
# output; everything else is machine- or build-specific. The
# '\d+ (layers|depth|ops)  "' alternative keeps the individual ranked entries
# under "Top shader groups:", so the printed ranking is really compared.
# Note: runtest uses re.match() (anchored at line start), so prefix with .*
filter_re = r".*(Shader compilation stats|Active layers|Network depth|Texture ops|Noise ops|Top shader groups|\d+ (layers|depth|ops)  \"|stat:)"
