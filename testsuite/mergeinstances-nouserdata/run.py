#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

# This test is specifically to test the opt_merge_instances_with_userdata
# option. If working correctly when set to zero, the a1 and a2 layers should
# NOT merge, and thus you will see two printf output line from "a".
# On the other hand, a single "a" printf means that the a1 and a2 layers
# merged, which is the correct default but should not happen when you
# set opt_merge_instances_with_userdata=0.

command = testshade("-options opt_merge_instances_with_userdata=0 " +
                    "-layer a1 a -layer a2 a -layer b b -connect a1 Cout b C1 " +
                    " -connect a2 Cout b C2 -o b.Cout out.exr")
