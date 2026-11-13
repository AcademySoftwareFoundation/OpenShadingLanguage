#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Regression test: per-parameter hints (--param:lockgeom=0, etc.) must apply
# only to the parameters of the layer they were given for, no matter how many
# --param options earlier layers had.

lg  = "--param:lockgeom=0 s 0.0 --param:lockgeom=0 t 0.0"
opt = "-g 2 2 -t 1"

def run(label, args):
    global command
    command += "echo " + label + " >> out.txt ;\n"
    command += testshade(opt + " " + args)

# Upstream layer with unhinted params; target hints must stay on target
run("case2", "--layer up --param a 1.0 --param b 2.0 --param c 3.0 up --layer target " + lg + " tgt")
run("case4", "--layer up --param a 1.0 up --layer target " + lg + " tgt")
# Upstream hints must not leak onto the target's params
run("case6", "--layer up --param:lockgeom=0 a 0.0 --param:lockgeom=0 b 0.0 up --layer target --param s 0.5 --param t 0.5 tgt")
# Three layers, mixed hinted/unhinted
run("three", "--layer up --param:lockgeom=0 a 0.0 --param b 2.0 up --layer mid --param m 1.0 --param:lockgeom=0 o 0.0 mid --layer target --param s 0.5 --param:lockgeom=0 t 0.0 tgt")
# --reparam (separate value list) must not disturb the hint pairing
run("reparam", "--iters 2 --layer up --param a 1.0 up --layer target --param:type=float:interactive=1 s 0.5 --param:lockgeom=0 t 0.0 tgt --reparam:type=float:interactive=1 target s 0.25")

outputs = [ "out.txt" ]
