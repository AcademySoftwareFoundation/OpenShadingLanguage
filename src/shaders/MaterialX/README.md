<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright Contributors to the Open Shading Language Project. -->


README for MaterialX Shader Nodes
=================================

### [MaterialX](http://www.materialx.org/)

This is a reference implementation of version 1.36 of the MaterialX
specification, drafted by Lucasfilm Ltd., in collaboration with Autodesk and
The Foundry.  This collection is intended to demonstrate the intended logic
of the MaterialX specification shading nodes as well as a practical
application of current features of the OSL library such as constructors and
operator overloads.

The MaterialX specification is authored and maintained by Doug Smythe and
Jonathan Stone at Lucasfilm Ltd. This OSL reference implementation was
primarily authored by Adam Martinez and Derek Haase.

This directory in the distribution contains `.osl` files that can be
compiled directly with `oslc`, as well as the source `.mx` template files,
which are used to generate the `.osl` (one for each of several specialized
types).

Each `.mx` file contains a type-independent template of a shader, using
special strings such as TYPE, TYPE_ZERO, TYPE_DEFAULT_IN, etc. You can
inspect the build_materialX_osl.py script to see how a template .mx files is
turned into .osl with the type macro substitutions. The logic for which type
specializations are applied to which node .mx templates is contained in the
CMakeLists.txt file found in OSL's src/shaders/MaterialX directory.
