<!-- Copyright Contributors to the Open Shading Language project. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

Standard Shader Ball
====================

`shaderball.obj` and `shaderball.mtl` are derived from the **StandardShaderBall**
asset published by the USD Working Group, which is licensed CC-BY-4.0:

  https://github.com/usd-wg/assets/tree/main/full_assets/StandardShaderBall

Credit: Chris Rydalch (geometry and textures), André Mazzone (specification and
validation), Thomas Anagnostou (original scene and inspiration).

Modifications
-------------

The original USD asset was imported into Blender 3.4.1 and re-exported as
Wavefront OBJ + MTL for `testrender`, which triangulates the mesh and drops the
shading networks. The exported `.mtl` retains the original material names, which
`testrender` uses to assign OSL shader groups.

Additional geometry in the form of emission planes were added in place of USD area
lights, which cannot be represented in OBJ format and which `testrender` can not use.
These are the `emitterTop0` through `emitterTop3` and `emitterLeft0` groups. The
modeled geometry is otherwise unchanged.
