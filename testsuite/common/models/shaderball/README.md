<!-- Copyright Contributors to the Open Shading Language project. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

Standard Shader Ball
====================

The geometry in this directory is derived from the **StandardShaderBall**
asset published by the USD Working Group:

  https://github.com/usd-wg/assets/tree/main/full_assets/StandardShaderBall

The matching textures live in `testsuite/render-shaderball/maps` and come
from the same asset.


Credits
-------

* Chris Rydalch — geometry and textures
* André Mazzone — specification and validation
* Thomas Anagnostou — original scene, inspiration, and consultation

Per the upstream README, the asset descends from the "Simball" / "Material
Preview" scene created by Thomas Anagnostou, which was originally released
under a Creative Commons Attribution-ShareAlike license. The USD Working
Group asset is released under Creative Commons Attribution 4.0.


License
-------

This material is licensed under the Creative Commons Attribution 4.0
International License (`SPDX-License-Identifier: CC-BY-4.0`).

To view a copy of this license, visit
https://creativecommons.org/licenses/by/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

The upstream license file is at:
https://github.com/usd-wg/assets/blob/main/full_assets/StandardShaderBall/LICENCE

Note that this is a different license from the BSD-3-Clause license that
covers OSL's own source code. See the repository's `THIRD-PARTY.md`.


Modifications
-------------

`shaderball.obj` and `shaderball.mtl` are **modified** from the upstream
asset. The upstream asset is distributed as USD with MaterialX shading
networks; OSL's `testrender` reads Wavefront OBJ, so the scene was imported
into Blender 3.4.1 and re-exported as OBJ + MTL. The changes are:

* **Format conversion**, USD to Wavefront OBJ. The USD stage is flattened
  into a single file, with each mesh written as an OBJ group.

* **Triangulation.** All faces in the exported mesh are triangles.
  UVs and normals are preserved.

* **Shading networks removed.** The upstream MaterialX networks are not
  carried over. The exported `.mtl` retains the upstream *material names*,
  which `testrender` uses to bind OSL shader groups.

* **Emissive geometry added.** The upstream asset lights its scene with
  USD area lights, which are analytic and are not representable
  in Wavefront OBJ. `testrender` has no area lights either, instead it
  illuminates a scene with emissive geometry. Light-emitting planes
  were added in their place, positioned to match the upstream light
  transforms. These are the `emitterTop0`--`emitterTop3` and
  `emitterLeft0` groups.

Apart from the added emissive planes, no changes were made to the shape of
the modeled geometry beyond the triangulation performed by the exporter.
