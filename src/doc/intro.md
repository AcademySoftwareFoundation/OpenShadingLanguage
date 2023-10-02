<!--
  Copyright Contributors to the Open Shading Language project.
  SPDX-License-Identifier: CC-BY-4.0
  SPDX-License-Identifier: BSD-3-Clause
-->


(chap-introduction=)
# Introduction

Welcome to Open Shading Language!


Open Shading Language (OSL) is a small but rich language for
programmable shading in advanced renderers and other applications, ideal
for describing materials, lights, displacement, and pattern generation.

OSL was developed by Sony Pictures Imageworks for use in its in-house
renderer used for feature film animation and visual effects. The
language specification was developed with input by other visual effects
and animation studios who also wish to use it.

OSL is distributed under the "New BSD" license.  In short, you are free
to use it in your own applications, whether they are free or commercial,
open or proprietary, as well as to modify the OSL code as you desire,
provided that you retain the original copyright notices as described in
the license.


#### How OSL is different from other shading languages

OSL has syntax similar to C, as well as other shading languages.
However, it is specifically designed for advanced rendering algorithms
and has features such as radiance closures, BSDFs, and deferred ray
tracing as first-class concepts.

OSL has several unique characteristics not found in other shading
languages (certainly not all together).  Here are some things you will
find are different in OSL compared to other languages:

* **Surface and volume shaders compute radiance closures, not final colors.**

  OSL's surface and volume shaders compute an explicit symbolic
  description, called a "closure," of the way a surface or volume
  scatters light, in units of radiance.  These radiance closures may be
  evaluated in particular directions, sampled to find important
  directions, or saved for later evaluation and re-evaluation.
  This new approach is ideal for a physically-based renderer that
  supports ray tracing and global illumination.

  In contrast, other shading languages usually compute just a surface
  color as visible from a particular direction.  These old shaders are
  "black boxes" that a renderer can do little with but execute to for
  this once piece of information (for example, there is no effective way
  to discover from them which directions are important to sample).
  Furthermore, the physical units of lights and surfaces are often
  underspecified, making it very difficult to ensure that shaders are
  behaving in a physically correct manner.

* **Surface and volume shaders do not loop over lights or shoot rays.**

  There are no "light loops" or explicitly traced rays in OSL surface
  shaders.  Instead, surface shaders compute a radiance closure
  describing how the surface scatters light, and a part of the renderer
  called an "integrator" evaluates the closures for a particular set of
  light sources and determines in which directions rays should be
  traced.  Effects that would ordinarily require explicit ray tracing,
  such as reflection and refraction, are simply part of the radiance
  closure and look like any other BSDF.

  Advantages of this approach include that integration and sampling may
  be batched or re-ordered to increase ray coherence; a "ray budget" can
  be allocated to optimally sample the BSDF; the closures may be used by
  for bidirectional ray tracing or Metropolis light transport; and the
  closures may be rapidly re-evaluated with new lighting without having
  to re-run the shaders.

* **Surface and light shaders are the same thing.**

  OSL does not have a separate kind of shader for light sources.  Lights
  are simply surfaces that are emissive, and all lights are area lights.

* **Transparency is just another kind of illumination.**

  You don't need to explicitly set transparency/opacity variables in the
  shader.  Transparency is just another way for light to interact with a
  surface, and is included in the main radiance closure computed by a
  surface shader.

* **Renderer outputs (AOV's) are specified using "light path expressions."**

  Sometimes it is desirable to output images containing individual
  lighting components such as specular, diffuse, reflection, individual
  lights, etc.  In other languages, this is usually accomplished by
  adding a plethora of "output variables" to the shaders that collect
  these individual quantities.

  OSL shaders need not be cluttered with any code or output variables to
  accomplish this.  Instead, there is a regular-expression-based
  notation for describing which light paths should contribute to which
  outputs.  This is all done on the renderer side (though supported by
  the OSL implementation).  If you desire a new output, there is no need
  to modify the shaders at all; you only need to tell the renderer the
  new light path expression.

* **Shaders are organized into networks.**

  OSL shaders are not monolithic, but rather can be organized into
  networks of shaders (sometimes called a shader group, graph, or DAG),
  with named outputs of some nodes being connected to named inputs of
  other nodes within the network.  These connections may be done
  dynamically at render time, and do not affect compilation of
  individual shader nodes.  Furthermore, the individual nodes are
  evaluated lazily, only their outputs are "pulled" from the later nodes
  that depend on them (shader writers may remain blissfully unaware of
  these details, and write shaders as if everything is evaluated
  normally).

* **No "uniform" and "varying" keywords in the language.**

  OSL shaders are evaluated in SIMD fashion, executing shaders on many
  points at once, but there is no need to burden shader writers with
  declaring which variables need to be uniform or varying.  

  In the open source OSL implementation, this is done both automatically
  and dynamically, meaning that a variable can switch back and forth
  between uniform and varying, on an instruction-by-instruction basis,
  depending on what is assigned to it.

* **Arbitrary derivatives without grids or extra shading points.**

  In OSL, you can take derivatives of any computed quantity in a shader,
  and use arbitrary quantities as texture coordinates and expect correct
  filtering.  This does not require that shaded points be arranged in a
  rectangular grid, or have any particular connectivity, or that any
  "extra points" be shaded.  

  In the open source OSL implementation, this is possible because
  derivatives are not computed by finite differences with neighboring
  points, but rather by "automatic differentiation," computing partial
  differentials for the variables that lead to derivatives, without any
  intervention required by the shader writer.



## Acknowledgments

The original designer and project leader of OSL is Larry Gritz. Other early
developers of OSL are (in order of joining the project): Cliff Stein, Chris
Kulla, Alejandro Conty, Jay Reynolds, Solomon Boulos, Adam Martinez, Brecht
Van Lommel.

Additionally, many others have contributed features, bug fixes, and other
changes: Steve Agland, Shane Ambler, Martijn Berger, Farchad Bidgolirad,
Nicholas Bishop, Stefan Büttner, Matthaus G. Chajdas, Thomas Dinges, Henri
Fousse, Syoyo Fujita, Derek Haase, Sven-Hendrik Haase, John Haddon, Daniel
Heckenberg, Ronan Keryell, Elvic Liang, Max Liani, Bastien Montagne, Erich
Ocean, Mikko Ohtamaa, Alex Schworer, Sergey Sharybin, Stephan Steinbach,
Esteban Tovagliari, Alexander von Knorring, Roman Zulak. (Listed
alphabetically; if we've left anybody out, please let us know.)

We cannot possibly express sufficient gratitude to the managers at Sony
Pictures Imageworks who allowed this project to proceed, supported it
wholeheartedly, and permitted us to release the source, especially Rob
Bredow, Brian Keeney, Barbara Ford, Rene Limberger, and Erik Strauss.

Huge thanks also go to the crack shading team at SPI, and the brave
lookdev TDs and CG supes willing to use OSL on their shows.  They served
as our guinea pigs, inspiration, testers, and a fantastic source of
feedback.  Thank you, and we hope we've been responsive to your needs.

OSL was not developed in isolation.  We owe a debt to the individuals
and studios who patiently read early drafts of the language
specification and gave us very helpful feedback and additional ideas,
and especially to those at other companies who have taken the risk of
incorporating OSL into their products and pipelines.

The open source OSL implementation incorporates or depends upon several
other open source packages:

- **OpenImageIO** Copyright Contributors to OpenImageIO.
  http://openimageio.org
- **Imath** Copyright Contributors to Imath.
  http://www.openexr.com
- **Boost** Copyright various authors.  http://www.boost.org
- **LLVM** Copyright 2003-2010 University of Illinois at
  Urbana-Champaign. http://llvm.org

These other packages are all distributed under licenses that allow them
to be used by and distributed with OSL.



