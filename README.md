README for Open Shading Language
================================

Build status:

[![Build Status](https://travis-ci.org/lgritz/OpenShadingLanguage.svg?branch=master)](https://travis-ci.org/lgritz/OpenShadingLanguage)

Table of contents
------------------

* Introduction
* How OSL is different
* What OSL consists of
* Building OSL
* Current state of the project and road map
* Contacts
* Credits


Introduction
------------

Welcome to Open Shading Language!

Open Shading Language (OSL) is a small but rich language for
programmable shading in advanced renderers and other applications, ideal
for describing materials, lights, displacement, and pattern generation.

OSL was developed by Sony Pictures Imageworks for use in its in-house
renderer used for feature film animation and visual effects. The
language specification was developed with input by other visual effects
and animation studios who also wish to use it.

OSL is robust and production-proven, and was the exclusive shading
system for work on big VFX films such as "Men in Black 3", "The Amazing
Spider-Man," "Oz the Great and Powerful," and "Edge of Tomorrow," as
well as animated features such as "Hotel Transylvania" and "Cloudy With
a Chance of Meatballs 2", and many other films completed or currently in
production.

The OSL code is distributed under the "New BSD" license (see the
"LICENSE" file that comes with the distribution), and the documentation
under the Creative Commons Attribution 3.0 Unported License
(http://creativecommons.org/licenses/by/3.0/).  In short, you are free
to use OSL in your own applications, whether they are free or
commercial, open or proprietary, as well as to modify the OSL code and
documentation as you desire, provided that you retain the original
copyright notices as described in the license.



How OSL is different
--------------------

OSL has syntax similar to C, as well as other shading languages.
However, it is specifically designed for advanced rendering algorithms
and has features such as radiance closures, BSDFs, and deferred ray
tracing as first-class concepts.

OSL has several unique characteristics not found in other shading
languages (certainly not all together).  Here are some things you will
find are different in OSL compared to other languages:

* Surface and volume shaders compute radiance closures, not final colors.

  OSL's surface and volume shaders compute an explicit symbolic
  description, called a "closure", of the way a surface or volume
  scatters light, in units of radiance.  These radiance closures may be
  evaluated in particular directions, sampled to find important
  directions, or saved for later evaluation and re-evaluation.
  This new approach is ideal for a physically-based renderer that
  supports ray tracing and global illumination.

  In contrast, other shading languages usually compute just a surface
  color as visible from a particular direction.  These old shaders are
  "black boxes" that a renderer can do little with but execute to find
  this one piece of information (for example, there is no effective way
  to discover from them which directions are important to sample).
  Furthermore, the physical units of lights and surfaces are often
  underspecified, making it very difficult to ensure that shaders are
  behaving in a physically correct manner.

* Surface and volume shaders do not loop over lights or shoot rays.

  There are no "light loops" or explicitly traced illumination rays in
  OSL surface shaders.  Instead, surface shaders compute a radiance
  closure describing how the surface scatters light, and a part of the
  renderer called an "integrator" evaluates the closures for a
  particular set of light sources and determines in which directions
  rays should be traced.  Effects that would ordinarily require explicit
  ray tracing, such as reflection and refraction, are simply part of the
  radiance closure and look like any other BSDF.

  Advantages of this approach include that integration and sampling may
  be batched or re-ordered to increase ray coherence; a "ray budget" can
  be allocated to optimally sample the BSDF; the closures may be used by
  for bidirectional ray tracing or Metropolis light transport; and the
  closures may be rapidly re-evaluated with new lighting without having
  to re-run the shaders.

* Surface and light shaders are the same thing.

  OSL does not have a separate kind of shader for light sources.  Lights
  are simply surfaces that are emissive, and all lights are area lights.

* Transparency is just another kind of illumination.

  You don't need to explicitly set transparency/opacity variables in the
  shader.  Transparency is just another way for light to interact with a
  surface, and is included in the main radiance closure computed by a
  surface shader.

* Renderer outputs (AOV's) may be specified using "light path expressions."

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

* Shaders are organized into networks.

  OSL shaders are not monolithic, but rather can be organized into
  networks of shaders (sometimes called a shader group, graph, or DAG),
  with named outputs of some nodes being connected to named inputs of
  other nodes within the network.  These connections may be done
  dynamically at render time, and do not affect compilation of
  individual shader nodes.  Furthermore, the individual nodes are
  evaluated lazily, only when their outputs are "pulled" from the later
  nodes that depend on them (shader writers may remain blissfully
  unaware of these details, and write shaders as if everything is
  evaluated normally).

* Arbitrary derivatives without grids or extra shading points.

  In OSL, you can take derivatives of any computed quantity in a shader,
  and use arbitrary quantities as texture coordinates and expect correct
  filtering.  This does not require that shaded points be arranged in a
  rectangular grid, or have any particular connectivity, or that any
  "extra points" be shaded.  This is because derivatives are not
  computed by finite differences with neighboring points, but rather by
  "automatic differentiation", computing partial differentials for the
  variables that lead to derivatives, without any intervention required
  by the shader writer.

* OSL optimizes aggressively at render time

  OSL uses the LLVM compiler framework to translate shader networks into
  machine code on the fly (just in time, or "JIT"), and in the process
  heavily optimizes shaders and networks with full knowledge of the
  shader parameters and other runtime values that could not have been
  known when the shaders were compiled from source code.  As a result,
  we are seeing our OSL shading networks execute 25% faster than the
  equivalent shaders hand-crafted in C!  (That's how our old shaders
  worked in our renderer.)



What OSL consists of
--------------------

The OSL open source distribution consists of the following components:

* oslc, a standalone compiler that translates OSL source code into
  an assembly-like intermediate code (in the form of .oso files).

* liboslc, a library that implements the OSLCompiler class, which
  contains the guts of the shader compiler, in case anybody needs to
  embed it into other applications and does not desire for the compiler
  to be a separate executable.

* liboslquery, a library that implements the OSLQuery class, which
  allows applications to query information about compiled shaders,
  including a full list of its parameters, their types, and any metadata
  associated with them.

* oslinfo, a command-line program that uses liboslquery to print to the
  console all the relevant information about a shader and its parameters.

* liboslexec, a library that implements the ShadingSystem class, which
  allows compiled shaders to be executed within an application.
  Currently, it uses LLVM to JIT compile the shader bytecode to x86
  instructions.

* testshade, a program that lets you execute a shader (or connected
  shader network) on a rectangular array of points, and save any of its
  outputs as images.  This allows for verification of shaders (and the
  shading system) without needing to be integrated into a fully
  functional renderer, and is the basis for most of our testsuite
  verification.  Along with testrender, testshade is a good example
  of how to call the OSL libraries.

* testrender, a tiny ray-tracing renderer that uses OSL for shading.
  Features are very minimal (only spheres are permitted at this time)
  and there has been no attention to performance, but it demonstrates how
  the OSL libraries may be integrated into a working renderer, what
  interfaces the renderer needs to supply, and how the BSDFs/radiance
  closures should be evaluated and integrated (including with multiple
  importance sampling).

* A few sample shaders.

* Documentation -- at this point consisting of the OSL language
  specification (useful for shader writers), but in the future will have
  detailed documentation about how to integrate the OSL libraries into
  renderers.



Building OSL
------------

Please see the "INSTALL" file in the OSL distribution for instructions
for building the OSL source code.



Current state of the project and road map
-----------------------------------------

At Sony Pictures Imageworks, we are exclusively using OSL in our
proprietary renderer, "Arnold."  Completed productions that used OSL for
shading have included:

    Men in Black 3
    The Amazing Spider-Man
    Hotel Transylvania
    Oz the Great and Powerful
    Smurfs 2
    Cloudy With a Chance of Meatballs 2
    Amazing Spider-Man 2
    Edge of Tomorrow
    Blended
    22 Jump Street
    Guardians of the Galaxy
    The Interview
    Fury
    American Sniper
    Pixels

And more are currently in production. Our shader-writing team works
entirely in OSL, all productions use OSL, and we've even removed all the
code from the renderer that allows people to write the old-style "C"
shaders.  At the time we removed the old shader facility, the OSL
shaders were consistently outperforming their equivalent old compiled C
shaders in the old system.

In the longer term, there are a number of projects we hope to get to
leading to a 2.x or 3.x cut of the language and library.  Among our
long-term goals:

* More documentation, in particular the "Integration Guide" that
  documents all the public APIs of the OSL libraries that you use when
  integrating into a renderer.  Currently, the source code to
  "testrender" is the best/only example of how to integrate OSL into a
  renderer.

* Our set of sample shaders is quite anemic.  We will eventually have a
  more extensive set of useful, production-quality shaders and utility
  functions you can call from your shaders.

* Currently "closure primitives" are implemented in C++ in the OSL
  library or in the renderer, but we would like a future spec of the
  language to allow new closure primitives to be implemented in OSL
  itself.

* Similarly, integrators are now implemented in the renderer, but we
  want a future OSL release to allow new integrators to be implemented
  in OSL itself.

* We would like to implement alternate "back ends" that would allow
  translation of OSL shaders (and shader networks) into code that can
  run on GPUs or other exotic hardware (at least for the biggest subset
  of OSL that can be expressed on such hardware).  This would, for
  example, allow you to view close approximations to your OSL shaders in
  realtime preview windows in a modeling system or lighting tool.

We (the renderer development team at Sony Pictures Imageworks) probably
can't do these all right away (in fact, probably can't do ALL of them in
any time range).  But we hope that as an open source project, other
users and developers will step up to help us explore more future
development avenues for OSL than we would be able to do alone.



Contacts
--------

[OSL GitHub page](https://github.com/imageworks/OpenShadingLanguage)

[Read or subscribe to the OSL development mail list](http://groups.google.com/group/osl-dev)

Email the lead architect:  lg AT imageworks DOT com

[Most recent PDF of the OSL language specification](https://github.com/imageworks/OpenShadingLanguage/blob/master/src/doc/osl-languagespec.pdf
)

[OSL home page at SPI](http://opensource.imageworks.com/?p=osl)

[Sony Pictures Imageworks main open source page](http://opensource.imageworks.com)

If you want to contribute code back to the project, you'll need to
sign [a Contributor License Agreement](http://opensource.imageworks.com/cla/).


Credits
-------

The original designer and open source administrator of OSL is Larry Gritz.

The main/early developers of OSL are (in order of joining the project):
Larry Gritz, Cliff Stein, Chris Kulla, Alejandro Conty, Jay Reynolds,
Solomon Boulos, Adam Martinez, Brecht Van Lommel.

Additionally, many others have contributed features, bug fixes, and other
small changes: Steve Agland, Shane Ambler, Martijn Berger, Nicholas Bishop,
Matthaus G. Chajdas, Thomas Dinges, Henri Fousse, Syoyo Fujita, Derek Haase,
Sven-Hendrik Haase, John Haddon, Daniel Heckenberg, Ronan Keryell, Max
Liani, Bastien Montagne, Erich Ocean, Mikko Ohtamaa, Alex Schworer, Sergey
Sharybin, Stephan Steinbach, Esteban Tovagliari, Alexander von Knorring.
(Listed alphabetically; if we've left anybody out, please let us know.)

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
specification and gave us very helpful feedback and additional ideas.
(I hope to mention them by name after we get permission of the people
and studios involved.)

The OSL implementation incorporates or depends upon several other open
source packages:

[OpenImageIO (c) Larry Gritz, et al](http://www.openimageio.org)

[Boost - various authors](http://www.boost.org)

[IlmBase (c) Industrial Light & Magic](http://www.openexr.com)

[LLVM Compiler Infrastructure](http://llvm.org)


