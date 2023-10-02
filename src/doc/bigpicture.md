<!--
  Copyright Contributors to the Open Shading Language project.
  SPDX-License-Identifier: CC-BY-4.0
-->


(chap-bigpicture=)
# The Big Picture


This chapter attempts to lay out the major concepts of OSL,
define key nomenclature, and sketch out how individual shaders fit 
together in the context of a renderer as a whole.

<!--
  Other than the background material of this chapter, the rest of this
  specification deals strictly with the language itself.  In the future,
  there will be separate (shorter) documents explaining in detail the use
  of the language compiler, the renderer-side issues, the library and APIs
  for how a renderer actually causes shaders to be evaluated, and so on.
-->

#### A shader is code that performs a discrete task

A shader is a program, with inputs and outputs, that performs a specific
task when rendering a scene, such as determining the appearance behavior
of a material or light.  The program code is written in OSL, the
specification of which comprises this document.

For example, here is a simple `gamma` shader that performs
simple gamma correction on is `Cin` input, storing the result
in its output `Cout`:

```{image} Figures/shaderschematic.png
   :width: 6.0in
```

The shader's inputs and outputs are called *shader parameters*.  
Parameters have default values, specified in the shader code, but may
also be given new values by the renderer at runtime.  

#### Shader instances

A particular shader may be used many times in a scene, on different
objects or as different layers in a shader group.  Each separate use of
a shader is called a *shader instance*.  Although all instances of
a shader are comprised of the same program code, each instance may
override any or all of its default parameter values with its own set of
*instance values*.

Below is a schematic showing a `gamma` instance with the `gam` parameter
overridden with an instance-specific value of `2.2`.

```{image} Figures/instanceschematic.png
   :width: 200px
```



#### Shader groups, layers, and connections

A *shader group* is an ordered sequence of individual shaders called *layers*
that are executed in turn.  Output parameters of an earlier-executed layer may
be *connected* to an input parameter of a later-executed layer. This connected
network of layers is sometimes called a *shader network* or a *shader DAG*
(directed acyclic graph).  Of course, it is fine for the shader group to
consist of a single shader layer.

Below is a schematic showing how several shader instances may be
connected to form a shader group.



```{image} Figures/groupschematic.png
   :width: 6.0in
```



And here is sample pseudo-code shows how the above network may be assembled
using an API in the renderer [^rendererapi]:

```
    ShaderGroupBegin ()
    Shader ("texturemap",               /* shader name */
            "tex1",                     /* layer name */
            "string name", "rings.tx")  /* instance variable */
    Shader ("texturemap", "tex2", "string name", "grain.tx")
    Shader ("gamma", "gam1", "float gam", 2.2)
    Shader ("gamma", "gam2", "float gam", 1)
    Shader ("wood", "wood1")
    ConnectShaders ("tex1",     /* layer name A */
                    "Cout",     /* an output parameter of A */
                    "gam1",     /* layer name B */
                    "Cin")      /* Connect this layer of B to A's Cout */
    ConnectShaders ("tex2", "Cout", "gam2", "Cin")
    ConnectShaders ("gam1", "Cout", "wood1", "rings")
    ConnectShaders ("gam2", "Cout", "wood1", "grain")
    ShaderGroupEnd ()
```

Or, expressed as serialized text (as detailed in Chapter [](chap-shader-groups):

    param string name "rings.tx" ;
    shader "texturemap" "tex1" ;
    param string name "grain.tx" ;
    shader "texturemap" "tex2" ;
    param float gam 2.2 ;
    shader "gamma" "gam1" ;
    param float gam 1.0 ;
    shader "gamma" "gam2" ;
    shader "wood" "wood1" ;
    connect tex1.Cout gam1.Cin ;
    connect tex2.Cout gam2.Cin ;
    connect gam1.Cout wood1.rings ;
    connect gam2.Cout wood1.grain ;

The rules for which data types may be connected are generally the same as
the rules determining which variables may be assigned to each other in OSL
source code:

* `source` and `dest` are the same data type.
* `source` and `dest` are both *triples* (`color`, `point`,
  `vector`, or `normal`), even if they are not the same kind of triple.
* `source` is an `int` and `dest` is a `float`.
* `source` is a `float` or `int` and `dest` is a
  *triple* (the scalar value will be replicated for all three components
  of the triple).
* `source` is a single component of an aggregate type (e.g. one
  channel of a `color`) and `dest` is a `float` (or vice versa).

#### Geometric primitives

The *scene* consists of primarily of geometric primitives,
light sources, and cameras.

*Geometric primitives* are shapes such as NURBS, subdivision surfaces,
polygons, and curves.  The exact set of supported primitives may vary
from renderer to renderer.

Each geometric primitive carries around a set of named *primitive
variables* (also sometimes called *interpolated values* or
*user data*).  Nearly all shape types will have, among their primitive
variables, control point positions that, when interpolated, actually
designate the shape.  Some shapes will also allow the specification of
normals or other shape-specific data.  Arbitrary user data may also be
attached to a shape as primitive variables.  Primitive variables may be
interpolated in a variety of ways: one constant value per primitive, one
constant value per face, or per-vertex values that are interpolated
across faces in various ways.

If a shader input parameter's name and type match the name and type
of a primitive variable on the object (and that input parameters is
not already explicitly connected to another layer's output), the
interpolated primitive variable will override the instance value or
default.


#### Attribute state and shader assignments

Every geometric primitive has a collection of *attributes* (sometimes
called the *graphics state*) that includes its transformation
matrix, the list of which lights illuminate it, whether it is one-sided
or two-sided, shader assignments, etc.  There may also be a long list of
renderer-specific or user-designated attributes associated with each
object.  A particular attribute state may be shared among many geometric
primitives.

The attribute state also includes shader assignments --- the shaders or
shader groups for each of several *shader uses*, such as surface
shaders that designate how light reflects or emits from each point on a shape,
displacement shaders that can add fine detail to the shape on a
point-by-point basis, and volume shaders that describe how light is
scattered within a region of space.  A particular renderer may have
additional shader types that it supports.


#### Shader execution state: parameter binding and global variables

When the body of code of an individual shader is about to execute, all
its parameters are *bound* --- that is, take on specific values
(from connections from other layers, interpolated primitive variables,
instance values, or defaults, in that order).

Certain state about the position on the surface where the shading is
being run is stored in so-called *global variables*.  This includes
such useful data as the 3D coordinates of the point being shaded, the
surface normal and tangents at that point, etc.

Additionally, the shader may query other information about other
elements of the attribute state attached to the primitive, and
information about the renderer as a whole (rendering options, etc.).

#### Surface and volume shaders compute closures

Surface shaders (and volume shaders) do not by themselves compute the
final color of light emanating from the surface (or along a volume).
Rather, they compute a *closure*, which is a symbolic representation
describing the appearance of the surface, that may be more fully
evaluated later.  This is in effect a parameterized formula, in which
some inputs have definite numeric values, but others may depend on
quantities not yet known (such as the direction from which the surface
is being viewed, and the amount of light from each source that is
arriving at the surface).

For example, a surface shader may compute its result like this:

    color paint = texture ("file.tx", u, v);
    Ci = paint * diffuse (N);

In this example, the variable `paint` will take on a specific numeric
value (by looking up from a texture map).  But the `diffuse()` function
returns a `color closure`, not a definite numeric `color`.  The output
variable `Ci` that represents the appearance of the surface is also a `color
closure`, whose numeric value is not known yet, except that it will be the
product of `paint`` and a Lambertian reflectance.



```{image} Figures/shaderexecschematic.png
```

The closures output by surface and volume shaders can do a number of
interesting things that a mere number cannot:

* Evaluate: given input and output light directions, compute the
  proportion of light propagating from input to output.
* Sample: given just an input (or output) direction, choose a
  scattering direction with a probability distribution that is
  proportional to the amount of light that will end up going in various
  directions.
* Integrate: given all lights and a view direction, compute
  the total amount of light leaving the surface in the view direction.
* Recompute: given changes only to lights (or only to one light),
  recompute the integrated result without recomputing other lights or
  any of the calculations that went into assembling constants in the
  closure (such as texture lookups, noise functions, etc.).

<!--
  At present, we are assuming that the primitive closure functions (such
  as `diffuse`, `ward`, `cooktorrance`, etc.) are all built
  into the renderer, or implemented as renderer plugins.  At a later time,
  possibly in a later draft or maybe not until a truly later version of
  the spec, we will fully spec it out so that closure primitive functions
  may be written in OSL.  But I fear that if we do it too soon,
  we'll screw it up.  But, yes, the eventual goal is for you to be able to
  write these primitive functions in the language itself.
-->


#### Integrators

The renderer contains a number of *integrators* (selectable via the
renderer's API) which will combine the color closures computed by
surfaces and volumes with the light sources and view-dependent
information, to yield the amount of light visible to the camera.

```{image} Figures/integratorschematic.png
   :width: 6.0in
```

<!--
   At present, this document is written as if the integrators are built
   into the renderer itself (or implemented as renderer plug-ins).  At a
   later time, we intend to make it possible for integrators themselves
   to be written in OSL.
-->


#### Units

You can tell the renderer (through a global option) what units the scene
is using for distance and time.  Then the shader has a built-in function
called `transformu()` that works a lot like `transform()`, but
instead of converting between coordinate systems, it converts among
units.  For example,

```
    displacement bumpy (float bumpdist = 1,
                        string bumpunits = "cm")
    {
        // convert bumpdist to common units
        float spacing = transformu (bumpunits, "common", bumpdist);
        float n = noise (P / spacing);
        displace (n);
    }
```

So you can write a shader to achieve some effect in real world units,
and that shader is totally reusable on another show that used different
modeling units.

It knows all the standard names like `cm`, `in`, `km`,
etc., and can convert among any of those, as well as between named
coordinate systems.  For example,

```
    float x = transformu ("object", "mm", 1);
```

now `x` is the number of millimeters per unit of "object" space on
that primitive.

[^rendererapi]: This document does not dictate a specific renderer API for
   declaring shader instances, groups, and connections; the code above is just
   an example of how it might be done.

