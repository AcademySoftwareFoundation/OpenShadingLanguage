<!--
  Copyright Contributors to the Open Shading Language project.
  SPDX-License-Identifier: CC-BY-4.0
-->


(chap-glossary)=
# Glossary

Attribute state

  : The set of variables that determines all the properties (other than shape)
    of a *geometric primitive* --- such as its local transformation matrix,
    the surface, displacement, and volume shaders to be used, which light
    sources illuminate objects that share the attribute state, whether
    surfaces are one-sided or two-sided, etc.  Basically all of the options
    that determine the behavior and appearance of the primitives, that are not
    determined by the shape itself or the code of the shaders.  A single
    attribute state may be shared among multiple geometric primitives.  Also
    sometimes called *graphics state*.

BSDF

  : Bidirectional scattering distribution function, a function that describes
    light scattering properties of a surface.

Built-in function

  : A function callable from within a shader, where the implementation of the
    function is provided by the renderer (as opposed to a function that the
    shader author writes in OSL itself).

Closure

  : A symbolic representation of a function to be called, and values for its
    parameters, that are packaged up to be evaluated at a later time to yield
    a final numeric value.

Connection

  : A routing of the value of an *output parameter* of one *shader layer* to
    an *input parameter* of another shader layer within the same *shader
    group*.

Default parameter value

  : The initial value of a *shader parameter*, if the renderer does not
    override it with an *instance value*, an interpolated *primitive
    variable*, or a *connection* to an output parameter of another *layer*
    within the *group*.  The default value of a shader parameter is explicitly
    given in the code for that shader, and may either be a constant or a
    computed expression.

EDF

  : Emission distribution function, a function that describes the distribution
    of light emitted by a light source.

Geometric primitive

  : A single shape, such as a NURBS patch, a polygon or subdivision mesh, a
    hair primitive, etc.

Global variables

  : The set of "built-in" variables describing the common renderer inputs to
    all shaders (as opposed to shader-specific parameters).  These include
    position (`P`), surface normal (`N`), surface tangents (`dPdu`, `dPdv`),
    as well as standard radiance output (`Ci`). Different *shader types*
    support different subsets of the global variables.

Graphics state

  : See *attribute state*.

Group

  : See *shader group*.

Input parameter

  : A read-only *shader parameter* that provides a value to control a
    shader's behavior.  Can also refer to a read-only parameter to a
    *shader function*.

Instance value

  : A constant value that overrides a default parameter value for a
    particular *shader instance*.  Each instance of a shader may have a
    completely different set of instance values for its parameters.

Layer

  : See *shader layer*.

Output parameter

  : A read/write *shader parameter* allows a shader to provide outputs beyond
    the *global variables* such as `Ci`.  Can also refer to a read/write
    parameter to a *shader function*, allowing a function to provide more
    outputs than a simple return value.

Primitive

  : Usually refers to a *geometric primitive*.

Primitive variable

  : A named variable, and values, attached to an individual geometric
    primitive.  Primitive variables may have one of several *interpolation
    methods* --- such as a single value for the whole primitive, a value for
    each piece or face of the primitive, or per-vertex values that are
    smoothly interpolated across the surface.

% Public method
% 
%   : A function within a shader that has an entry point that is visible and
%     directly callable by the renderer, as opposed to merely being called from
%     other code within the shader.  Public methods must be *top-level* (not
%     defined within other functions) and must be preceded by the `public`
%     keyword.

Shader

  : A small self-contained program written in OSL, used to extend the
    functionality of a renderer with custom behavior of materials and lights.
    A particular shader may have multiple *shader instances* within a scene,
    each of which has its unique *instance parameters*, transformation, etc.

Shader function

  : A function written in OSL that may be called from within a shader.

Shader group

  : An ordered collection of *shader instances* (individually called the
    *layers* of a group) that are executed to collectively determine material
    properties or displacement of a geometric primitive, emission of a light
    source, or scattering properties of a volume.  In addition to executing
    sequentially, layers within a group may optionally have any of their input
    parameters explicitly connected to output parameters of other layers
    within the group in an acyclic manner (thus, sometimes being referred to
    as a *shader network*).

Shader instance

  : A particular reference to a *shader*, with a unique set of *instance
    values*, transformation, and potentially other attributes.  Each shader
    instance is a separate entity, despite their sharing executable code.

Shader network

  : See *shader group*.

Shader layer

  : An individual *shader instance* within a *shader group*.

Shader parameter

  : A named input or output variable of a shader. *Input parameters* provide
    "knobs" that control the behavior of a shader; *output parameters*
    additional provide a way for shaders to produce additional output beyond
    the usual *global variables*.

Shading

  : The computations within a renderer that implement the behavior and visual
    appearance of materials and lights.

VDF

  : Volumetric distribution function, a function that describes the scattering
    properties of a volume.
