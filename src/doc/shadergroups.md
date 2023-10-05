---
numbering:
  heading_1: true
  heading_2: true
  heading_3: true
---

<!--
  Copyright Contributors to the Open Shading Language project.
  SPDX-License-Identifier: CC-BY-4.0
-->


(chap-shader-groups)=
# Describing shader groups

Below, we propose a simple grammar giving a standard way to serialize (i.e.,
express as text) a full shader group, including instance values and
connectivity of the shader layers. There are only three statements/operations:
set an instance value, make a shader instance, and connect two instances.

**`param`** *`type paramname value`*... **`;`** <br> **`param`** *`type paramname value`*... **`[[`** *`metadata`*... **`]] ;`**

  : Declare an instance value of a shader parameter to be applied to the next
    `shader` statement. We refer to the parameter values set, which have not
    yet had their shader declared, as *pending parameters*.

    The *`paramname`* is the name of the parameter whose instance values are
    being specified.

    The *`type`* is one of the basic numeric or string data types described in
    Chapter [](#chap-types) (`int`, `float`,`color`, `point`, `vector`, `normal`,
    `matrix`, or `string`), or an array thereof (indicated by the usual notation
    of `[size]`). The *`type`* must match the declared type of the parameter in
    the shader.

    The actual values are listed individually, with multiple values (in the
    case of an array or an aggregate such as a `color`) simply separated by
    whitespace. If fewer values are supplied than the total number of array
    elements or aggregate components, the remainder will be understood to be
    filled with 0 values. String values must be enclosed in double quotes
    (`"like this"`).

    The `param` statement is terminated by a semicolon (`;`).

    Optionally, metadata hints may be supplied enclosed by double brackets,
    immediately before the semicolon.


**`shader`** *`shadername layername`* **`;`**

  : Declares a shader instance, which will receive any pending parameters that
    were declared since the previous `shader` statement (and in the process,
    clear the list of pending parameters).

    The `shadername` is an identifier that specifies the name of the shader
    to use as the master for this instance. The `layername` is an identifier
    that names the layer (e.g., to subsequently specify it as a source or
    destination for connections).

    The `shader` statement is terminated by a semicolon (`;`).


**`connect`** *`source_layername`* **`.`** *`paramname`* $~~$ *` destination_layername`* **`.`** *`paramname`* **`;`**

  : Establish a connection between an output parameter of a source layer and an
    input parameter of a destination layer, both of which have been previously
    declared within this group. The source layer must have preceded the
    destination layer when they were declared, and the parameters must exist and
    be of a compatible type so that it is meaningful to establish a connection
    (for example, you may connect a `color` to a `color`, but you may not connect
    a `color` to a `matrix`).

    If the named parameters are structures, the two structures must have
    identical data layout, and establishing the connection will connect each
    corresponding data member.  It is also possible to make a connection of just
    a single member of a structure by using the usual ``dot'' syntax, for
    example, for layers `A` and `B`, `connect A.c.x B.y` might
    connect A's parameter `c`, member `x`, to B's parameter `y` (the
    types of `c.x` in A and `y` in B must match).

    The `connect` statement is terminated by a semicolon (`;`).

**Example**

    param string name "rings.tx" ;      # set pending `name'
    param float scale 3.5 ;             # set pending `scale'
    shader "texturemap" "tex1" ;        # tex1 layer, picks up `name', `scale'
    param string name "grain.tx" ;
    shader "texturemap" "tex2" ;
    param float gam 2.2 ;
    shader "gamma" "gam1" ;
    param float gam 1.0 ;
    shader "gamma" "gam2" ;
    param color woodcolor 0.42 0.38 0.22 ;       # example of a color param
    shader "wood" "wood1" ;
    connect tex1.Cout gam1.Cin ;        # connect tex1's Cout to gam1's Cin
    connect tex2.Cout gam2.Cin ;
    connect gam1.Cout wood1.rings ;
    connect gam2.Cout wood1.grain ;
