<!--
  Copyright Contributors to the Open Shading Language project.
  SPDX-License-Identifier: CC-BY-4.0
-->

# Gross syntax, shader types, parameters

The overall structure of a shader is as follows:

> *optional-function-or-struct-declarations* \
  \
  shader-type shader-name `(` optional-parameters `)` \
  `{` \
     *statements* \
  `}`

Note that *statements* may include function or structure definitions, local
variable declarations, or public methods, as well as ordinary execution
instructions (such as assignments, etc.).

## Shader types

Shader types include the following: `surface`, `displacement`, `light`,
`volume`, and generic `shader`.  Some operations may only be performed from
within certain types of shaders (e.g., one may only call `displace()` or alter
`P` in a displacement shader), and some global variables may only be accessed
from within certain types of shaders (e.g., `dPdu` is not defined inside a
volume shader).

Following are brief descriptions of the basic types of shaders:


`surface` shaders

: Surface shaders determine the basic material properties of a surface and
  how it reacts to light.  They are responsible for computing a
  `color closure` that describes the material, and optionally setting
  other user-defined output variables.  They may not alter
  the position of the surface.

  Surface shaders are written as if they describe the behavior of a single
  point on the primitive, and the renderer will choose the positions
  surface at which the shader must be evaluated.

  Surface shaders also are used to describe emissive objects, i.e., light
  sources.  OSL does not need a separate shader type to describe lights.

`displacement` shaders

: Displacement shaders alter the position and shading normal (or,
  optionally, just the shading normal) to make a piece of geometry appear
  deformed, wrinkled, or bumpy.  They are the only kind of shader that
  is allowed to alter a primitive's position.

`volume` shaders

: Volume shaders describe how a participating medium (air, smoke, glass,
  etc.) reacts to light and affects the appearance of objects on the other
  side of the medium.  They are similar to `surface` shaders, except
  that they may be called from positions that do not lie upon (and are not
  necessarily associated with) any particular primitive.

`shader` generic shaders

: Generic shaders are used for utility code, generic routines that may be
  called as individual layers in a shader group.  Generic shaders need not
  specify a shader type, and therefore may be reused from inside surface,
  displacement, or volume shader groups.  But as a result, they may
  not contain any functionality that cannot be performed from inside all
  shader types (for example, they may not alter `P`, which can only be done
  from within a displacement shader).


## Shader parameters

An individual shader has (optionally) many *parameters* whose
values may be set in a number of ways so that a single shader may have
different behaviors or appearances when used on different objects.

### Shader parameter syntax

Shader parameters are specified in the shader declaration, in
parentheses after the shader's name.  This is much like the parameters
to an OSL function (or a function in C or similar languages),
except that shader parameters must have an *initializer*, giving a
default value for the parameter.  Shader parameter default initializers
may be expressions (i.e., may be computed rather than restricted to
numeric constants), and are evaluated in the order that the parameters
are declared, and may include references to previously-declared
parameters.  Formally, the grammar for a simple parameter
declaration looks like this:

> *type parametername* `=` *default-expression*

where *type* is one of the data types described in Chapter [](#chap-types),
*parametername* is the name of the parameter, and *default-expression* is a
valid expression (see Section [](#sec-expressions)).  Multiple parameters are
simply separated by commas:

> *type1 parameter1* `=` *expr1* `,` *type2 parameter2* `=` *expr2* `,` ...

Fixed-length, one-dimensional array parameters are declared as follows:

> *type parametername* `[` *array-length* `] = {` *expr0* `,` *expr1* ... `}`

where *array-length* is a positive integer constant giving the length of the
array, and the initializer is a series of initializing expressions listed
between curly braces.  The first initializing expression provides the
initializer for the first element of the array, the second expression provides
the initializer for the second element of the array, and so on.  If the number
of initializing expressions is less than the length of the array, any
additional array elements will have undefined values.

Arrays may also be declared without a set length:

> *type parametername* `[ ] = {` *expr0* `,` *expr1* ... `}`

where no array length is found between the square brackets.
This indicates that the array's length will be determined based on
whatever is passed in --- a connection from the output of another shader
in the group (take on the length of that output), an instance value
(take on the length specified by the declaration of the instance value),
or a primitive variable (length determined by its declaration on the
primitive).  If no instance value, primitive value, or connection is
supplied, then the number of initializing expressions will determine the
length, as well as the default values, of the array.

Structure parameters are also straightforward to declare:

> *structure-type parametername* `= {` *expr0* `,` *expr1* ... `}`

where *structure-type* is the name of a previously-declared `struct` type, and
the *expr* initializers correspond to each respective field within the
structure.  An initializer of appropriate type is required for every field of
the structure.

### Shader output parameters

Shader parameters are, by default, read-only in the body of the
shader.  However, special \emph{output parameters} may be altered
by execution of the shader.  Parameters may be designated outputs
by use of the {\cf output} keyword immediately prior to the
type declaration of the parameter:

> `output` *type parametername* `=` *expr*

(Output parameters may be arrays and structures, but we will omit spelling out
the obvious syntax here.)

Output parameters may be connected to inputs of later-run shader layers in the
shader group, may be queried by later-run shaders in the group via message
passing (i.e., `getmessage()` calls), or used by the renderer as an output
image channel (in a manner described through the renderer's API).

### Shader parameter example

Here is an example of a shader declaration, with several parameters:

```
surface wood ( 
           /* Simple params with constant initializers */
               float Kd = 0.5,
               color woodcolor = color (.7, .5, .3),
               string texturename = "wood.tx",
           /* Computed from an earlier parameter */
               color ringcolor = 0.25 * woodcolor,
           /* Fixed-length array */
               color paintcolors[3] = { color(0,.25,0.7), color(1,1,1),
                                        color(0.75,0.5,0.2) },
           /* variable-length array */
               int pattern[] = { 2, 4, 2, 1 },
           /* output parameter */
               output color Cunlit = 0
             )
{
   ...
}
```

### How shader parameters get their values

Shader parameters get their values in the following manner,
in order of decreasing priority:

1. If the parameter has been designated by the renderer to be connected to an
   output parameter of a previously-executed shader layer within the shader
   group, that is the value it will get.
2. If the parameter matches the name and type of a per-primitive, per-face, or
   per-vertex *primitive variable* on the particular piece of geometry
   being shaded, the parameter's value will be computed by interpolating the
   primitive variable for each position that must be shaded.
3. If there is no connection or primitive variable, the parameter may will
   take on an *instance value*, if that parameter was given an explicit
   per-instance value at the time that the renderer referenced the shader
   (associating it with an object or set of objects).
4. If none of these overrides is present, the parameter's value will be
   determined by executing the parameter initialization code in the shader.

This triage is performed per parameter, in order of declaration.  So, for
example, in the code sample above where the default value for `ringcolor1 is a
scaled version of `woodcolor`, this relationship would hold whether
`woodcolor` was the default, an instance value, an interpolated primitive
value, or was connected to another layer's output.  Unless `ringcolor` itself
was given an instance, primitive, or connection value, in which case that's
what would be used.



## Shader metadata

A shader may optionally include *metadata* (data *about* the
shader, as opposed to data *used by* the shader).  Metadata may be
used to annotate the shader or any of its individual parameters with
additional hints or information that will be compiled into the shader
and may be queried by applications.  A common use of metadata is to
specify user interface hints about shader parameters --- for example,
that a particular parameter should only take on integer values, should
have an on/off checkbox, is intended to be a filename, etc.

Metadata is specified inside double brackets `[[` and `]]` enclosing a
comma-separated list of metadata items.  Each metadatum looks like a parameter
declaration --- having a data type, name, and initializer.  However, metadata
may only be simple types or arrays of simple types (not structs or closures)
and their value initializers must be numeric or string constants (not computed
expressions).

Metadata about the shader as a whole is placed between the shader name
and the parameter list.  Metadata about shader parameters are placed
immediately after the parameter's initializing expression, but before
the comma or closing parentheses that terminates the parameter
description.

Below is an example shader declaration showing the use of shader and
parameter metadata:

```
surface wood 
            [[ string help = "Realistic wood shader" ]]
    ( 
       float Kd = 0.5
           [[ string help = "Diffuse reflectivity",
              float min = 0, float max = 1 ]] ,
       color woodcolor = color (.7, .5, .3)
           [[ string help = "Base color of the wood" ]],
       color ringcolor = 0.25 * woodcolor
           [[ string help = "Color of the dark rings" ]],
       string texturename = "wood.tx"
           [[ string help = "Texture map for the grain",
              string widget = "filename" ]],
       int pattern = 0
           [[ string widget = "mapper",
              string options = "oak:0|elm:1|walnut:2" ]]
    )
{
   ...
}
```

The metadata are not semantically meaningful; that is, the metadata does
not affect the actual execution of the shader.  Most metadata exist only
to be embedded in the compiled shader and able to be queried by other
applications, such as to construct user interfaces for shader assignment
that allow usage tips, appropriate kinds of widgets for setting each
parameter, etc.  

The choice of metadata and their meaning is completely up to the shader
writer and/or modeling system.  However, we propose some conventions
below.  These conventions are not intended to be comprehensive, nor to
meet all your needs --- merely to establish a common nomenclature for
the most common metadata uses.

The use of metadata is entirely optional on the part of the shader
writer, and any application that queries shader metadata is free to
honor or ignore any metadata it finds.

string label
: A short label to be displayed in the UI for this parameter.  If not
  present, the parameter name itself should be used as the widget label.

string help
: Help text that describes the purpose and use of the shader or parameter.

string page
: Helps to group related widgets by ``page.''

string widget
: The type of widget that should be used to adjust this parameter.
  Suggested widget types:

  "number"
  : Provide a slider and/or numeric input. This is the default widget type for
    `float` or `int` parameters.  Numeric inputs also may be influenced by the
    following metadata: `min`, `max`, `sensitivity`, `digits`, `slider`,
    `slidermin`, `slidermax`, `slidercenter`, `sliderexponent`.
  
  "string"
  : Provide a text entry widget. This is the default widget type for
    {\cf string} parameters.
  
  "boolean"
  : Provide a pop-up menu with "Yes" and "No" options. Works on strings
    or numbers.  With strings, "Yes" and "No" values are used, with
    numbers, 0 and 1 are used.
  
  "checkBox"
  : A boolean widget displayed as a checkbox. Works on strings or
    numbers. With strings, "Yes" and "No" values are used, with numbers,
    0 and 1 are used.
  
  "popup"
  : A pop-up menu of literal choices. This widget further requires
    parameter metadata `options` (a string listing the supported
    menu items, delimited by the `|` character), and optionally `editable`
    (an integer, which if nonzero means the widget should allow the
    text field should be directly editable).  For example:
    ```
      string wrap = "default"
          [[ string widget = "popup",
             string options = "default|black|clamp|periodic|mirror" ]]
    ```
  
  "mapper"
  : A pop-up with associative choices (an enumerated type, if the values are
    integers).  This widget further requires parameter metadata `options`, a
    `|`-delimited string with "key:value" pairs.  For example:
    ```
      int pattern = 0
          [[ string widget = "mapper",
             string options = "oak:0|elm:1|walnut:2" ]]
    ```
  
  "filename"
  : A file selection dialog.
  
  "null"
  : A hidden widget.


float min, float max, int min, int max
: The minimum and/or maximum value that the parameter may take on.

float sensitivity, int sensitivity
: The precision or step size for incrementing or decrementing the value
  (within the appropriate min/max range).

int digits
: The number of digits to show (-1 for full precision).

int slider
: If nonzero, enables display of a slider sub-widget.  This also respects the
  following additional metadata that control the slider specifically:
  `slidermin` (minimum value for the slider, `slidermax` (maximum value for
  the slider), `slidercenter` (origin value for the slider), `sliderexponent`
  (nonlinear slider options).

string URL
: Provides a URL for full documentation of the shader or parameter.

string units
: Gives the assumed units, if any, for the parameter (e.g., `cm`, `sec`,
  `degrees`). The compiler or renderer may issue a warning if it detects that
  this assumption is being violated (for example, the compiler can warn if a
  `degrees` variable is passed as the argument to `cos`).


% We never implemented this:
% ## Public methods
% 
% Ordinary (non-public) functions inside a shader may be called only from
% within the shader; they do not generate entry points that the renderer
% is aware of.
% 
% A *public method* is a function that may be directly called by the
% renderer.  Only top-level local functions of a shader --- that is,
% declared within the braces that define the local scope of the shader,
% but not within any other such function --- may be public methods.  A
% function may be designated a public method by using the `public`
% keyword immediately before the function declaration:
% 
% > *shader-type shader-name* `(` *params* `)` \
% > `{` \
% > 
% > `public` *return-type* *function-name* `(` *optional-parameters* `)` \
% >  `{` \
% > *statements* \
% >  `}` \
% > 
% > ...\\
% > `}`


% A given renderer will publish a list of public methods (names, arguments
% expected, and return value) that has particular meaning for that
% renderer.  For example, a renderer may honor a public method
% ```
%     public float maxdisplacement ()
% ```
% that computes and returns the maximum distance
% that a displacement shader will move any surface points.

% At some later point, this spec will recommend several ``standard''
% public methods that should be honored by most renderers.

