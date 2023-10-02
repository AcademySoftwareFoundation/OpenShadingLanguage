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


(chap-types)=
# Data types

OSL provides several built-in simple data types for performing
computations inside your shader:

Type        | Explanation
:---------- | :-----------
`int`	    | Integer data
`float`	    | Scalar floating-point data (numbers)
`point`, `vector`, `normal` | Three-dimensional positions, directions, and face orientations
`color`	    | Spectral reflectivities and light energy values
`matrix`    | $4 \times 4$ transformation matrices
`string`	| Character strings (such as filenames)
`void`      | Indicates functions that do not return a value

In addition, you may create arrays and structures (much like C), and
OSL has a new type of data structure called a *closure*.

The remainder of this chapter will describe the simple and aggregate
data types available in OSL.

(sec-datatypes-int)=
## `int`

The basic type for discrete numeric values is `int`.  The size of
the `int` type is renderer-dependent, but is guaranteed to be at
least 32 bits.

Integer constants are constructed the same way as in C.  The following
are examples of `int` constants: `1`, `-2`, etc. Integer
constants may be specified as hexadecimal, for example: `0x01cf`.

Unlike C, no unsigned, bool, char, short, or long types are supplied.
This is to simplify the process of writing shaders (as well as
implementing shading systems).

The following operators may be used with `int` values (in order of
decreasing precedence, with each box holding operators of the same
precedence):


Operation | Result | Explanation
:--------- | :------ | :---
*int* `++` | `int` | post-increment by 1
*int* `--` | `int` | post-decrement by 1
  |  | 
  |  | 
`++` *int* | `int` | pre-increment by 1
`--` *int* | `int` | pre-decrement by 1
`-` *int* | `int` | unary negation
`~` *int* | `int` | bitwise complement (1 and 0 bits flipped)
`!` *int* | `int` | boolean `not' (1 if operand is zero, otherwise 0)
  |  | 
  |  | 
*int* `*` *int* | `int` | multiplication
*int* `/` *int* | `int` | division
*int* `%` *int* | `int` | modulus
  |  | 
  |  | 
*int* `+` *int* | `int` | addition
*int* `-` *int* | `int` | subtraction
  |  | 
  |  | 
*int* `<<` *int* | `int` | shift left
*int* `>>` *int* | `int` | shift right
  |  | 
  |  | 
*int* `<` *int* | `int` | 1 if the first value is less than the second, else 0
*int* `<=` *int* | `int` | 1 if the first value is less or equal to the second, else 0
*int* `>` *int* | `int` |  1 if the first value is greater than the second, else 0
*int* `>=` *int* | `int` | 1 if the first value is greater than or equal to the second, else 0
  |  | 
  |  | 
*int* `==` *int* | `int` | 1 if the two values are equal, else 0
*int* `!=` *int* | `int` | 1 if the two values are different, else 0
  |  | 
  |  | 
*int* `&` *int* | `int` | bitwise and
  |  | 
  |  | 
*int* `^` *int* | `int` | bitwise exclusive or
  |  | 
  |  | 
*int* `\|` *int* | `int` | bitwise or
  |  | 
  |  | 
*int* `&&` *int* | `int` | boolean and (1 if both operands are nonzero, otherwise 0)
  |  | 
  |  | 
*int* `\|\|` *int* | `int` | boolean or (1 if either operand is nonzero, otherwise 0)


Note that the `not`, `and`, and `or` keywords are synonyms for `!`, `&&`, and
`||`, respectively.



## `float`

The basic type for scalar floating-point numeric values is `float`.  The
size of the `float` type is renderer-dependent, but is guaranteed to
be at least IEEE 32-bit float (the standard C `float` data type).
Individual renderer implementations may choose to implement `float` with
even more precision (such as using the C `double` as the underlying
representation).

Floating-point constants are constructed the same way as in C. The following
are examples of `float` constants:  `1.0`, `2.48`, `-4.3e2`.

An `int` may be used in place of a `float` when used with any valid
`float` operator.  In such cases, the `int` will be promoted to a
`float` and the resulting expression will be `float`.  An `int` may
also be passed to a function that expects a `float` parameters, with the
`int` automatically promoted to `float`.

The following operators may be used with `float` values (in order of
decreasing precedence, with each box holding operators of the same
precedence):


Operation | Result | Explanation
:--------- | :------ | :---
*float* `++` | `float` | post-increment by 1
*float* `--` | `float` | post-decrement by 1
  |  | 
  |  | 
`++` *float*  | `float` | pre-increment by 1
`--` *float*  | `float` | pre-decrement by 1
`-` *float* | `float` | unary negation
  |  | 
  |  | 
*float* `*` *float* | `float` | multiplication
*float* `/` *float* | `float` | division
  |  | 
  |  | 
*float* `+` *float* | `float` | addition
*float* `-` *float* | `float` | subtraction
  |  | 
  |  | 
*float* `<` *float* | `int` | 1 if the first value is less than the second, else 0
*float* `<=` *float* | `int` | 1 if the first value is less or equal to the second, else 0
*float* `>` *float* | `int` |  1 if the first value is greater than the second, else 0
*float* `>=` *float* | `int` | 1 if the first value is greater than or equal to the second, else 0
  |  | 
  |  | 
*float* `==` *float* | `int` | 1 if the two values are equal, else 0
*float* `!=` *float* | `int` | 1 if the two values are different, else 0



## `color`

The `color` type is used to represent 3-component (RGB) spectral
reflectivities and light energies.  You can assemble a
color out of three floats, either representing an RGB triple or some
other color space known to the renderer, as well as from a single
float (replicated for all three channels).  Following are some examples:

```
    color (0, 0, 0)              // black
    color ("rgb", .75, .5, .5)   // pinkish
    color ("hsv", .2, .5, .63)   // specify in "hsv" space
    color (0.5)                  // same as color (0.5, 0.5, 0.5)
```

All these expressions above return colors in "rgb" space.  Even the third
example returns a color in "rgb" space --- specifically, the RGB value of the
color that is equivalent to hue 0.2, saturation 0.5, and value 0.63.  In other
words, when assembling a color from components given relative to a specific
color space in this manner, there is an implied transformation to "rgb" space.
The following table lists the built-in color spaces.

% \caption{Names of color spaces.}\label{tab:colorspacenames}

| | |
:--------- | :------
`"rgb"` | The coordinate system that all colors start out in, and in which the renderer expects to find colors that are set by your shader.   
`"hsv"` | hue, saturation, and value. 
`"hsl"` | hue, saturation, and lightness. 
`"YIQ"` | the color space used for the NTSC television standard. 
`"XYZ"` | CIE *XYZ* coordinates. 
`"xyY"` | CIE *xyY* coordinates. 

Colors may be assigned another color or a `float` value (which sets
all three components to the value).  For example:

```
    color C;
    C = color (0, 0.3, 0.3);
    C = 0.5;                    // same as C = color (0.5, 0.5, 0.5)
```

Colors can have their individual components examined and set using the
`[]` array access notation.  For example:

```
    color C;
    float g = C[1];   // get the green component
    C[0] = 0.5;       // set the red component
```

Components 0, 1, and 2 are red, green, and blue, respectively.
It is an error to access a color component with an index outside the
$[0...2]$ range.

Color variables may also have their components referenced using "named
components" that look like accessing structure fields named `r`, `g`, and `b`,
as synonyms for `[0]`, `[1]`, and `[2]`, respectively:

```
    float green = C.g;   // get the green component
    C.r = 0.5;           // set the red component
```

The following operators may be used with `color` values (in order of
decreasing precedence, with each box holding operators of the same
precedence):


Operation | Result | Explanation
:--------- | :------ | :---
*color* `[` *int* `]` | `float` | component access
  |  | 
  |  | 
`-` *color* | `color` | unary negation
  |  | 
  |  | 
*color* `*` *color* | `color` | component-wise multiplication
*color* `*` *float* | `color` | scaling
*float* `*` *color* | `color` | scaling
*color* `/` *color* | `color` | component-wise division
*color* `/` *float* | `color` | scaling
*float* `/` *color* | `color` | scaling
  |  | 
  |  | 
*color* `+` *color* | `color` | component-wise addition
*color* `-` *color* | `color` | component-wise subtraction
  |  | 
  |  | 
*color* `==` *color* | `int` | 1 if the two values are equal, else 0
*color* `!=` *color* | `int` | 1 if the two values are different, else 0

All of the binary operators may combine a scalar value (`float` or
`int`) with a `color`, treating the scalar if it were a `color` with
three identical components.


## Point-like types: `point`, `vector`, `normal`

Points, vectors, and normals are similar data types with identical
structures but subtly different semantics.  We will frequently refer to
them collectively as the "point-like" data types when making
statements that apply to all three types.

A `point` is a position in 3D space.  A `vector` has a length and
direction, but does not exist in a particular location.  A `normal` is a
special type of vector that is *perpendicular* to a surface, and
thus describes the surface's orientation.  Such a perpendicular vector
uses different transformation rules than ordinary vectors, as we will
describe below.

All of these point-like types are internally represented by three
floating-point numbers that uniquely describe a position or
direction relative to the three axes of some coordinate system.  

All points, vectors, and normals are described relative to some
coordinate system.  All data provided to a shader (surface information,
graphics state, parameters, and vertex data) are relative to one
particular coordinate system that we call the `"common"` coordinate
system.  The `"common"` coordinate system is one that is convenient
for the renderer's shading calculations.

You can "assemble" a point-like type out of three floats using a
constructor:

```
        point (0, 2.3, 1)
        vector (a, b, c)
        normal (0, 0, 1)
```

These expressions are interpreted as a point, vector, and normal
whose three components are the floats given, relative to "common" space .

As with colors, you may also specify the coordinates relative to some other
coordinate system:

```
    Q = point ("object", 0, 0, 0);
```

This example assigns to `Q` the point at the origin of "object" space.
However, this statement does *not* set the components of `Q` to (0,0,0)!
Rather, `Q` will contain the "common" space coordinates of the point that is
at the same location as the origin of "object" space.  In other words, the
point constructor that specifies a space name implicitly specifies a
transformation to "common" space.  This type of constructor also can be used
for vectors and normals.

The choice of "common" space is renderer-dependent, though will usually
be equivalent to either "camera" space or "world" space.

Some computations may be easier in a coordinate system other than
"common" space.  For example, it is much more convenient to apply a
"solid texture" to a moving object in its "object" space than in
"common" space.  For these reasons, OSL provides a built-in
`transform()` function that
allows you to transform points, vectors, and normals 
among different coordinate systems (see Section [](#sec-stdlib-geom)).  Note,
however, that OSL does not keep track of which point variables are
in which coordinate systems.  It is the responsibility of the shader
programmer to keep track of this and ensure that, for example, lighting
computations are performed using quantities in "common" space.

Several coordinate systems are predefined by name, listed in the following
table.  Additionally, a renderer will probably allow for additional coordinate
systems to be named in the scene description, and these names may also be
referenced inside your shader to designate transformations.

% \caption{Names of predeclared geometric spaces.\label{tab:spacenames}}

`"common"`
: The coordinate system that all spatial values start out in and the one in
  which all lighting calculations are carried out.  Note that the choice of
  `"common"` space may be different on each renderer. 

`"object"`
: The local coordinate system of the graphics primitive (sphere, patch, etc.)
  that we are shading. 

`"shader"`
: The local coordinate system active at the time that the shader was
  instanced. 

`"world"`
: The world coordinate system designated in the scene. 
  
`"camera"`
: The coordinate system with its origin at the center of the camera lens,
  $x$-axis pointing right, $y$-axis pointing up, and $z$-axis pointing into
  the screen. 

`"screen"`
: The coordinate system of the camera's image plane (after perspective
  transformation, if any).  Coordinate (0,0) of `"screen"` space is looking
  along the $z$-axis of "camera" space. 

`"raster"`
: 2D pixel coordinates, with (0,0) as the upper-left
  corner of the image and (xres, yres) as the lower-right corner. 

`"NDC"`
: 2D Normalized Device Coordinates --- like raster space, but normalized so
  that $x$ and $y$ both run from 0 to 1 across the whole image, with (0,0)
  being at the upper left of the image, and (1,1) being at the lower right. 


Point types can have their individual components examined and set using
the `[]` array access notation.  For example:

```
    point P;
    float y = P[1];   // get the y component
    P[0] = 0.5;       // set the x component
```

Components 0, 1, and 2 are $x$, $y$, and $z$, respectively.
It is an error to access a point component with an index outside the
$[0...2]$ range.

Point-like variables may also have their components referenced using "named
components" that look like accessing structure fields named `x`, `y`, and `z`,
as synonyms for `[0]`, `[1]`, and `[2]`, respectively:

```
    float yval = P.y;    // get the [1] or y component
    P.x = 0.5;           // set the [0] or x component
```

The following operators may be used with point-like values (in order of
decreasing precedence, with each box holding operators of the same
precedence):


Operation | Result | Explanation
:--------- | :------ | :---
*ptype* `[` `int` `]` | `float` | component access
  |  | 
  |  | 
`-` *ptype* | `vector` | component-wise unary negation
  |  | 
  |  | 
*ptype* `*` *ptype* | *ptype* | component-wise multiplication
`float` `*` *ptype* | *ptype* | scaling of all components
*ptype* `*` `float` | *ptype* | scaling of all components
*ptype* `/` *ptype* | *ptype* | component-wise division
*ptype* `/` `float` | *ptype* | division of all components
`float` `/` *ptype* | *ptype* | division by all components
  |  | 
  |  | 
*ptype* `+` *ptype* | *ptype* | component-wise addition
*ptype* `-` *ptype* | `vector` | component-wise subtraction
  |  | 
  |  | 
*ptype* `==` *ptype* | `int` | 1 if the two values are equal, else 0
*ptype* `!=` *ptype* | `int` | 1 if the two values are different, else 0


The generic *ptype* is listed in places where any of `point`, `vector`, or
`normal` may be used.

All of the binary operators may combine a scalar value (`float` or `int`) with
a point-like type, treating the scalar if it were point-like with three
identical components.


## `matrix`

OSL has a `matrix` type that represents the transformation matrix required to
transform points and vectors between one coordinate system and another.
Matrices are represented internally by 16 floats (a $4 \times 4$ homogeneous
transformation matrix).

A `matrix` can be constructed from a single float or 16 floats.  For example:

```
    matrix zero = 0;   // makes a matrix with all 0 components
    matrix ident = 1;  // makes the identity matrix

    // Construct a matrix from 16 floats
    matrix m = matrix (m00, m01, m02, m03, m10, m11, m12, m13, 
                       m20, m21, m22, m23, m30, m31, m32, m33);
```

Assigning a single floating-point number $x$ to a matrix will result
in a matrix with diagonal components all being $x$ and other
components being zero (i.e., $x$ times the identity matrix).
Constructing a matrix with 16 floats will create the matrix whose
components are those floats, in row-major order.  

Similar to point-like types, a `matrix` may be constructed in
reference to a named space:

```
    // Construct matrices relative to something other than "common"
    matrix q = matrix ("shader", 1);
    matrix m = matrix ("world", m00, m01, m02, m03, m10, m11, m12, m13, 
                               m20, m21, m22, m23, m30, m31, m32, m33);
```

The first form creates the matrix that transforms points from "shader" space
to "common" space.  Transforming points by this matrix is identical to calling
`transform("shader", "common", ...)`. The second form prepends the
current-to-world transformation matrix onto the $4 \times 4$ matrix with
components $m_{0,0} ... m_{3,3}$. Note that although we have used `"shader"`
and `"world"` space in our examples, any named space is acceptable.

A matrix may also be constructed from the names of two coordinate
systems, yielding the matrix that transforms coordinates from the
first named space to the second named space:

```
    matrix m = matrix ("object", "world");
```

The example returns the *object-to-world* transformation matrix.

Matrix variables can be tested for equality and inequality with the `==` and
`!=` boolean operators.  Also, the `*` operator between matrices denotes
matrix multiplication, while `m1 / m2` denotes multiplying `m1` by the inverse
of matrix `m2`.  Thus, a matrix can be inverted by writing `1/m`.  In
addition, some functions will accept matrix variables as arguments, as
described in Section [](#chap-stdlibrary).

Individual components of a matrix variable may be set or accessed using array
notation, for example,

```
    matrix M;
    float x = M[row][col];
    M[row][col] = 1;
```

Valid component indices are integers on $[0...3]$.  It is an error to access a
matrix component with either a row or column outside this range.

The following operators may be used with matrices (in order of decreasing
precedence, with each box holding operators of the same precedence):

Operation | Result | Explanation
:--------- | :------ | :---
*matrix* `[` *int* `][` *int* `]`  | `float`  | component access (row, column)
  |  | 
  |  | 
`-` *matrix* | `matrix` | unary negation
  |  | 
  |  | 
*matrix* `*` *matrix* | `matrix` | matrix multiplication
*matrix* `*` *float*  | `matrix` | component-wise scaling
*float* `*` *matrix*  | `matrix` | component-wise scaling
*matrix* `/` *matrix* | `matrix` | multiply the first matrix by the *inverse of the second
*matrix* `/` *float* | `matrix` | component-wise division
*float* `/` *matrix* | `matrix` | multiply the *float* by the *inverse of the matrix
  |  | 
  |  | 
*matrix* `==` *matrix* | *int* | 1 if the two values are equal, else 0
*matrix* `!=` *matrix* | *int* | 1 if the two values are different, else 0


## `string`

The `string` type may hold character strings.  The main application
of strings is to provide the names of files where textures may be
found.  Strings can be compared using `==` and `!=`.

String constants are denoted by surrounding the characters with double
quotes, as in `"I am a string literal"`.  As in C programs, string
literals may contain escape sequences such as `\n` (newline),
`\r` (carriage return), `\t` (tab), `\"` (double quote),
`\\` (backslash).

Two quote-quoted string literals that are separated only by whitespace
(spaces, tabs, or newlines) will be automatically concatenated into a
single string literal.  In other words,

```text
    "foo"  "bar"
```

is exactly equivalent to `"foobar"`.

## `void`

The `void` type is used to designate a function that does not return a value.
No variable may have type `void`.

## Arrays

Arrays of any of the basic types are supported, provided that they
are 1D and statically sized, using the usual syntax for C-like languages:

```
    float d[10];                       // Declare an uninitialized array
    float c[3] = { 0.1, 0.2, 3.14 };   // Initialize the array

    float f = c[1];                    // Access one element
```

The built-in function `arraylength()` returns the number of elements in an
array.  For example:

```
    float c[3];
    int clen = arraylength(c);        // should return 3
```

There are two circumstances when arrays do not need to have a declared
length --- an array parameter to a function, and a shader parameter that
is an array.  This is indicated by empty array brackets, as shown in the
following example:

```
    float sum (float x[])
    {
        float s = 0;
        for (int i = 0;  i < arraylength(x);  ++i)
            s += x[i];
        return s;
    }
```

It is allowed in OSL to copy an entire array at once using the `=` operator,
provided that the arrays contain elements of the same type and that the
destination array is at least as long as the source array.  For example:

```
    float array[4], anotherarray[4];
    ...
    anotherarray = array;
```


## Structures

Structures are used to group several fields of potentially different types
into a single object that can be referred to by name.  You may then use the
structure type name to declare structure variables as you would for any of the
built-in types.  Structure elements are accessed using the `.` ("dot")
operator.  The syntax for declaring and using structures is similar to C or
C++:

```
    struct RGBA {                    // Define a structure type
        color rgb;
        float alpha;
    };

    RGBA col;                        // Declare a structure
    r.rgb = color (1, 0, 0);         // Assign to one field
    color c = r.rgb;                 // Read from a structure field

    RGBA b = { color(.1,.2,.3), 1 }; // Member-by-member initialization
```

You can use "constructor expressions" for a your struct types much like
you can construct built-in types like `color` or `point`:

> *struct_name* `(` *first_member_value*, ... `}`

For example,

```
    RGBA c = RGBA(col,alpha);        // Constructor syntax

    RGBA add (RGBA a, RGBA b)
    {
        return RGBA (a.rgb+b.rgb, a.a+b.a);   // return expression
    }

    // pass constructor expression as a parameter:
    RGBA d = add (c, RGBA(color(.3,.4,.5), 0));
```

You may also use the *compound initializer list* syntax to construct a type
when it can be deduced from context which compound type is required. For
example, this is equivalent to the preceding example:

```
    RGBA c = {col,alpha};    // deduce by what is being assigned to

    RGBA add (RGBA a, RGBA b)
    {
        return { a.rgb+b.rgb, a.a+b.a }; // deduce by func return type
    }

    RGBA d = add (c, {{.3,.4,.5}, 0}); // deduce by expected arg type
```

It is permitted to have a structure field that is an array, as well as to have
an array of structures.  But it is not currently permitted to "nest" arrays
(that is, to have an array of structs which contain members that are arrays).

```
    struct A {
        color a;
        float b[4];       // Ok: struct may contain an array
    };

    RGBA vals[4];         // Ok: Array of structures
    vals[0].a = 0;

    A d[5];               // NO: Array of structures that contain arrays
```



## Closures

A `closure`is an expression or function call that will be stored, along
with necessary contextual information, to be evaluated at a later time.

In general, the type "`closure` *gentype*" behaves exactly like a
*gentype*, except that its numeric values may not be examined or
used for the duration of the shader's execution.  For example, a
`color closure` behaves mostly like a color --- you can multiply it by a
scalar, assign it to a `color closure` variable, etc. --- but you may not
assign it to an ordinary `color` or examine its individual component's
numeric values.

It is legal to assign `0` to a closure, which is understood to mean setting it
to a *null closure* (even though in all other circumstances, assigning a
`float` to a `closure`would not be allowed).

At present, the only type of `closure`supported by OSL is the `color closure`,
and the only allowed operations are those that let you form a linear
combination of `color closure`'s.  Additional closure types and operations are
reserved for future use.

Allowable operations on `color closure` include:

Operation | Result | Explanation
:--------- | :------ | :---
`-` *colorclosure* | `color closure` | unary negation
  |  | 
  |  | 
*color* `*` *colorclosure* | `color closure` | component-wise scaling
*colorclosure* `*` *color* | `color closure` | component-wise scaling
*float* `*` *colorclosure* | `color closure` | scaling
*colorclosure* `*` *float* | `color closure` | scaling
  |  | 
  |  | 
*colorclosure* `+` *colorclosure* | `color closure` | component-wise addition
% *colorclosure* `-` *colorclosure* | `color closure` | component-wise subtraction


