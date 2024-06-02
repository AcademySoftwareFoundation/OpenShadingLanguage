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


(chap-stdlibrary)=
# Standard Library Functions


% \def`float`colorpoint{The *`type`* may be any of `float`, `color`,
%   `point`, `vector`, or `normal`.  For `color` and `point`-like types, the
%   computations are performed component-by-component (separately for $x$,
%   $y$, and $z$).\xspace}

## Basic math functions

### Mathematical constants

OSL defines several mathematical constants:

| Constant   | Value |
| :--------- | :---- |
`M_PI`       | $\pi$
`M_PI_2`     | $\pi/2$
`M_PI_4`     | $\pi/4$
`M_2_PI`     | $2/\pi$
`M_2PI`      | $2\pi$
`M_4PI`      | $4\pi$
`M_2_SQRTPI` | $2/\sqrt{\pi}$
`M_E`        | $e$
`M_LN2`      | $\ln 2$
`M_LN10`     | $\ln 10$
`M_LOG2E`    | $\log_2 e$
`M_LOG10E`   | $\log_{10} e$
`M_SQRT2`    | $\sqrt{2}$
`M_SQRT1_2`  | $\sqrt{1/2}$

### Mathematical functions

Most of these functions operate on a generic *type* that my be any of `float`,
`color`, `point`, `vector`, or `normal`.  For `color` and `point`-like types,
the computations are performed component-by-component (separately for `x`,
`y`, and `z`).


*`type`* **`radians`** (*`type`* `deg`) <br> *`type`* **`degrees`** (*`type`* `rad`)

  : Convert degrees to radians or radians to degrees.


*`type`* **`cos`** (*`type`* `x`) <br> *`type`* **`sin`** (*`type`* `x`) <br> *`type`* **`tan`** (*`type`* `x`)

  : Computes the cosine, sine, or tangent of `x` (measured in radians).


`void` **`sincos`** (*`type`* `x`, `output` *`type`* `sinval`, `output` *`type`* `cosval`)

  : Computes both the sine and cosine of $x$ (measured in radians).  If both
    are needed, this function is less expensive than calling `sin()` and
    `cos()` separately.


*`type`* **`acos`** (*`type`* `x`) <br> *`type`* **`asin`** (*`type`* `y`) <br> *`type`* **`atan`** (*`type`* `y_over_x`) <br> *`type`* **`atan2`** (*`type`* `y`, *`type`* `x`)
  : Compute the principal value of the arc cosine, arc sine, and arc
    For `acos()` and `asin()`, the value of the argument
    will first be clamped to $[-1,1]$ to avoid invalid domain.

    For `acos()`, the result will always be in the range of $[0, \pi]$,
    and for `asin()` and `atan()`, the result will always be in the
    range of $[-\pi/2, \pi/2]$.  For `atan2()`, the signs of both
    arguments are used to determine the quadrant of the return value.


*`type`* **`cosh`** (*`type`* `x`) <br> *`type`* **`sinh`** (*`type`* `x`) <br> *`type`* **`tanh`** (*`type`* `x`)
  : Computes the hyperbolic cosine, sine, and tangent of $x$ (measured in
    radians).


*`type`* **`pow`** (*`type`* `x`, *`type`* `y`) <br> *`type`* **`pow`** (*`type`* `x, float y`)
  : Computes $x^y$.  This function will return 0 for "undefined" operations,
    such as `pow(-1,0.5)`.


*`type`* **`exp`** (*`type`* `x`) <br> *`type`* **`exp2`** (*`type`* `x`) <br> *`type`* **`expm1`** (*`type`* `x`)
  : Computes $e^x$, $2^x$, and $e^x-1$, respectively.  Note that `expm1(x)` is
    accurate even for very small values of $x$.


*`type`* **`log`** (*`type`* `x`) <br> *`type`* **`log2`** (*`type`* `x`) <br> *`type`* **`log10`** (*`type`* `x`) <br> *`type`* **`log`** (*`type`* `x, float b`)
  : Computes the logarithm of $x$ in base $e$, 2, 10, or arbitrary base $b$,
    respectively.


*`type`* **`logb`** (*`type`* `x`)
  : Returns the exponent of x, as a floating-point number.


*`type`* **`sqrt`** (*`type`* `x`) <br> *`type`* **`inversesqrt`** (*`type`* `x`)
  : Computes $\sqrt{x}$ and $1/\sqrt{x}$.  Returns 0 if $x<0$.


*`type`* **`cbrt`** (*`type`* `x`)
  : Computes $\sqrt[3]{x}$. The sign of the return value will match $x$.


`float` **`hypot`** `(float x, float y)` <br> `float` **`hypot`** `(float x, float y, float z)`
  : Computes $\sqrt{x^2+y^2}$ and $\sqrt{x^2+y^2+z^z}$, respectively.


*`type`* **`abs`** (*`type`* `x`) <br> *`type`* **`fabs`** (*`type`* `x`) 
  : Absolute value of $x$.  (The two functions are synonyms.)


*`type`* **`sign`** (*`type`* `x`)
  : Returns 1 if $x>0$, -1 if $x<0$, 0 if $x=0$.


*`type`* **`floor`** (*`type`* `x`) <br> *`type`* **`ceil`** (*`type`* `x`) <br> *`type`* **`round`** (*`type`* `x`) <br> *`type`* **`trunc`** (*`type`* `x`)

  : Various rounding methods: `floor` returns the largest integer less than or
    equal to $x$; `ceil` returns the smallest integer greater than or equal to
    $x$; `round` returns the closest integer to $x$, in either direction; and
    `trunc` returns the integral part of $x$ (equivalent to `floor` if $x>0$
    and `ceil` if $x<0$).


*`type`* **`fmod`** (*`type`* `a`, *`type`* `b`) <br> *`type`* **`mod`** (*`type`* `a`, *`type`* `b`)

  : The `fmod()` function returns the floating-point remainder of $a/b$, i.e.,
    is the floating-point equivalent of the integer `%` operator. It is nearly
    identical to the C or C++ `fmod` function, except that in OSL, `fmod(a,0)`
    returns `0`, rather than `NaN`.  Note that if $a < 0$, the return value
    will be negative.

    The `mod()` function returns $a - b*\mbox{floor}(a/b)$, which will
    always be a positive number or zero.

    As an example, `fmod(-0.25,1.0) = -0.25`, but `mod(-0.25,1.0) =
    0.75`.  For positive `a` they return the same value.

    For both functions, the *`type`* may be any of `float`, `point`,
    `vector`, `normal`, or `color`.


*`type`* **`min`** (*`type`* `a`, *`type`* `b`) <br> *`type`* **`max`** (*`type`* `a`, *`type`* `b`) <br> *`type`* **`clamp`** (*`type`* `x`, *`type`* `minval`, *`type`* `maxval`)
  : The `min()` and `max()` functions return the minimum or maximum,
    respectively, of a list of two or more values.  The `clamp` function
    returns

        min(max(x,minval),maxval)

    that is, the value $x$ clamped to the specified range.


*`type`* **`mix`** (*`type`* `x`, *`type`* `y`, *`type`* `alpha`) <br> *`type`* **`mix`** (*`type`* `x`, *`type`* `y, float alpha`)
  : The `mix` function returns a linear blending:
    $ x*(1-\alpha) + y*(\alpha) $


*`type`* **`select`** (*`type`* `x`, *`type`* `y`, *`type`* `cond`) <br> *`type`* **`select`** (*`type`* `x`, *`type`* `y, float cond`) <br> *`type`* **`select`** (*`type`* `x`, *`type`* `y, int cond`)
  : The `select` function returns `x` if `cond` is zero, or `y`
    if `cond` is nonzero. This is roughly equivalent to `(cond ? y : x)`,
    except that if `cond` is a component-based type (such as `color`), the
    selection happens on a component-by-component basis. It is presumed that
    the underlying implementation is not a true conditional and will not incur
    any branching penalty.


`int` **`isnan`** `(float x)` <br> `int` **`isinf`** `(float x)` <br> `int` **`isfinite`** `(float x)`
  : The `isnan()` function returns 1 if $x$ is a not-a-number (NaN) value, 0
    otherwise.  The `isinf()` function returns 1 if $x$ is an infinite (`Inf` or
    `-Inf`) value, 0 otherwise.  The `isfinite()` function returns 1 if $x$ is
    an ordinary number (neither infinite nor `NaN`), 0 otherwise.  

`float` **`erf`** `(float x)` <br> `float` **`erfc`** `(float x)`
  : The `erf()` function returns the error function 
    ${\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2}} dt$.
    The `erfc` returns the complementary error function `1-erf(x)`
    (useful in maintaining precision for large values of $x$).


(sec-stdlib-geom)=
## Geometric functions

*`ptype`* **`ptype`** `(float f)`<br> *`ptype`* **`ptype`** `(float x, float y, float z)`

  : Constructs a point-like value (*`ptype`* may be any of `point`, `vector`,
    or `normal`) from individual `float` values.  If constructed from a single
    `float`, the value will be replicated for $x$, $y$, and $z$.


*`ptype`* **`ptype`** `(string space, f)`<br> *`ptype`* **`ptype`** `(string space, float x, float y, float z)`

  : Constructs a point-like value (*`ptype`* may be any of `point`,
    `vector`, or `normal`) from individual `float` coordinates, relative
    to the named coordinate system.  In other words,
    ```
        point (space, x, y, z)
    ```
    is equivalent to
    ```
        transform (space, "common", point(x,y,z))
    ```
    (And similarly for `vector`/`normal`.)


`float` **`dot`** `(vector A, vector B)`

  : Returns the inner product of the two vectors (or normals), i.e., 
    $A \cdot B = A_x B_x + A_y B_y + A_z B_z$.


`vector` **`cross`** `(vector A, vector B)`

  : Returns the cross product of two vectors (or normals), i.e., $A \times B$.


`float` **`length`** `(vector V)`<br> `float` **`length`** `(normal V)`

  : Returns the length of a vector or normal.


`float` **`distance`** `(point P0, point P1)`

  : Returns the distance between two points.


`float` **`distance`** `(point P0, point P1, point Q)`

  : Returns the distance from `Q` to the closest point on the line segment
    joining `P0` and `P1`.


`vector` **`normalize`** `(vector V)`<br> `normal` **`normalize`** `(normal V)`

  : Return a vector in the same direction as $V$ but with length 1,
    that is, `V / length(V)`.


`vector` **`faceforward`** `(vector N, vector I, vector Nref)` <br> `vector` **`faceforward`** `(vector N, vector I)`

  : If `dot (Nref, I)` $<0$, returns `N`, otherwise returns `-N`. For the
    version with only two arguments, `Nref` is implicitly `Ng`, the true
    surface normal.  The point of these routines is to return a version of `N`
    that faces towards the camera --- in the direction "opposite" of `I`.

    To further clarify the situation, here is the implementation of
    `faceforward` expressed in OSL:

    ```
    vector faceforward (vector N, vector I, vector Nref)
    {
        return (I.Nref > 0) ? -N : N;
    }

    vector faceforward (vector N, vector I)
    {
        return faceforward (N, I, Ng);
    }
    ```


`vector` **`reflect`** `(vector I, vector N)`

  : For incident vector `I` and surface orientation `N`, returns the
    reflection direction `R = I - 2*(N.I)*N`.  Note that `N` must be
    normalized (unit length) for this formula to work properly.


`vector` **`refract`** `(vector I, vector N, float eta)`

  : For incident vector `I` and surface orientation `N`, returns the
    refraction direction using Snell's law. The `eta` parameter is the ratio
    of the index of refraction of the volume containing `I` divided by the
    index of refraction of the volume being entered. The result is not
    necessarily normalized and a zero-length vector is returned in the case of
    total internal reflection.  For reference, here is the equivalent OSL of
    the implementation:

    ```
    vector refract (vector I, vector N, float eta)
    {
        float IdotN = dot (I, N);
        float k = 1 - eta*eta * (1 - IdotN*IdotN);
        return (k < 0) ? vector(0,0,0) : (eta*I - N * (eta*IdotN + sqrt(k)));
    }
    ```


`void` **`fresnel`** `(vector I, normal N, float eta, output float Kr, output float Kt, output vector R, output vector T)` 

  : According to Snell's law and the Fresnel equations, `fresnel` computes the
    reflection and transmission direction vectors `R` and `T`, respectively,
    as well as the scaling factors for reflected and transmitted light, `Kr`
    and `Kt`.  The `I` parameter is the normalized incident ray, `N` is the
    normalized surface normal, and `eta` is the ratio of refractive index of
    the medium containing `I` to that on the opposite side of the surface.


`point` **`rotate`** `(point Q, float angle, point P0, point P1)` <br> `point` **`rotate`** `(point Q, float angle, vector axis)`

  : Returns the point computed by rotating point `Q` by `angle` radians about
    the axis that passes from point `P0` to `P1`, or about the `axis` vector
    centered on the origin.


*`ptype`* **`transform`** `(string tospace,` *`ptype`* `p)` <br> *`ptype`* **`transform`** `(string fromspace, string tospace,` *`ptype`* `p)` <br> *`ptype`* **`transform`** `(matrix Mto,` *`ptype`* `p)`

  : Transform a `point`, `vector`, or `normal` (depending on the type of the
    *ptype p* argument) from the coordinate system named by `fromspace` to the
    one named by `tospace`.  If `fromspace` is not supplied, `p` is assumed to
    be in "common" space coordinates, so the transformation will be from
    "common" space to `tospace`.  A $4 \times 4$ matrix may be passed directly
    rather than specifying coordinate systems by name.

    Depending on the type of the passed point `p`, different transformation
    semantics will be used.  A `point` will transform as a position, a
    `vector` as a direction without regard to positioning, and a `normal` will
    transform subtly differently than a `vector` in order to preserve
    orthogonality to the surface under nonlinear scaling.
    
    Technically, what happens is this: The *from* and
    *to* spaces determine a $4 \times 4$ matrix.  A `point` $(x,y,z)$
    will transform the 4-vector $(x,y,z,1)$ by the matrix; a `vector` will
    transform $(x,y,z,0)$ by the matrix; a `normal` will transform
    $(x,y,z,0)$ by the inverse of the transpose of the matrix.


`float` **`transformu`** `(string tounits, float x)` <br> `float` **`transformu`** `(string fromunits, string tounits, float x)`

  : Transform a measurement from `fromunits` to `tounits`.  If `fromunits` is
    not supplied, $x$ will be assumed to be in "common" space units.

    For length conversions, unit names may be any of: `"mm"`, `"cm"`,
    `"m"`, `"km"`, `"in"`, `"ft"`, `"mi"`, or the name of any
    coordinate system, including `"common"`, `"world"`, `"shader"`, or
    any other named coordinate system that the renderer knows about.

    For time conversions, units may be any of: `"s"`, `"frames"`, or
    `"common"` (which indicates whatever timing units the renderer is
    using).

    It is only valid to convert length units to other length units, or time
    units to other time units.  Attempts to convert length to time or vice
    versa will result in an error.  Don't even think about trying to convert
    monetary units to time.



## Color functions

`color` **`color`** `(float f)` <br> `color` **`color`** `(float r, float g, float b)`

  : Constructs a `color` from individual `float` values.  If constructed
    from a single `float`, the value will be replicated for `r`, $g$, and $b$.

`color` **`color`** `(string colorspace, f)` <br> `color` **`color`** `(string colorspace, float r, float g, float b)`

  : Constructs an RGB `color` that is equivalent to the individual `float`
    values in a named color space.  In other words,
    ```
        color (colorspace, r, g, b)
    ```
    is equivalent to
    ```
        transformc (colorspace, "rgb", color(r, g, b))
    ```

`float` **`luminance`** `(color rgb)`

  : Returns the linear luminance of the color `rgb`, which is
    implemented per the ITU-R standard as $0.2126 R + 0.7152 G + 0.0722 B$.

`color` **`blackbody`** `(float temperatureK)`

  : The `blackbody()` function returns the blackbody emission (the
    incandescent glow of warm bodies) expected from a material of the given
    temperature in Kelvin, in units of $W/m^2$. Note that `emission()` has
    units of radiance, so will require a scaling factor of $1/\pi$ on
    surfaces, and $1/4\pi$ on volumes to convert to $W/m^2/sr$.

`color` **`wavelength_color`** `(float wavelength_nm)`

  : Returns an RGB color corresponding as closely as possible to the
    perceived color of a pure spectral color of the given wavelength (in nm).

`color` **`transformc`** `(string fromspace, string tospace, color Cfrom)` <br> `color` **`transformc`** `(string tospace, color Cfrom)`

  : Transforms color `Cfrom` from color space `fromspace` to color space
    `tospace`.  If `fromspace` is not supplied, it is assumed to be
    transforming from "RGB" space.



## Matrix functions

`matrix` **`matrix`** `(float m00, float m01, float m02, float  m03,` <br> $~~~~~~~~~~~~~~~~~~~~~~~$ `float m10, float m11, float m12, float m13,` <br> $~~~~~~~~~~~~~~~~~~~~~~~$ `float m20, float m21, float m22, float m23,` <br> $~~~~~~~~~~~~~~~~~~~~~~~$ `float m30, float m31, float m32, float m33)`

  : Constructs a `matrix` from 16 individual `float` values, in row-major order.  

`matrix` **`matrix`** `(float f)`

  : Constructs a `matrix` with `f` in all diagonal components, 0 in all other
    components.  In other words, `matrix(1)` is the identity matrix, and
    `matrix(f)` is `f*matrix(1)`.

`matrix` **`matrix`** `(string fromspace, float m00, ..., float m33)` <br> `matrix` **`matrix`** `(string fromspace, float f)`

  : Constructs a `matrix` relative to the named space, multiplying it by the
    *space*-to-common transformation matrix.  If the coordinate system name is
    unknown, it will be assumed to be the identity matrix.

    Note that `matrix (space, 1)` returns the *space*-to-common transformation
    matrix. If the coordinate system name is unknown, it will be assumed to be
    the identity matrix.

`matrix` **`matrix`** `(string fromspace, string tospace)`

  : Constructs a `matrix` that can be used to transform coordinates from
    `fromspace` to `tospace`.  If either of the coordinate
    system names are unknown, they will be assumed to be the identity matrix.

`int` **`getmatrix`** `(string fromspace, string tospace, output matrix M)`

  : Sets `M` to the `matrix` that transforms coordinates from `fromspace` to
    `tospace`.  Return 1 upon success, or 0 if either of the coordinate system
    names are unknown (in which case `M` will not be modified).  This is very
    similar to the `matrix(from,to)` constructor, except that `getmatrix()`
    allows the shader to gracefully handle unknown coordinate system names.

`float` **`determinant`** `(matrix M)`

  : Computes the determinant of matrix `M`.

`matrix` **`transpose`** `(matrix M)`

  : Computes the transpose of matrix `M`.



## Pattern generation

`float` **`step`** `(float edge, float x)` <br> *`type`* **`step`** (*`type`* `edge`, *`type`* `x`)

  : Returns 0 if $x < {\mathit edge}$ and 1 if $x \ge {\mathit edge}$.

    The *`type`* may be any of of `float`, `color`, `point`, `vector`, or
    `normal`.  For `color` and `point`-like types, the computations are
    performed component-by-component (separately for $x$, $y$, and $z$).


`float` **`linearstep`** `(float edge0, float edge1, float x)` <br> *`type`* **`linearstep`** (*`type`* `edge0`, *`type`* `edge1`, *`type`* `x`)

  : Returns 0 if `x` $\le$ `edge0`, and 1 if `x` $\ge$ `edge1`, and performs a
    linear interpolation between 0 and 1 when `edge0` $<$ `x` $<$ `edge1`.
    This is equivalent to `step(edge0, x)` when `edge0 == edge1`. For `color`
    and `point`-like types, the computations are performed
    component-by-component (separately for $x$, $y$, and $z$).


`float` **`smoothstep`** `(float edge0, float edge1, float x)` <br> *`type`* **`smoothstep`** (*`type`* `edge0`, *`type`* `edge1`, *`type`* `x`)

  : Returns 0 if `x` $\le$ `edge0`, and 1 if `x` $\ge$ `edge1`, and performs a
    smooth Hermite interpolation between 0 and 1 when `edge0` $<$ `x` $<$
    `edge1`. This is useful in cases where you would want a thresholding
    function with a smooth transition.

    The *`type`* may be any of of `float`, `color`, `point`, `vector`, or
    `normal`.  For `color` and `point`-like types, the computations are
    performed component-by-component.


`float` **`smooth_linearstep`** `(float edge0, float edge1, float x, float eps)` <br> *`type`* **`smooth_linearstep`** (*`type`* `edge0`, *`type`* `edge1`, *`type`* `x`, *`type`* eps)

  : This function is strictly linear between `edge0 + eps` and `edge1 - eps`
    but smoothly ramps to 0 between `edge0 - eps` and `edge0 + eps`
    and smoothly ramps to 1 between `edge1 - eps` and `edge1 + eps`.
    It is 0 when `x` $\le$ `edge0-eps,` and 1 if `x` $\ge$ `edge1 + eps`,
    and performs a linear interpolation between 0 and 1 when
    `edge0` < x < `edge1`. For `color` and `point`-like types, the
    computations are performed component-by-component.


%## Noise functions


*`type`* **`noise`** `(string noisetype, float u, ...)` <br> *`type`* **`noise`** `(string noisetype, float u, float v, ...)` <br> *`type`* **`noise`** `(string noisetype, point p, ...)` <br> *`type`* **`noise`** `(string noisetype, point p, float t, ...)`

  : Returns a continuous, pseudo-random (but repeatable) scalar field defined
    on a domain of dimension 1 (`float`), 2 (2 `float`'s), 3 (`point`), or 4
    (`point` and `float`), with a return value of either 1D (`float`) or 3D
    (`color`, `point`, `vector`, or `normal`).

    The `noisename` specifies which of a variety of possible noise
    functions will be used:

    `"perlin", "snoise"`

    : A signed Perlin-like gradient noise with an output range of $[-1,1]$,
      approximate average value of $0$, and is exactly $0$ at integer lattice
      points.  This is equivalent to the `snoise()` function.

    `"uperlin", "noise"`
    : An unsigned Perlin-like gradient noise with an output range of $(0,1)$,
      approximate average value of $0.5$, and is exactly $0.5$ at integer
      lattice points.  This is equivalent to the `noise()` function (the
      one that doesn't take a name string).

    `"cell"`
    : A discrete function that is constant on $[i,i+1)$ for all integers $i$
      (i.e., `cellnoise(x) == cellnoise(floor(x))`), but has a different
      and uncorrelated value at every integer.  The range is $[0,1]$, its
      large-scale average is 0.5, and its values are evenly distributed over
      $[0,1]$.

    `"hash"`
    : A function that returns a different, uncorrelated (but deterministic and
      repeatable) value at every real input coordinate.  The range is $[0,1]$
      its large-scale average is 0.5, and its values are evenly distributed over
      $[0,1]$.

    `"simplex"`
    : A signed simplex noise with an output range of $[-1,1]$,
      approximate average value of $0$.

    `"usimplex"`
    : An unsigned simplex noise with an output range of $[0,1]$,
      approximate average value of $0.5$.

    `"gabor"`
    : A band-limited, filtered, sparse convolution noise based on the
      Gabor impulse function (see Lagae et al., SIGGRAPH 2012).  Our Gabor
      noise is designed to have somewhat similar frequency content and range
      as Perlin noise (range $[-1,1]$, approximately large-scale average of
      $0$).  It is significantly more expensive than Perlin noise, but its
      advantage is that it correctly filters automatically based on the input
      derivatives.  Gabor noise allows several optional parameters to the
      `noise()` call:

      `"anisotropic",` *int* <br> `"direction",` *vector*
      : If `anisotropic` is 0 (the default), Gabor noise
        will be isotropic.  If `anisotropic` is 1, the Gabor noise will
        be anisotropic with the 3D frequency given by the `direction` 
        vector (which defaults to `(1,0,0)`).  If `anisotropic` is
        2, a hybrid mode will be used which is anisotropic along the
        `direction` vector, but radially isotropic perpendicular to that
        vector.  The `direction` vector is not used if `anisotropic`
        is 0.
  
      `"bandwidth",` *float*
      : Controls the bandwidth for Gabor noise. The default is 1.0.
  
      `"impulses",` *float*
      : Controls the number of impulses per cell for Gabor noise.
        The default is 16.
  
      `"do_filter",` *int*
      : If `do_filter` is 0, no filtering/antialiasing will
        be performed.  The default is 1 (yes, do filtering).  There is probably
        no good reason to ever turn off the filtering, it is primarily to test
        that the filtering is working properly.
    
    Note that some of the noise varieties have an output range of $[-1,1]$
    but others have range $[0,1]$; some may automatically antialias their
    output (based on the derivatives of the lookup coordinates) and others
    may not, and various other properties may differ.  The user should be
    aware of which noise varieties are useful in various circumstances.

    A particular renderer's implementation of OSL may supply additional
    noise varieties not described here.

    The `noise()` functions take optional arguments after their
    coordinates, passed as token/value pairs (similarly to optional texture
    arguments).  Generally, such arguments are specific to the type of
    noise, and are ignored for noise types that don't understand them.


*`type`* **`pnoise`** (string noisetype, float u, float uperiod) <br> *`type`* **`pnoise`** (string noisetype, float u, float v, float uperiod, float vperiod) <br> *`type`* **`pnoise`** (string noisetype, point p, point pperiod) <br> *`type`* **`pnoise`** (string noisetype, point p, float t, point pperiod, float tperiod)

  : Periodic version of `noise()`, in which the domain wraps with the given
    period(s).  Generally, only integer-valued periods are supported.


*`type`* **`noise`** (float u) <br> *`type`* **`noise`** (float u, float v) <br> *`type`* **`noise`** (point p) <br> *`type`* **`noise`** (point p, float t)

*`type`* **`snoise`** (float u) <br> *`type`* **`snoise`** (float u, float v) <br> *`type`* **`snoise`** (point p) <br> *`type`* **`snoise`** (point p, float t)

  : The old `noise(...coords...)` function is equivalent to
    `noise("uperlin",...coords...)` and `snoise(...coords...)`
    is equivalent to `noise("perlin",...coords...)`.


*`type`* **`pnoise`** (float u, float uperiod) <br> *`type`* **`pnoise`** (float u, float v, float uperiod, float vperiod) <br> *`type`* **`pnoise`** (point p, point pperiod) <br> *`type`* **`pnoise`** (point p, float t, point pperiod, float tperiod) 

*`type`* **`psnoise`** (float u, float uperiod) <br> *`type`* **`psnoise`** (float u, float v, float uperiod, float vperiod) <br> *`type`* **`psnoise`** (point p, point pperiod) <br> *`type`* **`psnoise`** (point p, float t, point pperiod, float tperiod)

  : The old `pnoise(...coords...)` function is equivalent to
    `pnoise("uperlin",...coords...)` and `psnoise(...coords...)`
    is equivalent to `pnoise("perlin",...coords...)`.


*`type`* **`cellnoise`** `(float u)` <br> *`type`* **`cellnoise`** `(float u, float v)` <br> *`type`* **`cellnoise`** `(point p)` <br>
*`type`* **`cellnoise`** `(point p, float t)`

  : The old `cellnoise(...coords...)` function is equivalent to
    `noise("cell",...coords...)`.


*`type`* **`hashnoise`** `(float u)` <br> *`type`* **`hashnoise`** `(float u, float v)` <br> *`type`* **`hashnoise`** `(point p)` <br> *`type`* **`hashnoise`** `(point p, float t)`

  : Returns a deterministic, repeatable hash of the 1-, 2-, 3-, or 4-D
    coordinates.  The return values will be evenly distributed on $[0,1]$ and
    be completely repeatable when passed the same coordinates again, yet will
    be uncorrellated to hashes of any other positions (including nearby
    points).  This is like having a random value indexed spatially, but that
    will be repeatable from frame to frame of an animation (provided its input
    is *precisely* identical).


`int` **`hash`** `(float u)` <br> `int` **`hash`** `(float u, float v)` <br> `int` **`hash`** `(point p)` <br> `int` **`hash`** `(point p, float t)` <br> `int` **`hash`** `(int i)`

  : Returns a deterministic, repeatable integer hash of the 1-, 2-, 3-, or 4-D
    coordinates.


*`type`* **`spline`** `(string basis, float x,` *`type`* $\mathtt{y}_0$, *`type`* $\mathtt{y}_1$, ... *`type`* $\mathtt{y}_{n-1}$`)` <br> *`type`* **`spline`** `(string basis, float x,` *`type`* `y[])` <br> *`type`* **`spline`** `(string basis, float x, int nknots,` *`type`* `y[])`

  : As $x$ varies from 0 to 1, `spline` returns the value of a cubic
    interpolation of uniformly-spaced knots $y_0$...$y_{n-1}$, or
    $y[0]$...$y[n-1]$ for the array version of the call (where $n$ is the
    length of the array), or $y[0]$...$y[nknots-1]$ for the version that
    explicitly specifies the number of knots (which may be less than the
    full array length).  The input value $x$ will be clamped to lie
    on $[0,1]$.  The *`type`* may be any of `float`, `color`,
    `point`, `vector`, or `normal`; for multi-component types (e.g. `color`),
    each component will be interpolated separately.

    The type of interpolation is specified by the `basis` parameter,
    which may be any of: `"catmull-rom"`, `"bezier"`, `"bspline"`,
    `"hermite"`, `"linear"`, or `"constant"`. Some basis types require
    particular numbers of knot values -- Bezier splines
    require $3n+1$ values, Hermite splines require $2n+2$ values, and all of
    Catmull-Rom, linear, and constant requires $3+n$, where in all cases,
    $n \ge 1$ is the number of spline segments.

    To maintain consistency with the other spline types, `"linear"` splines will
    ignore the first and last data value; interpolating piecewise-linearly
    between $y_1$ and $y_{n-2}$, and `"constant"` splines ignore the first
    and the two last data values.


`float` **`splineinverse`** `(string basis, float v, float y`{sub}`0`, `... float y`{sub}`n-1` `)` <br> `float` **`splineinverse`** `(string basis, float v, float y[])` <br> `float` **`splineinverse`** `(string basis, float v, int nknots, float y[])`

  : Computes the *inverse* of the `spline()` function, i.e., returns
    the value $x$ for which

        spline (basis, x, y...)

    would return value $v$.  Results are undefined if the knots do not
    specify a monotonic (only increasing or only decreasing) set of values.

    Note that the combination of `spline()` and `splineinverse()` makes
    it possible to compute a full spline-with-nonuniform-abscissae:

        float v = splineinverse (basis, x, nknots, abscissa);
        result = spline (basis, v, nknots, value);



## Derivatives and area operators

`float` **`Dx`** `(float a)`, **`Dy`** `(float a)`, **`Dz`** `(float a) `<br> `vector` **`Dx`** `(point a)`, **`Dy`** `(point a)`, **`Dz`** `(point a)` <br> `vector` **`Dx`** `(vector a)`, **`Dy`** `(vector a)`, **`Dz`** `(vector a)`<br> `color` **`Dx`** `(color a)`, **`Dy`** `(color a)`, **`Dz`** `(color a)`

  : Compute an approximation to the partial derivatives of $a$ with respect to
    each of two principal directions, $\partial a / \partial x$ and
    $\partial a / \partial y$.  Depending on the renderer implementation,
    those directions may be aligned to the image plane, on the surface of the
    object, or something else.

    The `Dz` function is only meaningful for volumetric shading, and
    is expected to return `0` in other contexts.  It is also possible
    that particular OSL implementations may only return "correct" `Dz`
    values for particular inputs (such as `P`).

`float` **`filterwidth`** `(float x)` <br> `vector` **`filterwidth`** `(point x)` <br> `vector` **`filterwidth`** `(vector x)`

  : Compute differentials of the argument `x`, i.e., the approximate change in
    `x` between adjacent shading samples.


`float` **`area`** `(point p)`

  : Returns the differential area of position `p` corresponding to this
    shading sample.  If `p` is the actual surface position `P`, then `area(P)`
    will return the surface area of the section of the surface that is
    "covered" by this shading sample.

`vector` **`calculatenormal`** `(point p)`

  : Returns a vector perpendicular to the surface that is defined by point `p`
    (as `p` is computed at all points on the currently-shading surface),
    taking into account surface orientation.

`float` **`aastep`** `(float edge, float s)` <br> `float` **`aastep`** `(float edge, float s, float ds)` <br> `float` **`aastep`** `(float edge, float s, float dedge, float ds)`

  : Computes an antialiased step function, similar to `step(edge,s)` but
    filtering the edge to take into account how rapidly `s` and `edge` are
    changing over the surface.  If the differentials `ds` and/or `dedge` are
    not passed explicitly, they will be automatically computed (using
    `filterwidth()`).


## Displacement functions

`void` **`displace`** `(float amp)` <br> `void` **`displace`** `(string space, float amp)` <br> `void` **`displace`** `(vector offset)`

  : Displace the surface in the direction of the shading normal `N` by `amp`
    units as measured in the named `space` (or "common" space if none is
    specified).  Alternately, the surface may be moved by a fully general
    `offset`, which does not need to be in the direction of the surface
    normal.

    In either case, this function both displaces the surface and adjusts the
    shading normal `N` to be the new surface normal of the displaced surface
    (properly handling both continuously smooth surfaces as well as
    interpolated normals on faceted geometry, without introducing faceting
    artifacts).

`void` **`bump`** `(float amp)` <br> `void` **`bump`** `(string space, float amp)` <br> `void` **`bump`** `(vector offset)`

  : Adjust the shading normal `N` to be the surface normal as if the surface
    had been displaced by the given amount (see the `displace()` function
    description), but without actually moving the surface positions.


## String functions

`void` **`printf`** `(string fmt, ...)`

  : Much as in C, `printf` takes a format string `fmt` and an argument list,
    and prints the resulting formatted string to the console.

    Where the `fmt` contains a format string similar to `printf` in the C
    language. The `%d`, `%i`, `%o`, and `%x` arguments expect an `int`
    argument.  The `%f`, `%g`, and `%e` expect a `float`, `color`, point-like,
    or `matrix` argument (for multi-component types such as `color`, the
    format will be applied to each of the components).  The `%s` expects a
    `string` or `closure` argument.

    All of the substitution commands follow the usual C/C++ formatting rules,
    so format commands such as `"%6.2f"`, etc., should work as
    expected.

`string` **`format`** `(string fmt, ...)`

  : The `format` function works similarly to `printf`, except that instead of
    printing the results, it returns the formatted text as a `string`.

`void` **`error`** `(string fmt, ...)` <br> `void` **`warning`** `(string fmt, ...)`

  : The `error()` and `warning()` functions work similarly to `printf`, but
    the results will be printed as a renderer error or warning message,
    possibly including information about the name of the shader and the object
    being shaded, and other diagnostic information.

`void` **`fprintf`** `(string filename, string fmt, ...)`

  : The `fprintf()` function works similarly to `printf`, but rather than
    printing to the default text output stream, the results will be
    concatenated onto the end of the text file named by `filename`.

`string` **`concat`** `(string s1, ..., string sN)`

  : Concatenates a list of strings, returning the aggregate string.

`int` **`strlen`** `(string s)`

  : Return the number of characters in string `s`.

`int` **`startswith`** `(string s, string prefix)`

  : Return 1 if string `s` begins with the substring `prefix`, otherwise
    return 0.

`int` **`endswith`** `(string s, string suffix)`

  : Return 1 if string `s` ends with the substring `suffix`, otherwise return
    0.

`int` **`stoi`** `(string str)`

  : Convert/decode the initial part of `str` to an `int` representation.  Base
    10 is assumed. The return value will be `0` if the string doesn't appear
    to hold valid representation of the destination type.

`float` **`stof`** `(string str)`

  : Convert/decode the initial part of `str` to a `float` representation. The
    return value will be `0` if the string doesn't appear to hold valid
    representation of the destination type.

`string` **`substr`** `(string str, output string results[], string sep, int maxsplit)` <br> `string` **`substr`** `(string str, output string results[], string sep)` <br> `string` **`substr`** `(string str, output string results[])`

  : Fills the `result` array with the words in the string `str`, using `sep`
    as the delimiter string.  If `maxsplit` is supplied, at most `maxsplit`
    splits are done.  If `sep` is `""` (or if not supplied), any whitespace
    string is a separator.  The value returned is the number of elements
    (separated strings) written to the `results` array.

`string` **`substr`** `(string s, int start, int length)` <br> `string` **`substr`** `(string s, int start)`

  : Return at most `length` characters from `s`, starting with the character
    indexed by `start` (beginning with 0).  If `length` is omitted, return the
    rest of `s`, starting with `start`.  If `start` is negative, it counts
    backwards from the end of the string (for example, `substr(s,-1)` returns
    just the last character of `s`).

`int` **`getchar`** `(string s, int n)`

  : Returns the numeric value of the $n^{\mathrm{th}}$ character of the string,
    or `0` if `N` does not index a valid character of the string.

`int` **`hash`** `(string s)`

  : Returns a deterministic, repeatable hash of the string.

`int` **`regex_search`** `(string subject, string regex)` <br> `int` **`regex_search`** `(string subject, int results[], string regex)`

  : Returns 1 if any substring of `subject` matches a standard POSIX regular
    expression `regex`, 0 if it does not.

    In the form that also supplies a `results` array, when a match is
    found, the array will be filled in as follows: 

    `results[0]` 
      : the character index of the start of the sequence that matched the
       regular expression.

    `results[1]` 
      : the character index of the end (i.e., one past the last matching
        character) of the sequence that matched the regular expression. 

    `results[` $2i$ `]` 
      : the character index of the start of the sequence that matched
        sub-expression $i$ of the regular expression. 
      
    `results[` $2i+1$ `]` 
      : the character index of the end (i.e., one past the last matching
        character) of the sequence that matched sub-expression $i$ of the
        regular expression.

    Sub-expressions are denoted by surrounding them in parentheses in the
    regular expression.

    A few examples illustrate regular expression searching:

    ```
        r = regex_search ("foobar.baz", "bar");    //  = 1
        r = regex_search ("foobar.baz", "bark");   //  = 0

        int match[2];
        regex_search ("foobar.baz", match, "[Oo]{2}") = 1
                                          (match[0] == 1, match[1] == 3)
        substr ("foobar.baz", match[0], match[1]-match[0]) = "oo"

        int match[6];
        regex_search ("foobar.baz", match, "(f[Oo]{2}).*(.az)") = 1
        substr ("foobar.baz", match[0], match[1]-match[0]) = "foobar.baz"
        substr ("foobar.baz", match[2], match[3]-match[2]) = "foo"
        substr ("foobar.baz", match[4], match[5]-match[4]) = "baz"
    ```


`int` **`regex_match`** `(string subject, string regex)` <br> `int` **`regex_match`** `(string subject, int results[], string regex)`

  : Identical to `regex_search`, except that it must match the *whole*
    `subject` string, not merely a substring.




(sec-stdlib-texture)=
## Texture

*`type`* **`texture`** `(string filename, float s, float t,` *`...params...`* `)` <br> *`type`* **`texture`** `(string filename, float s, float t,` <br> $~~~~~~~~~~~~~~~~~~~~~~~$ `float dsdx, float dtdx, float dsdy, float dtdy,` *`...params...`* `)`

  : Perform a texture lookup of an image file, indexed by 2D coordinates
    (`s`,`t`), antialiased over a region defined by the differentials `dsdx`,
    `dtdx`, `dsdy` and `dtdy` (which are computed automatically from `s` and
    `t`, if not supplied).  Whether the results are assigned to a `float` or a
    `color` (or type cast to one of those) determines whether the texture
    lookup is a single channel or three channels.

    The 2D lookup coordinate(s) may be followed by optional key-value
    arguments (see Section [](#sec-syntax-functioncalls)) that control the behavior of `texture()`:

    `"blur",` *`float`*
      : Additional blur when looking up the texture value (default: 0).  The
        blur amount is relative to the size of the texture (i.e., 0.1 blurs by a
        kernel that is 10% of the full width and height of the texture).

        The blur may be specified separately in the `s` and `t` directions by
        using the `"sblur"` and `"tblur"` parameters, respectively.

    `"width",` *`float`*
      : Scale (multiply) the size of the filter as defined by the differentials
        (or implicitly by the differentials of `s` and `t`).  The default is
        `1`, meaning that no special scaling is performed.  A width of
        `0` would effectively turn off texture filtering entirely.

        The width value may be specified separately in the `s` and `t`
        directions by using the `"swidth"` and `"twidth"` parameters,
        respectively.

    `"wrap",` *`string`*
      : Specifies how the texture *wraps* coordinates outside the $[0,1]$
        range.  Supported wrap modes include: `"black"`, `"periodic"`,
        `"clamp"`, `"mirror"`, and `"default"` (which is the default).  A
        value of `"default"` indicates that the renderer should use any wrap
        modes specified in the texture file itself (a non-`"default"` value
        overrides any wrap mode specified by the file).

        The wrap modes may be specified separately in the `s` and `t` directions
        by using the `"swrap"` and `"twrap"` parameters, respectively.

    `"firstchannel",` *`int`*
      : The first channel to look up from the texture map (default: 0).

    `"subimage",` *`int`* <br> "`subimage",` *`string`*
      : Specify the subimage (by numerical index, or name) of the subimage
        within a multi-image texture file (default: subimage 0).

    `"fill",` *`float`*
      : The value to return for any channels that are requested,
        but not present in the texture file (default: 0).

    `"missingcolor",` *`color`*, `"missingalpha",` *`float`*
      : If present, supplies a missing color (and alpha value) that will
        be used for missing or broken textures -- *instead of* treating
        it as an error.  If you want a missing or broken texture to be reported
        as an error, you must not supply the optional `"missingcolor"`
        parameter.

    `"alpha",` *`floatvariable`*
      : The alpha channel (presumed to be the next channel following the
        channels returned by the `texture()` call) will be stored in the
        variable specified.  This allows for RGBA lookups in a single call to
        `texture()`.

    `"errormessage",` *`stringvariable`*
      : If this option is supplied, any error messages generated by the texture
        system will be stored in the specified variable rather than issuing error
        calls to the renderer, thus leaving it up to the shader to handle any
        errors. The error message stored will be `""` if no error occurred.

    `"interp",` *`string`*
      : Overrides the texture interpolation method: `"smartcubic"` (the 
        default), `"cubic"`, `"linear"`, or `"closest"`.


*`type`* **`texture3d`** `(string filename, point p,` *`...params...`* `)` <br> *`type`* **`texture3d`** `(string filename, point p, vector dpdx, vector dpdy, vector dpdz,` *`...params...`* `)`

  : Perform a 3D lookup of a volume texture, indexed by 3D coordinate
    `p`, antialiased over a region defined by the differentials
    `dpdx`, `dpdy`, and `dpdz` (which are computed
    automatically from `p`, if not supplied).  Whether the results
    are assigned to a `float` or a `color` (or type cast to one of those)
    determines whether the texture lookup is a single channel or three
    channels.

    The `p` coordinate (and `dpdx`, `dpdy`, and `dpdz`
    derivatives, if supplied) are assumed to be in "common" space and will be
    automatically transformed into volume local coordinates, if such a
    transormation is specified in the volume file itself.

    The 3D lookup coordinate may be followed by optional token/value
    arguments that control the behavior of `texture3d()`:

    `"blur"`, *`float`*
      : Additional blur when looking up the texture value (default: 0).  The
        blur amount is relative to the size of the texture (i.e., 0.1 blurs by a
        kernel that is 10% of the full width, height, and depth of the texture).

        The blur may be specified separately in the `s`, `t`, and `r`
        directions by using the `"sblur"`, `"tblur"`, and `"rblur"`
        parameters, respectively.

    `"width"`, *`float`*
      : Scale (multiply) the size of the filter as defined by the differentials
        (or implicitly by the differentials of `s`, `t`, and `r`).  The default is
        `1`, meaning that no special scaling is performed.  A width of
        `0` would effectively turn off texture filtering entirely.

        The width value may be specified separately in the `s`, `t`, and `r`
        directions by using the `"swidth"`, `"twidth"`, and `"rwidth"`    parameters,
        respectively.

    `"wrap"`, *`string`*
      : Specifies how the texture *wraps* coordinates outside the $[0,1]$
        range.  Supported wrap modes include: `"black"`, `"periodic"`,
        `"clamp"`, `"mirror"`, and `"default"` (which is the default).  A
        value of `"default"` indicates that the renderer should use any wrap
        modes specified in the texture file itself (a non-`"default"` value
        overrides any wrap mode specified by the file).

        The wrap modes may be specified separately in the `s`, `t`, and `r`
        directions by using the `"swrap"`, `"twrap"`, and `"rwrap"`
        parameters, respectively.

    `"firstchannel"`, *`int`*
      : The first channel to look up from the texture map (default: 0).

    `"subimage"`, *`int`* <br> `"subimage"`, *`string`*
      : Specify the subimage (by numerical index, or name) of the subimage
        within a multi-image texture file (default: subimage 0).

    `"fill"`, *`float`*
      : The value to return for any channels that are requested,
        but not present in the texture file (default: 0).

    `"missingcolor"`, *`color`*, `"missingalpha"`, *`float`*
      : If present, supplies a missing color (and alpha value) that will
        be used for missing or broken textures -- *instead of* treating
        it as an error.  If you want a missing or broken texture to be reported
        as an error, you must not supply the optional ``"missingcolor"``
        parameter.

    `"time"`, *`float`*
      : A time value to use if the volume texture specifies a time-varying
        local transformation (default: 0).

    `"alpha"`, *`floatvariable`*
      : The alpha channel (presumed to be the next channel following the
        channels returned by the `texture3d()` call) will be stored in the
        variable specified.  This allows for RGBA lookups in a single call to
        `texture3d()`.

    `"errormessage"`, *'stringvariable'*
      : If this option is supplied, any error messages generated by the texture
        system will be stored in the specified variable rather than issuing error
        calls to the renderer, thus leaving it up to the shader to handle any
        errors. The error message stored will be `""` if no error occurred.


*`type`* **`environment`** `(string filename, vector R,` *`...params...`* `)` <br> *`type`* **`environment`** `(string filename, vector R, vector dRdx, vector dRdy,` *`...params...`* `)`

  : Perform an environment map lookup of an image file, indexed by direction
    `R`, antialiased over a region defined by the differentials `dRdx`, `dRdy`
    (which are computed automatically from `R`, if not supplied). Whether the
    results are assigned to a `float` or a `color` (or type cast to one of
    those) determines whether the texture lookup is a single channel or three
    channels.

    The lookup direction (and optional derivatives) may be followed by optional
    token/value arguments that control the behavior of `environment()`:

    "blur", *`float`*
      : Additional blur when looking up the texture value (default: 0).  The
        blur amount is relative to the size of the texture (i.e., 0.1 blurs by a
        kernel that is 10% of the full width and height of the texture).

        The blur may be specified separately in the `s` and `t` directions by
        using the `"sblur"` and `"tblur"` parameters, respectively.

    "width", *`float`*
      : Scale (multiply) the size of the filter as defined by the differentials
        (or implicitly by the differentials of `s` and `t`).  The default is
        `1`, meaning that no special scaling is performed.  A width of
        `0` would effectively turn off texture filtering entirely.

        The width value may be specified separately in the `s` and `t`
        directions by using the `"swidth"` and `"twidth"` parameters,
        respectively.

    "firstchannel", *`int`*
      : The first channel to look up from the texture map (default: 0).

    "fill", *`float`*
      : The value to return for any channels that are requested, but 
        not present in the texture file (default: 0).

    "missingcolor", *`color`*, "missingalpha", *`float`*
      : If present, supplies a missing color (and alpha value) that will
        be used for missing or broken textures -- *instead of* treating
        it as an error.  If you want a missing or broken texture to be reported
        as an error, you must not supply the optional `"missingcolor"`
        parameter.

    "alpha", *`floatvariable`*
      : The alpha channel (presumed to be the next channel following the
        channels returned by the `environment()` call) will be stored in the
        variable specified.  This allows for RGBA lookups in a single call to
        `environment()`.

    "errormessage", *`stringvariable`*
      : If this option is supplied, any error messages generated by the texture
        system will be stored in the specified variable rather than issuing error
        calls to the renderer, thus leaving it up to the shader to handle any
        errors. The error message stored will be `""` if no error occurred.


`int` **`gettextureinfo`**` (string texturename, string paramname, output` *`type`* `destination)` <br> `int` **`gettextureinfo`** `(string texturename, float s, float t, string paramname, output` *`type`* `destination)`

  : Retrieves a parameter from a named texture file.  If the file is found,
    and has a parameter that matches the name and type specified, its value
    will be stored in `destination` and `gettextureinfo()` will
    return `1`.  If the file is not found, or doesn't have a matching
    parameter (including if the type does not match), `destination`
    will not be modified and `gettextureinfo()` will return `0`.

    The version of `gettextureinfo()` that takes `s` and `t` parameters
    retrieves information about the texture file that will be used for those
    texture coordinates. This can be useful for UDIM textures that may use
    different texture files for different regions, based on the corodinates. For
    regular, non-UDIM textures, the coordinates, if supplied, will be ignored.
    When UDIM textures are queried without coordinates supplied, it will succeed
    and return the texture info only if that parameter is found and has the same
    value in all files comprising the UDIM set. (Note: the version with
    coordinates was added in OSL 1.12.)

    Valid parameters recognized are listed below:

    Name | Type | Description
    :--- | :--- | :----------
    `"exists"` | `int` | Result is 1 if the file exists and is an texture format that OSL can read, or 0 if the file does not exist, or could not be properly read as a texture. Note that unlike all other queries, this query will "succeed" (return `1`) if the file does not exist.
    `"resolution"` | `int[2]` | The resolution ($x$ and $y$) of the highest MIPmap level stored in the texture map.
    `"resolution"` | `int[3]` | The resolution ($x$, $y$, and $z$) of the highest MIPmap level stored in the 3D texture map.  If it isn't a volumetric texture, the third component (`z` resolution) will be 1.
    `"channels"` | `int` | The number of channels in the texture map.
    `"type"` | `string` | Returns the semantic type of the texture, one of: `"Plain Texture"`, `"Shadow"`, `"Environment"`, `Volume Texture"`.
    `"subimages"` | `int` | Returns the number of subimages in the texture file.
    `"textureformat"` | `string` | Returns the texture format, one of: "`Plain Texture`", "`Shadow`", "`CubeFace Shadow`", "`Volume Shadow`", "`CubeFace Environment`", "`LatLong Environment`", "`Volume Texture`". Note that this differs from `"type"` in that it specifically distinguishes between the different types of shadows and environment maps. 
    `"datawindow"` | `int[]` | Returns the pixel data window of the image.  The argument is an `int` array either of length 4 or 6, in which will be placed the (xmin, ymin, xmax, ymax) or (xmin, ymin, zmin, xmax, ymax, zmax), respectively. (N.B. the `z` values may be useful for 3D/volumetric images; for 2D images they will be 0).
    `"displaywindow"` | `int[]` | Returns the display (a.k.a. full) window of the image.  The argument is an `int` array either of length 4 or 6, in which will be placed the (xmin, ymin, xmax, ymax) or (xmin, ymin, zmin, xmax, ymax, zmax), respectively. (N.B. the `z` values may be useful for 3D/volumetric images; for 2D images they will be 0).
    `"worldtocamera"` | `matrix` | If the texture is a rendered image, retrieves the world-to-camera 3D transformation matrix that was used when it was created.
    `"worldtoscreen"` | `matrix` | If the texture is a rendered image, retrieves the matrix that projected points from world space into a 2D screen coordinate system where $x$ and $y$ range from $-1$ to $+1$.
    `"averagecolor"` | `color` | Retrieves the average color (first three channels) of the texture.
    `"averagealpha"` | `float` | Retrieves the average alpha (the channel with `"A"` name) of the texture.
    *anything else* | *any* | Searches for matching name and type in the metadata or other header information of the texture file.


`int` `pointcloud_search` `(string ptcname, point pos, float radius, int maxpoints, [int sort,] string attr, Type data[], ..., string attrN, Type dataN[] )`

  : Search the named point cloud for the `maxpoints` closest points to `pos`
    within the given `radius`, returning the values of any named attributes of
    those points in the the given `data` arrays.  If the optional `sort`
    parameter is present and is nonzero, the ordering of the points found will
    be sorted by distance from `pos`, from closest to farthest; otherwise, the
    results are guaranteed to be the `maxpoints` closest to `pos`, but not
    necessarily sorted by distance (this may be faster for some
    implementations than when sorted results are required).  The return value
    is the number of points returned, ranging from 0 (nothing found in the
    neighborhood) to the lesser of `maxpoints` and the actual lengths of the
    arrays (the arrays will never be written beyond their actual length).

    These attribute names are reserved:

    | Name | Type | Description |
    | :--- | :--- | :---------- |
    | "`position`" | `point` | The position of each point |
    | "`distance`" | `float` | The distance between the point and `pos` |
    | "`index`"    | `int`   | The point's unique index within the cloud |

    Note that the named point cloud will be created, if it does not yet
    exist in memory, and that it will be initialized by reading a point
    cloud from disk, if there is one matching the name.

    Generally, the element type of the data arrays must match exactly the type
    of the point data attribute, or else you will get a runtime error. But there
    are two exceptions: (1) "triple" types (`color`, `point`, `vector`, `normal`)
    are considered interchangeable; and (2) it is legal to retrieve `float`
    arrays (e.g., a point cloud attribute that is `float[4]`) into a regular
    array of `float`, and the results will simply be concatenated into the
    larger array (which must still be big enough, in total, to hold
    `maxpoints` of the data type in the file).

    Example:

    ```
          float r = 3.0;
          point pos[10];
          color col[10];
          int n = pointcloud_search ("particles.ptc", P, r, 10,
                                     "position", pos, "color", col);
          printf ("Found %d particles within radius %f of (%p)\n", r, P);
          for (int i = 0;  i < n;  ++i)
              printf ("  position (%f) -> color (%g)\n", pos[i], col[i]);
    ```


`int` **`pointcloud_get`** `(string ptcname, int indices[], int count, string attr,` *`type`* `data[])`

  : Given a point cloud and a list of points `indices[0..count-1]`,
    store the attribute named by `attr` for each point, respectively, in
    `data[0..count-1]`.  Return 1 if successful, 0 for failure, which 
    could include the attribute not matching the type of `data`, invalid
    indices, or an unknown point cloud file.

    This can be used in conjunction with `pointcloud_search()`, as
    in the following example:

    ```
        float r = 3.0;
        int indices[10];
        int n = pointcloud_search ("particles.ptc", P, r, 10,
                                   "index", indices);
        float temp[10];         // presumed to be "float" attribute
        float quaternions[40];  // presumed to be "float[4]" attribute
        int ok = pointcloud_get ("particles.ptc", indices, n,
                                 "temperature", temp,
                                 "quat", quaternions);
    ```

    As with `pointcloud_search`, the element type of the data array must
    either be equivalent to the point cloud attribute being retrieved, or else
    when retrieving `float` arrays (e.g., a point cloud attribute
    that is `float[4]`) into a regular array of `float`, and the
    results will simply be concatenated into the larger array (which must
    still be big enough, in total, to hold `maxpoints` of the data type
    in the file).


`int` **`pointcloud_write`** `(string ptcname, point pos, string attr1,` *`type`* `data1, ...)`

  : Save the tuple (`attr1`, `data1`, ..., `attrN`, `dataN`) at position `pos`
    in a named point cloud.  The point cloud will be saved when the frame is
    finished computing.  Return 1 if successful, 0 for failure, which could
    include the attributes not matching names or types at different positions
    in the point cloud.

    Example:
    ```
          color C = ...;
          int ok = pointcloud_write ("particles.ptc", P, "normal", N, "color", C);
    ```



## Material Closures

For `closure color` functions, the return "value" is symbolic and may be
passed to an output variable or assigned to `Ci`, to be evaluated at a later
time convenient to the renderer in order to compute the exitant radiance in
the direction `-I`.  But the shader itself cannot examine the numeric values
of the `closure color`.

The intent of this specification is to give a minimal but useful set of
material closures that you can expect any renderer implementation to provide.
Individual renderers may supply additional closures that are specific to the
workings of that renderer.  Additionally, individual renderers may allow
additional parameters or controls on the standard closures, passed as
token/value pairs following the required arguments (much like the optional
arguments to the `texture()` function). Consult the documentation for your
specific renderer for details.

OSL's standard material closures are by synchronized to match the names and
properties of the physically-based shading nodes of MaterialX v1.38
(https://www.materialx.org/).

### Surface BSDF closures

`closure color` **`oren_nayar_diffuse_bsdf`** `(normal N, color albedo, float roughness, int energy_compensation=0)`

  : Constructs a diffuse reflection BSDF based on the Oren-Nayar reflectance
    model.

    Parameters include:

    `N`
      : Normal vector of the surface point being shaded.

    `albedo`
      : Surface albedo.

    `roughness`
      : Surface roughness [0,1]. A value of 0.0 gives Lambertian reflectance.

    `energy_compensation`
      : Optional int parameter to select if energy compensation should be applied.

    The Oren-Nayar reflection model is described in  M. Oren and S. K.
    Nayar, "Generalization of Lambert's Reflectance Model," Proceedings of
    SIGGRAPH 1994, pp.239-246 (July, 1994).

    The energy compensated model is described in the white paper: "An energy-preserving Qualitative Oren-Nayar model" by Jamie Portsmouth.

`closure color` **`burley_diffuse_bsdf`** `(normal N, color albedo, float roughness)`

  : Constructs a diffuse reflection BSDF based on the corresponding component of 
    the Disney Principled shading model.

    Parameters include:

    `N`
      : Normal vector of the surface point being shaded.

    `albedo`
      : Surface albedo.

    `roughness`
      : Surface roughness [0,1]. A value of 0.0 gives Lambertian reflectance.
    
      %     
      % \apiitem{label}
      %   % Optional string parameter to name this component. For use in AOVs / LPEs.
      % 


`closure color` **`dielectric_bsdf`** `(normal N, vector U, color reflection_tint,  color transmission_tint, float roughness_x, float roughness_y,  float ior, string distribution)`

  : Constructs a reflection and/or transmission BSDF based on a microfacet
    reflectance model and a Fresnel curve for dielectrics. The two tint
    parameters control the contribution of each reflection/transmission lobe.
    The tints should remain 100% white for a physically correct dielectric,
    but can be tweaked for artistic control or set to 0.0 for disabling a
    lobe.

    The closure may be vertically layered over a base BSDF for the surface
    beneath the dielectric layer. This is done using the `layer()` closure. By
    chaining multiple `dielectric_bsdf` closures you can describe a surface
    with multiple specular lobes. If transmission is enabled (`transmission_tint`
    $>$ 0.0) the closure may be layered over a VDF closure describing the surface
    interior to handle absorption and scattering inside the medium.

    Parameters include:

    `N`
      : Normal vector of the surface point being shaded.

    `U`
      : Tangent vector of the surface point being shaded.

    `reflection_tint`
      : Weight per color channel for the reflection lobe. Should be (1,1,1) for a physically-correct dielectric surface, but can be tweaked for artistic control. Set to (0,0,0) to disable reflection.

    `transmission_tint`
      : Weight per color channel for the transmission lobe. Should be (1,1,1) for a physically-correct dielectric surface, but can be tweaked for artistic control. Set to (0,0,0) to disable transmission.

    `roughness_x`
      : Surface roughness in the U direction with a perceptually linear response over its range.

    `roughness_y`
      : Surface roughness in the V direction with a perceptually linear response
      over its range.

    `ior`
      : Refraction index.

    `distribution`
      : Microfacet distribution. An implementation is expected to support the
      following distributions: `"ggx"`

    `thinfilm_thickness`
      : Optional float parameter for thickness of an iridescent thin film layer on top of this BSDF. Given in nanometers.

    `thinfilm_ior`
      : Optional float parameter for refraction index of the thin film layer.
    
      %     
      % \apiitem{label}
      %   % Optional string parameter to name this component. For use in AOVs / LPEs.
      % 


`closure color` **`conductor_bsdf`** `(normal N, vector U, float roughness_x, float roughness_y, color ior, color extinction, string distribution)`

  : Constructs a reflection BSDF based on a microfacet reflectance model. Uses a
    Fresnel curve with complex refraction index for conductors/metals. If an
    artistic parametrization is preferred the `artistic_ior()` utility
    function can be used to convert from artistic to physical parameters.

    Parameters include:

    `N`
      : Normal vector of the surface point being shaded.

    `U`
      : Tangent vector of the surface point being shaded.

    `roughness_x`
      : Surface roughness in the U direction with a perceptually linear response over its range.

    `roughness_y`
      : Surface roughness in the V direction with a perceptually linear response over its range.

    `ior`
      : Refraction index.

    `extinction`
      : Extinction coefficient.

    `distribution`
      : Microfacet distribution. An implementation is expected to support the following distributions: `"ggx"`

    `thinfilm_thickness`
      : Optional float parameter for thickness of an iridescent thin film layer on top of this BSDF. Given in nanometers.

    `thinfilm_ior`
      : Optional float parameter for refraction index of the thin film layer.
    
      %     
      % \apiitem{label}
      %   % Optional string parameter to name this component. For use in AOVs / LPEs.
      % 


`closure color` **`generalized_schlick_bsdf`**` (normal N, vector U, color reflection_tint, color transmission_tint, float roughness_x, float roughness_y, color f0, color f90, float exponent, string distribution)`

  : Constructs a reflection and/or transmission BSDF based on a microfacet
    reflectance model and a generalized Schlick Fresnel curve. The two tint
    parameters control the contribution of each reflection/transmission lobe.

    The closure may be vertically layered over a base BSDF for the surface beneath
    the dielectric layer. This is done using the layer() closure. By chaining
    multiple `dielectric_bsdf` closures you can describe a surface with multiple
    specular lobes. If transmission is enabled (`transmission_tint` $>$ 0.0) the
    closure may be layered over a VDF closure describing the surface interior to
    handle absorption and scattering inside the medium.

    Parameters include:

    `N`
      : Normal vector of the surface point being shaded.

    `U`
      : Tangent vector of the surface point being shaded.

    `reflection_tint`
      : Weight per color channel for the reflection lobe. Set to (0,0,0) to disable reflection.

    `transmission_tint`
      : Weight per color channel for the transmission lobe. Set to (0,0,0) to disable transmission.

    `roughness_x`
      : Surface roughness in the U direction with a perceptually linear response over its range.

    `roughness_y`
      : Surface roughness in the V direction with a perceptually linear response over its range.

    `f0`
      : Reflectivity per color channel at facing angles.

    `f90`
      : Reflectivity per color channel at grazing angles.

    `exponent`
      : Variable exponent for the Schlick Fresnel curve, the default value should be 5.

    `distribution`
      : Microfacet distribution. An implementation is expected to support the following distributions: `"ggx"`

    `thinfilm_thickness`
      : Optional float parameter for thickness of an iridescent thin film layer on top of this BSDF. Given in nanometers.

    `thinfilm_ior`
      : Optional float parameter for refraction index of the thin film layer.
    
      %     
      % \apiitem{label}
      %   % Optional string parameter to name this component. For use in AOVs / LPEs.
      % 
  

`closure color` **`translucent_bsdf`** `(normal N, color albedo)`

  : Constructs a translucent (diffuse transmission) BSDF based on the Lambert
    reflectance model.

    Parameters include:

    `N`
      : Normal vector of the surface point being shaded.
    
    `albedo`
      : Surface albedo.
    
    `roughness`
      : Surface roughness [0,1]. A value of 0.0 gives Lambertian reflectance.
    
      %     
      % \apiitem{label}
      %   % Optional string parameter to name this component. For use in AOVs / LPEs.
      % 
  

`closure color` **`transparent_bsdf`** `( )`

  : Constructs a closure that represents straight transmission through a surface.


`closure color` **`subsurface_bssrdf`** `( )`

  : Constructs a BSSRDF for subsurface scattering within a homogeneous medium.

    Parameters include:

    `N`
      : Normal vector of the surface point being shaded.
    
    `albedo`
      : Single-scattering albedo of the medium.
    
    `transmission_depth`
      : Distance travelled inside the medium by white light before its color becomes transmission_color by Beer's law. Given in scene length units, range [0,infinity). Together with transmission_color this determines the extinction coefficient of the medium.
    
    `transmission_color`
      : Desired color resulting from white light transmitted a distance of 'transmission_depth' through the medium. Together with transmission_depth this determines the extinction coefficient of the medium.
    
    `anisotropy`
      : Scattering anisotropy [-1,1]. Negative values give backwards scattering, positive values give forward scattering, and 0.0 gives uniform scattering.
    
    %     
      % \apiitem{label}
      %   % Optional string parameter to name this component. For use in AOVs / LPEs.
      % 
  

`closure color` **`sheen_bsdf`** `(normal N, color albedo, float roughness)`

  : Constructs a microfacet BSDF for the back-scattering properties of cloth-like
    materials. This closure may be vertically layered over a base BSDF, where
    energy that is not reflected will be transmitted to the base closure.

    Parameters include:

    `N`
      : Normal vector of the surface point being shaded.
              
    `albedo`
      : Surface albedo.
    
    `roughness`
      : Surface roughness [0,1].
    
      %     
      % \apiitem{label}
      %   % Optional string parameter to name this component. For use in AOVs / LPEs.
      % 
  


### Volumetric material closures


`closure color` **`anisotropic_vdf`** `(color albedo, color extinction, float anisotropy)`

  : Constructs a VDF scattering light for a general participating medium, based on
    the Henyey-Greenstein phase function. Forward, backward and uniform scattering
    is supported and controlled by the anisotropy input.

    Parameters include:

    `albedo`
      : Single-scattering albedo of the medium.

    `extinction`
      : Volume extinction coefficient.

    `anisotropy`
      : Scattering anisotropy [-1,1]. Negative values give backwards scattering,
        positive values give forward scattering, and 0.0 gives uniform scattering.

    %     
      % \apiitem{label}
      %   % Optional string parameter to name this component. For use in AOVs / LPEs.
      % 


`closure color` **`medium_vdf`** `(color albedo, float transmission_depth, color transmission_color, float anisotropy, float ior, int priority)`

  : Constructs a VDF for light passing through a dielectric homogeneous medium,
    such as glass or liquids. The parameters `transmission_depth` and
    `transmission_color` control the extinction coefficient of the medium in an
    artist-friendly way. A priority can be set to determine the ordering of
    overlapping media.

    Parameters include:

    `albedo`
       : Single-scattering albedo of the medium.

    `transmission_depth`
       : Distance travelled inside the medium by white light before its color becomes
        transmission_color by Beer's law. Given in scene length units, range
        [0,infinity). Together with transmission_color this determines the
        extinction coefficient of the medium.

    `transmission_color`
       : Desired color resulting from white light transmitted a distance of
        'transmission_depth' through the medium. Together with transmission_depth
        this determines the extinction coefficient of the medium.

    `anisotropy`
       : Scattering anisotropy [-1,1]. Negative values give backwards scattering,
        positive values give forward scattering, and 0.0 gives uniform scattering.

    `ior`
       : Refraction index of the medium.

    `priority`
       : Priority of this medium (for nested dielectrics).

    %     
      % \apiitem{label}
      %   % Optional string parameter to name this component. For use in AOVs / LPEs.
      % 
  


### Light emission closures

`closure color` **`uniform_edf`** `(color emittance)`

  : Constructs an EDF emitting light uniformly in all directions. This is used to
    represent a glowing/emissive material. When called in the context of a surface
    shader group, it implies that light is emitted in a full hemisphere centered
    around the surface normal. When called in the context of a volume shader
    group, it implies that light is emitted evenly in all directions around the
    point being shaded.

    The `emittance` parameter is the amount of emission and has units of radiance
    (e.g., $\mathrm{W}\cdot\mathrm{sr}^{-1}\cdot\mathrm{m}^{-2}$). This means that
    a surface directly seen by the camera will directly reproduce the closure
    weight in the final pixel, regardless of being a surface or a volume.

    For an emissive surface, if you divide the return value of `uniform_edf()` by
    `surfacearea() * M_PI`, then you can easily specify the total emissive power
    of the light (e.g., $\mathrm{W}$), regardless of its physical size.


### Layering and Signaling closures

`closure color` **`layer`** `(closure color top, closure color base)`

  : Vertically layer a layerable BSDF such as `dielectric_bsdf`,
    `generalized_schlick_bsdf` or `sheen_bsdf` over a BSDF or VDF. The
    implementation is target specific, but a standard way of handling this is by
    albedo scaling, using `base*(1-reflectance(top)) + top`, where
    `reflectance()` calculates the directional albedo of a given top BSDF.


`closure color` **`holdout`** `( )`

  : Returns a `closure color` that does not represent any additional light
    reflection from the surface, but does signal to the renderer that 
    the surface is a *holdout object* (appears transparent in
    the final output yet hides objects behind it).  "Partial holdouts"
    may be designated by weighting the `holdout()` closure by a
    weight that is less than 1.0.


`closure color` **`debug`** `(string outputname)`

  : Returns a `closure color` that does not represent any additional light
    reflection from the surface, but does signal to the renderer to add
    the weight of the closure (which may be a `float` or a `color`)
    to the named output (i.e., AOV).



### Material utility functions

`void` **`artistic_ior`** `(color reflectivity, color edge_tint, output color ior, output color extinction)`

  : Converts the artistic parameterization reflectivity and edge_tint to
    complex IOR values. To be used with the `conductor_bsdf()` closure.

    Parameters include:

    `reflectivity`
      : Reflectivity per color channel at facing angles ($r$ parameter in [OG14])

    `edge_tint`
      : Color bias for grazing angles ($g$ parameter in [OG14]).
        NOTE: This is not equal to 'f90' in a Schlick Fresnel parameterization.

    `ior`
      : Output refraction index.
    
    `extinction`
      : Output extinction coefficient.

    Reference: [OG14] Ole Gulbrandsen, "Artist Friendly Metallic Fresnel",
    Journal of Computer Graphics Tools 3(4), 2014.
    http://jcgt.org/published/0003/04/03/paper.pdf


### Deprecated closures

These were described in the original OSL language specification, but
beginning with OSL 1.12, these are considered deprecated. Support for
them will be removed entirely in OSL 2.0.

#### Deprecated Surface closures

`closure color` **`diffuse`** `(normal N)`

  : Returns a `closure color` that represents the Lambertian diffuse
    reflectance of a smooth surface,

    $$ \int_{\Omega}{\frac{1}{\pi} \max(0, N \cdot \omega) Cl(P,\omega) d\omega} $$

    where $N$ is the unit-length forward-facing surface normal at `P`,
    $\Omega$ is the set of all outgoing directions in the hemisphere
    surrounding $N$, and $Cl(P,\omega)$ is the incident radiance at
    `P` coming from the direction $-\omega$.


`closure color` **`phong`** `(normal N, float exponent)`

  : Returns a `closure color` that represents specular reflectance of the
    surface using the Phong BRDF.  The `exponent` parameter
    indicates how smooth or rough the material is (higher `exponent`
    values indicate a smoother surface).


`closure color` **`oren_nayar`** `(normal N, float sigma)`

  : Returns a `closure color` that represents the diffuse reflectance of a
    rough surface, implementing the Oren-Nayar reflectance formula.  The
    `sigma` parameter indicates how smooth or rough the
    microstructure of the material is, with 0 being perfectly smooth and
    giving an appearance identical to `diffuse()`.

    The Oren-Nayar reflection model is described in  M. Oren and S. K.
    Nayar, "Generalization of Lambert's Reflectance Model," Proceedings of
    SIGGRAPH 1994, pp.239-246 (July, 1994).

    % Like all `closure color`s, the return "value" is symbolic and may be
    % evaluated at a later time convenient to the renderer in order to compute
    % the exitant radiance in the direction `-I`.  But aside from the
    % fact that the shader cannot examine the numeric values of the
    % `closure color`, you may program \emph{as if} {\cf oren_nayar()} was
    % implemented as follows:
    % 
    % ```
    %     normal Nf = faceforward (normalize(N), I);
    %     vector V = -normalize(I);
    %     float sigma2 = sigma * sigma;
    %     float A = 1 - 0.5 * sigma2 / (sigma2 + 0.33);
    %     float B = 0.45 * sigma2 / (sigma2 + 0.09);
    %     float  theta_r = acos (dot (V, Nf));         // Angle between V and N
    %     vector V_perp_N = normalize(V-Nf*dot(V,Nf)); // Part of V perpendicular to N
    %     color C = 0;
    %     for all lights within the hemisphere defined by (P, Nf, PI/2) {
    %         /* L is the direction of light i, Cl is its incoming radiance */
    %         vector LN = normalize(L);
    %         float cos_theta_i = dot(LN, N);
    %         float cos_phi_diff = dot (V_perp_N, normalize(LN - Nf*cos_theta_i));
    %         float theta_i = acos (cos_theta_i);
    %         float alpha = max (theta_i, theta_r);
    %         float beta = min (theta_i, theta_r);
    %         C += Cl * cos_theta_i * 
    %                 (A + B * max(0,cos_phi_diff) * sin(alpha) * tan(beta));
    %     }
    %     return C;
    % ```



`closure color` **`ward`** `(normal N, vector T, float xrough, float yrough)`

  : Returns a `closure color` that represents the anisotropic specular
    reflectance of the surface at `P`.  The `N` and `T` vectors, both
    presumed to be unit-length, are the surface normal and tangent, used to
    establish a local coordinate system for the anisotropic effects.  The
    `xrough` and `yrough` specify the amount of roughness in the
    tangent (`T`) and bitangent (`N` $\times$ `T`) directions,
    respectively.

    The Ward BRDF is described in Ward, G., "Measuring and Modeling
    Anisotropic Reflection," Proceedings of SIGGRAPH 1992.

    % Like all `closure color`s, the return "value" is symbolic and is may be
    % evaluated at a later time convenient to the renderer in order to compute
    % the exitant radiance in the direction `-I`.  But aside from the
    % fact that the shader cannot examine the numeric values of the
    % `closure color`, you may program *as if* `ward()` was
    % implemented as follows:
    % 
    % ```
    %     float sqr (float x) { return x*x; }
    % 
    %     vector V = -normalize(I);
    %     float cos_theta_r = clamp (dot(N,V), 0.0001, 1);
    %     vector X = T / xroughness;
    %     vector Y = cross(T,N) / yroughness;
    %     color C = 0;
    %     for all lights within the hemisphere defined by (P, N, PI/2) {
    %         /* L is the direction of light i, Cl is its incoming radiance */
    %         vector LN = normalize (L);
    %         float cos_theta_i = dot (LN,N);
    %         if (cos_theta_i > 0.0) {
    %             vector H = normalize (V + LN);
    %             float rho = exp (-2 * (sqr(dot(X,H)) +
    %                                    sqr(dot(Y,H))) / (1 + dot(H,N)))
    %                          / sqrt (cos_theta_i * cos_theta_r);
    %             C += Cl * cos_theta_i * rho;
    %         }
    %     }
    %     return C / (4 * xroughness * yroughness);
    % ```


`closure color` **`microfacet`** `(string distribution, normal N, float alpha, float eta, int refract)`

  : Returns a `closure color` that represents scattering on the surface
    using some microfacet distribution. A simplified isotropic version of
    the previous function.


`closure color` **`reflection`** `(normal N, float eta)`

  : Returns a `closure color` that represents sharp mirror-like reflection
    from the surface.  The reflection direction will be automatically computed
    based on the incident angle.  The `eta` parameter is the index of
    refraction of the material.  The `reflection()` closure behaves
    as if it were implemented as follows:

    ```
        vector R = reflect (I, N);
        return raytrace (R);
    ```


`closure color` **`refraction`** `(normal N, float eta)`

  : Returns a `closure color` that represents sharp glass-like refraction of
    objects "behind" the surface.  The `eta` parameter is the ratio
    of the index of refraction of the medium on the "inside" of the
    surface divided by the index of refration of the medium on the
    "outside" of the surface.  The "outside" direction is the one
    specified by `N`.  

    The refraction direction will be automatically computed based on the
    incident angle and `eta`, and the radiance returned will be
    automatically scaled by the Fresnel factor for dielectrics.
    The `refraction()` closure behaves as if it were implemented as follows:

    ```
        float Kr, Kt;
        vector R, T;
        fresnel (I, N, eta, Kr, Kt, R, T);
        return Kt * raytrace (T);
    ```


`closure color` **`transparent`** `( )`

  : Returns a `closure color` that shows the light *behind* the surface
    without any refractive bending of the light directions.
    The `transparent()` closure behaves as if it were implemented
    as follows:

    ```
        return raytrace (I);
    ```


`closure color` **`translucent`** `( )`

  : Returns a `closure color` that represents the Lambertian diffuse
    translucence of a smooth surface, which is much like `diffuse()`
    except that it gathers light from the *far* side of the surface.
    The `translucent()` closure behaves as if it were implemented
    as follows:

    ```
        return diffuse (-N);
    ```


#### Deprecated Volumetric closures

`closure color` **`isotropic`** `( )`

  : Returns a `closure color` that represents the scattering of an isotropic
    volumetric material, scattering light evenly in all directions,
    regardless of its original direction.


`closure color` **`henyey_greenstein`** `(float g)`

  : Returns a `closure color` that represents the directional volumetric
    scattering by small suspended particles.  The `g` parameter is the
    anisotropy factor, in the range $(-1, 1)$, with positive values
    indicating predominantly forward-scattering, negative values indicating
    predominantly back-scattering, and value of $g=0$ resulting in isotropic
    scattering.


`closure color` **`absorption`** `( )`

  : Returns a `closure color` that does not represent any additional
    light scattering, but rather signals to the renderer the absorption
    represents the scattering of an isotropic
    volumetric material, scattering light evenly in all directions,
    regardless of its original direction.


#### Deprecated Emission closures

`closure color` **`emission`** `( )`

  : Returns a `closure color` that represents a glowing/emissive material.
    When called in the context of a surface shader group, it implies that
    light is emitted in a full hemisphere centered around the surface
    normal. When called in the context of a volume shader group, it implies
    that light is emitted evenly in all directions around the point being
    shaded.

    The weight of the emission closure has units of radiance (e.g.,
    $\mathrm{W}\cdot\mathrm{sr}^{-1}\cdot\mathrm{m}^{-2}$). This means that a surface
    directly seen by the camera will directly reproduce the closure weight in the
    final pixel, regardless of being a surface or a volume.

    For an emissive surface, if you divide the return value of `emission()` by
    `surfacearea() * M_PI`, then you can easily specify the total emissive
    power of the light (e.g., $\mathrm{W}$), regardless of its physical size.


`closure color` **`background`** ( )

  : Returns a `closure color` that represents the radiance of the
    "background" infinitely far away in the view direction.  The
    implementation is renderer-specific, but often involves looking
    up from an HDRI environment map.



(sec-stdlib-state)=
## Renderer state and message passing

`int` **`getattribute`** `(string name, output` *`type`* `destination)` <br> `int` **`getattribute`** `(string name, int arrayindex, output` *`type`* `destination)` <br> `int` **`getattribute`** `(string object, string name, output` *`type`* `destination)` <br> `int` **`getattribute`** `(string object, string name, int arrayindex, output` *`type`* `destination)`

  : Retrieves a named renderer attribute or the value of an interpolated
    geometric variable.
    If an object is explicitly named, that is the only place that will be
    searched (`"global"` means the global scene-wide attributes).  For the
    forms of the function with no object name, or if the object name is the
    empty string `""`, the renderer will first search
    per-object attributes on the current object (or interpolated variables
    with that name attached to the object), then if not found it will search
    global scene-wide attributes.

    If the attribute is found and can be converted to the type of
    `destination`, the attribute's value will be stored in
    `destination` and `getattribute` will return `1`.  If not
    found, or the type cannot be converted, `destination` will not be
    modified and `getattribute` will return `0`.

    The automatic type conversions include those that are allowed by
    assignment in OSL source code: `int` to `float`, `float` to
    `int` (truncation), `float` (or `int`) to *triple*
    (replicating the value), any *triple* to any other *triple*.
    Additionally, the following conversions which are not allowed by
    assignment in OSL source code will also be performed by this call: 
    `float` (or `int`) to `float[2]` (replication into both
    array elements), `float[2]` to *triple* (setting the third
    component to 0).

    The forms of this function that have the the `arrayindex` parameter
    will retrieve the individual indexed element of the named array.  In this
    case, `name` must be an array attribute, the type of
    `destination` must be the type of the array element (not the type
    of the whole array), and the value of `arrayindex` must be a valid
    index given the array's size.

    Tables giving "standardized" names for different kinds of attributes may
    be found below. All renderers are expected to use the same names for these
    attributes, but are free to choose any names for additional attributes they
    wish to make queryable.

    Names of standard attributes that may be retrieved:

    Name | Type | Description
    :--- | :--- | :-----------
    `"osl:version"`           | `int`    | Major x 10000 + Minor x 100 + patch.
    `"shader:shadername"`     | `string` | Name of the shader master.
    `"shader:layername"`      | `string` | Name of the layer instance.
    `"shader:groupname"`      | `string` | Name of the shader group.

    Names of standard camera attributes that may be retrieved are in the table
    below. If the `getattribute()` function specifies an `objectname` parameter
    and it is the name of a valid camera, the value specific to that camera is
    retrieved. If no specific camera is named, the global or default camera is
    implied.

    Name | Type | Description
    :--- | :--- | :-----------
    `"camera:resolution"`     | `int[2]`    | Image resolution.
    `"camera:pixelaspect"`    | `float`     | Pixel aspect ratio.
    `"camera:projection"`     | `string`    | Projection type (e.g., `"perspective"`,   `"orthographic"`, etc.) 
    `"camera:fov"`            | `float`     | Field of fiew.
    `"camera:clip_near"`      | `float`     | Near clip distance.
    `"camera:clip_far"`       | `float`     | Far clip distance.
    `"camera:clip"`           | `float[2]`  | Near and far clip distances.
    `"camera:shutter_open"`   | `float`     | Shutter open time.
    `"camera:shutter_close"`  | `float`     | Shutter close time.
    `"camera:shutter"`        | `float[2]`  | Shutter open and close times.
    `"camera:screen_window"`  | `float[4]`  | Screen window (xmin, ymin, xmax, ymax).


`void` **`setmessage`** `(string name, output` *`type`* `value)`

  : Store a name/value pair in an area where it can later be retrieved by
    other shaders attached to the same object.  If there is already a message
    with the same name attached to this shader invocation, it will be replaced
    by the new value. The message `value` may be any basic scalar type, array,
    or closure, but may not be a `struct`.


`int` **`getmessage`** `(string name, output` *`type`* `destination)` <br> `int` **`getmessage`** `(string source, string name, output` *`type`* `destination)`

  : Retrieve a message from another shader attached to the same object. If a
    message is found with the given name, and whose type matches that of
    `destination`, the value will be stored in `destination` and
    `getmessage()` will return `1`.  If no message is found that matches both
    the name and type, `destination` will be unchanged and `getmessage()` will
    return `0`.

    The `source`, if supplied, designates from where the message should
    be retrieved, and may have any of the following values:

    `"trace"`

      : Retrieves data about the object hit by the last `trace` call made.
        Data recognized include:

        Name | Type | Description
        :--- | :--- | :-----------
        `"hit"`     | `int`   | Zero if the ray hit nothing, 1 if it hit.
        `"hitdist"` | `float` | The distance to the hit.
        `"geom:name"` | `string` | The name of the object hit.
        *other*     |         | Retrieves the named global (`P`, `N`, etc.), shader parameter, or set message of the closest object hit (only if it was a shaded ray).

    Note that which information may be retrieved depends on whether the
    ray was traced with the optional `"shade"` parameter indicating
    whether or not the shader ought to execute on the traced ray.  If
    `"shade"` was 0, you may retrieve "globals" (`P`, `N`, etc.), interpolated
    vertex variables, shader instance values, or graphics state
    attributes (object name, etc.).  But `"shade"` must be nonzero to
    correctly retrieve shader output variables or messages that are set
    by the shader (via `setmessage()`).


%First, `getmessage()` will search the source's message list as set
%by `setmessage()` when that shader ran.  If not found, then the
%source's shader parameters will be searched.  If multiple shaders of
%the same type are present, they will be searched starting with the
%last one executed, and moving backwards in execution order (in other
%words, later-executed shaders in the network effectively take precedence
%over parameteters or messages from earlier-executed shaders).


`float` **`surfacearea`** `( )`

  : Returns the surface area of the area light geometry being shaded.  This is
    meant to be used in conjunction with `emission()` in order to produce the
    correct emissive radiance given a user preference for a total wattage for
    the area light source.  The value of this function is not expected to be
    meaningful for non-light shaders.


`int` **`raytype`** `(string name)`

  : Returns 1 if ray being shaded is of the given type, or 0 if the ray is not
    of that type or if the ray type name is not recognized by the renderer.

    The set of ray type names is customizeable for renderers supporting
    OSL, but is expected to include at a minimum `"camera"`,
    `"shadow"`, `"diffuse"`, `"glossy"`, `"reflection"`,
    `"refraction"`.  They are not necessarily mutually exclusive, with the
    exception that camera rays should be of class `"camera"` and no other.


`int` **`backfacing`** `( )`

  : Returns 1 if the surface is being sampled as if "seen" from the back
    of the surface (or the "inside" of a closed object). Returns 0 if seen
    from the "front" or the "outside" of a closed object.


`int` **`isconnected`** `(`*`type`* `parameter)`

  : Returns 1 if the argument is a shader parameter and is connected
    to an earlier layer in the shader group, 
    2 if the argument is a shader output parameter connected
    to a later layer in the shader group, 3 if connected to both
    earlier and later layers, otherwise returns 0.
    Remember that function arguments in OSL are always pass-by-reference,
    so `isconnected()` applied to a function parameter will depend on
    what was passed in as the actual parameter.


`int` **`isconstant`** `(`*`type`* `expr)`

  : Returns 1 if the expression can, at runtime (knowing the values of all the
    shader group's parameter values and connections), be discerned to be
    reducible to a constant value, otherwise returns 0.

    This is primarily a debugging aid for advanced shader writers to verify
    their assumptions about what expressions can end up being constant-folded
    by the runtime optimizer.


## Dictionary Lookups

`int` **`dict_find`** `(string dictionary, string query)` <br> `int` **`dict_find`** `(int nodeID, string query)`

  : Find a node in the dictionary by a query.  The `dictionary` is
    either a string containing the actual dictionary text, or the name
    of a file containing the dictionary.  (The system can easily
    distinguish between them.)  XML dictionaries are currently
    supported, and additional formats may be supported in the future.
    The query is expressed in "XPath 1.0" syntax (or a reasonable subset
    therof).

    The return value is a *Node ID*, an opaque integer identifier
    that is the handle of a node within the dictionary data.  The value
    0 is reserved to mean "query not found" and the value -1 indicates
    that the dictionary was not a valid syntax (or, if a file, could not
    be read).  If more than one node within
    the dictionary matched the query, the node ID of the first match is
    returned, and `dict_next()` may be used to step to the next
    matching node.

    The version that takes a nodeID rather than a dictionary string
    simply interprets the query as being relative to the node
    specified by nodeID, as opposed to relative to the root of the
    dictionary.

    All expensive operations (such as reading the dictionary from a file
    and the initial parsing of the dictionary) are performed only once,
    and subsequent lookups merely copy data and are thus inexpensive.
    The `dictionary` string is, therefore, used as a hash into a
    cached data structure holding the parsed dictionary database.
    Implementations may also cache individual node lookups or type
    conversions behind the scenes.


`int` **`dict_next`** `(int nodeID)`

  : Return the node ID of the next node that matched the query that
    returned nodeID, or 0 if nodeID was the last matching node for its
    query.


`int` **`dict_value`** `(int nodeID, string attribname, output` *`type`* `value)`

  : Retrieves the named attribute of the given dictionary node, or the value
    of the node itself if `attribname` is the empty string `""`.  If the
    attribute is found, its value will be stored in `value` and 1 will be
    returned.  If the requested attribute is not found on the node, or if the
    type of `value` does not appear to match that of the named node, `value`
    will be unmodified and 0 will be returned.

    Type conversions are straightforward: anything may be retrieved as a
    string; to retrieve as an int or float, the value must parse as a
    single integer or floating point value; to retrieve as a point,
    vector, normal, color, or matrix (or any array), the value must
    parse as the correct number of values, separated by spaces and/or
    commas.


`int` **`trace`** `(point pos, vector dir, ...)`

  : Trace a ray from pos in the direction dir.  The ray is traced immediately,
    and may incur significant expense compared to rays that might be traced
    incidentally to evaluating the `Ci` closure.  Also, beware that this can
    be easily abused in such a way as to introduce view-dependence into
    shaders. The return value is 0 if the ray missed all geometry, 1 if it hit
    anything within the distance range.
    
    The following optional key-value arguments 
    (see Section [](#sec-syntax-functioncalls)) can be passed:

    `"mindist",` *`float`*

      : The minimum hit distance (default: 0).

        The units of the `"mindist"` and `"maxdist"` are determined by the
        renderer and are sometimes defined by the dir vector, which can lead
        to unexpected behavior.  So, generally, clearly written and portable
        shaders should pass a unit length (see Section [](#sec-stdlib-geom))
        dir vector.

    `"mindist",` *`float`*
      : The maximum hit distance (default: infinite).

    `"shade",` *`int`*
      : Defines whether objects hit will be shaded (default: 0).  

    `"traceset",` *`string`*

      : An optional named set of objects to ray trace (if preceded by a `-`
        character, it means to exclude that set).

        Information about the closest object hit by the ray may be retrieved
        using 
        
            getmessage("trace",...)
            
        (see Section [](#sec-stdlib-state)).

        The main purpose of this function is to allow shaders to "probe"
        nearby geometry, for example to apply a projected texture that can be
        blocked by geometry, apply more "wear" to exposed geometry, or make
        other ambient occlusion-like effects.

%  Any messages set by setmessage() prior to calling trace may be
%  retrieved by the shaders running on the object hit by the ray, via
%  `getmessage("parent",...)`.



## Miscellaneous

`int` **`arraylength`** `(`*`type`* `A[])`

  : Returns the length of the referenced array, which may be of any type.


`void` **`exit`** `()`

  : Exits the shader without further execution.  Within the main body of a
    shader, this is equivalent to calling `return`, but inside a function,
    `exit()` will exit the entire shader, whereas `return` would only exit the
    enclosing function.
