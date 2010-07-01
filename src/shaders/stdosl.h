/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.  All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// * Neither the name of Sony Pictures Imageworks nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////


#ifndef STDOSL_H
#define STDOSL_H


#ifndef M_PI
#define M_PI       3.1415926535897932        /* pi */
#define M_PI_2     1.5707963267948966        /* pi/2 */
#define M_PI_4     0.7853981633974483        /* pi/4 */
#define M_2_PI     0.6366197723675813        /* 2/pi */
#define M_2_SQRTPI 1.1283791670955126        /* 2/sqrt(pi) */
#define M_E        2.7182818284590452        /* e (Euler's number) */
#define M_LN2      0.6931471805599453        /* ln(2) */
#define M_LN10     2.3025850929940457        /* ln(10) */
#define M_LOG2E    1.4426950408889634        /* log_2(e) */
#define M_LOG10E   0.4342944819032518        /* log_10(e) */
#define M_SQRT2    1.4142135623730950        /* sqrt(2) */
#define M_SQRT1_2  0.7071067811865475        /* 1/sqrt(2) */
#endif



// Declaration of built-in functions
#define BUILTIN [[ int builtin = 1 ]]
#define BUILTIN_DERIV [[ int builtin = 1, int deriv = 1 ]]

#define PERCOMP1(name)                          \
    normal name (normal x) BUILTIN;             \
    vector name (vector x) BUILTIN;             \
    point  name (point x) BUILTIN;              \
    color  name (color x) BUILTIN;              \
    float  name (float x) BUILTIN;

#define PERCOMP2(name)                          \
    normal name (normal x, normal y) BUILTIN;   \
    vector name (vector x, vector y) BUILTIN;   \
    point  name (point x, point y) BUILTIN;     \
    color  name (color x, color y) BUILTIN;     \
    float  name (float x, float y) BUILTIN;

#define PERCOMP2F(name)                         \
    normal name (normal x, float y) BUILTIN;    \
    vector name (vector x, float y) BUILTIN;    \
    point  name (point x, float y) BUILTIN;     \
    color  name (color x, float y) BUILTIN;     \
    float  name (float x, float y) BUILTIN;


// Basic math
PERCOMP1 (degrees)
PERCOMP1 (radians)
PERCOMP1 (cos)
PERCOMP1 (sin)
PERCOMP1 (tan)
PERCOMP1 (acos)
PERCOMP1 (asin)
PERCOMP1 (atan)
PERCOMP2 (atan2)
PERCOMP1 (cosh)
PERCOMP1 (sinh)
PERCOMP1 (tanh)
PERCOMP2F (pow)
PERCOMP1 (exp)
PERCOMP1 (exp2)
PERCOMP1 (expm1)
PERCOMP1 (log)
PERCOMP2F (log)
PERCOMP1 (log2)
PERCOMP1 (log10)
PERCOMP1 (logb)
PERCOMP1 (sqrt)
PERCOMP1 (inversesqrt)
float hypot (float x, float y) BUILTIN;
float hypot (float x, float y, float z) BUILTIN;
PERCOMP1 (abs)
int abs (int x) BUILTIN;
PERCOMP1 (fabs)
PERCOMP1 (sign)
PERCOMP1 (floor)
PERCOMP1 (ceil)
PERCOMP1 (round)
PERCOMP1 (trunc)
PERCOMP2 (fmod)
PERCOMP2F (fmod)
PERCOMP2 (mod)
PERCOMP2F (mod)
int    mod (int x, int y) BUILTIN;
PERCOMP2 (min)
PERCOMP2 (max)
normal clamp (normal x, normal minval, normal maxval) BUILTIN;
vector clamp (vector x, vector minval, vector maxval) BUILTIN;
point  clamp (point x, point minval, point maxval) BUILTIN;
color  clamp (color x, color minval, color maxval) BUILTIN;
float  clamp (float x, float minval, float maxval) BUILTIN;
normal mix (normal x, normal y, normal a) BUILTIN;
vector mix (vector x, vector y, vector a) BUILTIN;
point  mix (point x, point y, point a) BUILTIN;
color  mix (color x, color y, color a) BUILTIN;
normal mix (normal x, normal y, float a) BUILTIN;
vector mix (vector x, vector y, float a) BUILTIN;
point  mix (point x, point y, float a) BUILTIN;
color  mix (color x, color y, float a) BUILTIN;
float  mix (float x, float y, float a) BUILTIN;
int isnan (float x) BUILTIN;
int isinf (float x) BUILTIN;
int isfinite (float x) BUILTIN;
float erf (float x) BUILTIN;
float erfc (float x) BUILTIN;

// Vector functions

vector cross (vector a, vector b) BUILTIN;
float dot (vector a, vector b) BUILTIN;
float length (vector v) BUILTIN;
float distance (point a, point b) BUILTIN;
float distance (point a, point b, point q) BUILTIN;
normal normalize (normal v) BUILTIN;
vector normalize (vector v) BUILTIN;
vector faceforward (vector N, vector I, vector Nref) BUILTIN;
vector faceforward (vector N, vector I) BUILTIN;
vector reflect (vector I, vector N) BUILTIN;
vector refract (vector I, vector N, float eta) BUILTIN;
point rotate (point q, float angle, point a, point b) BUILTIN;

normal transform (matrix Mto, normal p) BUILTIN;
vector transform (matrix Mto, vector p) BUILTIN;
point transform (matrix Mto, point p) BUILTIN;

// Implementation of transform-with-named-space in terms of matrices:

point transform (string tospace, point x)
{
    return transform (matrix ("common", tospace), x);
}

point transform (string fromspace, string tospace, point x)
{
    return transform (matrix (fromspace, tospace), x);
}


vector transform (string tospace, vector x)
{
    return transform (matrix ("common", tospace), x);
}

vector transform (string fromspace, string tospace, vector x)
{
    return transform (matrix (fromspace, tospace), x);
}


normal transform (string tospace, normal x)
{
    return transform (matrix ("common", tospace), x);
}

normal transform (string fromspace, string tospace, normal x)
{
    return transform (matrix (fromspace, tospace), x);
}

float transformu (string tounits, float x) BUILTIN;
float transformu (string fromunits, string tounits, float x) BUILTIN;



// Color functions

float luminance (color c) {
    return dot ((vector)c, vector(0.2126, 0.7152, 0.0722));
}

color transformc (string to, color x) BUILTIN;
color transformc (string from, string to, color x) BUILTIN;


// Matrix functions

float determinant (matrix m) BUILTIN;
matrix transpose (matrix m) BUILTIN;



// Pattern generation

float step (float edge, float x) BUILTIN;
float smoothstep (float edge0, float edge1, float x) BUILTIN;


// Derivatives and area operators


// Displacement functions


// String functions

int strlen (string s) BUILTIN;
int startswith (string s, string prefix) BUILTIN;
int endswith (string s, string suffix) BUILTIN;
string substr (string s, int start, int len) BUILTIN;
string substr (string s, int start) { return substr (s, start, strlen(s)); }


// Texture


// Closures

closure color cloth(normal N, float s, float t, color diff_warp, color diff_weft, 
                    color spec_warp, color spec_weft, float fresnel_warp, float fresnel_weft,
                    float spread_x_mult, float spread_y_mult, int pattern, float pattern_angle,
                    float warp_width_scale, float weft_width_scale, float thread_count_mult_u,
                    float thread_count_mult_v)
{

    return cloth(N, s, t, Dx(s), Dx(t), Dy(s), Dy(t), area(P), dPdu, diff_warp, diff_weft, spec_warp, spec_weft,
                 fresnel_warp, fresnel_weft, spread_x_mult, spread_y_mult, pattern, pattern_angle, 
                 warp_width_scale, weft_width_scale, thread_count_mult_u, thread_count_mult_v);
}

closure color cloth(normal N, float s, float t, color diff_warp, color diff_weft, 
                    color spec_warp, color spec_weft, float fresnel_warp, float fresnel_weft,
                    float spread_x_mult, float spread_y_mult, int pattern, float pattern_angle,
                    float warp_width_scale, float weft_width_scale, float thread_count_mult_u,
                    float thread_count_mult_v, string tok, string val)
{

    return cloth(N, s, t, Dx(s), Dx(t), Dy(s), Dy(t), area(P), dPdu, diff_warp, diff_weft, spec_warp, spec_weft,
                 fresnel_warp, fresnel_weft, spread_x_mult, spread_y_mult, pattern, pattern_angle, 
                 warp_width_scale, weft_width_scale, thread_count_mult_u, thread_count_mult_v, tok, val);
}



// Renderer state


// Miscellaneous




#undef BUILTIN
#undef BUILTIN_DERIV
#undef PERCOMP1
#undef PERCOMP2
#undef PERCOMP2F

#endif /* STDOSL_H */
