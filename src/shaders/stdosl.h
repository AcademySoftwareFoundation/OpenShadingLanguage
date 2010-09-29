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



// Declaration of built-in functions and closures
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
normal degrees (normal x) { return x*(180.0/M_PI); }
vector degrees (vector x) { return x*(180.0/M_PI); }
point  degrees (point x)  { return x*(180.0/M_PI); }
color  degrees (color x)  { return x*(180.0/M_PI); }
float  degrees (float x)  { return x*(180.0/M_PI); }
normal radians (normal x) { return x*(M_PI/180.0); }
vector radians (vector x) { return x*(M_PI/180.0); }
point  radians (point x)  { return x*(M_PI/180.0); }
color  radians (color x)  { return x*(M_PI/180.0); }
float  radians (float x)  { return x*(M_PI/180.0); }
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
point  log (point a,  float b) { return log(a)/log(b); }
vector log (vector a, float b) { return log(a)/log(b); }
color  log (color a,  float b) { return log(a)/log(b); }
float  log (float a,  float b) { return log(a)/log(b); }
PERCOMP1 (log2)
PERCOMP1 (log10)
PERCOMP1 (logb)
PERCOMP1 (sqrt)
PERCOMP1 (inversesqrt)
float hypot (float a, float b) { return sqrt (a*a + b*b); }
float hypot (float a, float b, float c) { return sqrt (a*a + b*b + c*c); }
PERCOMP1 (abs)
int abs (int x) BUILTIN;
PERCOMP1 (fabs)
int fabs (int x) BUILTIN;
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
normal clamp (normal x, normal minval, normal maxval) { return max(min(x,maxval),minval); }
vector clamp (vector x, vector minval, vector maxval) { return max(min(x,maxval),minval); }
point  clamp (point x, point minval, point maxval) { return max(min(x,maxval),minval); }
color  clamp (color x, color minval, color maxval) { return max(min(x,maxval),minval); }
float  clamp (float x, float minval, float maxval) { return max(min(x,maxval),minval); }
//normal clamp (normal x, normal minval, normal maxval) BUILTIN;
//vector clamp (vector x, vector minval, vector maxval) BUILTIN;
//point  clamp (point x, point minval, point maxval) BUILTIN;
//color  clamp (color x, color minval, color maxval) BUILTIN;
//float  clamp (float x, float minval, float maxval) BUILTIN;
normal mix (normal x, normal y, normal a) { return x*(1-a) + y*a; }
normal mix (normal x, normal y, float  a) { return x*(1-a) + y*a; }
vector mix (vector x, vector y, vector a) { return x*(1-a) + y*a; }
vector mix (vector x, vector y, float  a) { return x*(1-a) + y*a; }
point  mix (point  x, point  y, point  a) { return x*(1-a) + y*a; }
point  mix (point  x, point  y, float  a) { return x*(1-a) + y*a; }
color  mix (color  x, color  y, color  a) { return x*(1-a) + y*a; }
color  mix (color  x, color  y, float  a) { return x*(1-a) + y*a; }
float  mix (float  x, float  y, float  a) { return x*(1-a) + y*a; }
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
vector reflect (vector I, vector N) { return I - 2*dot(N,I)*N; }
vector refract (vector I, vector N, float eta) {
    float IdotN = dot (I, N);
    float k = 1 - eta*eta * (1 - IdotN*IdotN);
    return (k < 0) ? vector(0,0,0) : (eta*I - N * (eta*IdotN + sqrt(k)));
}
void fresnel (vector I, normal N, float eta,
              output float Kr, output float Kt,
              output vector R, output vector T)
{
    float sqr(float x) { return x*x; }
    float c = dot(I, N);
    if (c < 0)
        c = -c;
    R = reflect(I, N);
    float g = 1.0 / sqr(eta) - 1.0 + c * c;
    if (g >= 0.0) {
        g = sqrt (g);
        float beta = g - c;
        float F = (c * (g+c) - 1.0) / (c * beta + 1.0);
        F = 0.5 * (1.0 + sqr(F));
        F *= sqr (beta / (g+c));
        Kr = F;
        Kt = (1.0 - Kr) * eta*eta;
        // OPT: the following recomputes some of the above values, but it 
        // gives us the same result as if the shader-writer called refract()
        T = refract(I, N, eta);
    } else {
        // total internal reflection
        Kr = 1.0;
        Kt = 0.0;
        T = vector (0,0,0);
    }
#undef sqr
}

void fresnel (vector I, normal N, float eta,
              output float Kr, output float Kt)
{
    vector R, T;
    fresnel(I, N, eta, Kr, Kt, R, T);
}

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

float step (float edge, float x) { return (x>=edge) ? 1.0 : 0.0 ; }
float smoothstep (float edge0, float edge1, float x) BUILTIN;


// Derivatives and area operators


// Displacement functions


// String functions

int strlen (string s) BUILTIN;
int startswith (string s, string prefix) BUILTIN;
int endswith (string s, string suffix) BUILTIN;
string substr (string s, int start, int len) BUILTIN;
string substr (string s, int start) { return substr (s, start, strlen(s)); }

// Define concat in terms of shorter concat
string concat (string a, string b, string c) {
    return concat(concat(a,b), c);
}
string concat (string a, string b, string c, string d) {
    return concat(concat(a,b,c), d);
}
string concat (string a, string b, string c, string d, string e) {
    return concat(concat(a,b,c,d), e);
}
string concat (string a, string b, string c, string d, string e, string f) {
    return concat(concat(a,b,c,d,e), f);
}


// Texture


// Closures

closure color diffuse(normal N) BUILTIN;
closure color translucent(normal N) BUILTIN;
closure color reflection(normal N, float eta) BUILTIN;
closure color reflection(normal N) BUILTIN;
closure color refraction(normal N, float eta) BUILTIN;
closure color dielectric(normal N, float eta) BUILTIN;
closure color transparent() BUILTIN;
closure color microfacet_ggx(normal N, float ag, float eta) BUILTIN;
closure color microfacet_ggx_refraction(normal N, float ag, float eta) BUILTIN;
closure color microfacet_beckmann(normal N, float ab, float eta) BUILTIN;
closure color microfacet_beckmann_refraction(normal N, float ab, float eta) BUILTIN;
closure color ward(normal N, vector T,float ax, float ay) BUILTIN;
closure color phong(normal N, float exponent) BUILTIN;
closure color phong_ramp(normal N, float exponent, color colors[8]) BUILTIN;
closure color hair_diffuse(vector T) BUILTIN;
closure color hair_specular(vector T, float offset, float exponent) BUILTIN;
closure color ashikhmin_velvet(normal N, float sigma, float eta) BUILTIN;
closure color westin_backscatter(normal N, float roughness) BUILTIN;
closure color westin_sheen(normal N, float edginess) BUILTIN;
closure color bssrdf_cubic(color radius) BUILTIN;
closure color emission(float inner_angle, float outer_angle) BUILTIN;
closure color emission(float outer_angle) BUILTIN;
closure color emission() BUILTIN;
closure color debug(string tag) BUILTIN;
closure color background() BUILTIN;
closure color holdout() BUILTIN;
closure color subsurface(float eta, float g, float mfp, float albedo) BUILTIN;

closure color cloth(normal N, float s, float t, float dsdx, float dtdx, float dsdy, float dtdy,
                    float area_scaled, vector dPdu, color diff_warp_col, color diff_weft_col,
                    color spec_warp_col, color spec_weft_col, float fresnel_warp, float fresnel_weft,
                    float spread_x_mult, float spread_y_mult, int pattern, float pattern_angle,
                    float warp_width_scale, float weft_width_scale, float thread_count_mult_u,
                    float thread_count_mult_v) BUILTIN;
closure color cloth_specular(normal N, color spec_col[4], float eta[4], int thread_pattern[4],
                             float pattern_weight[4], int   current_thread, float brdf_interp,
                             float btf_interp, float uux, float vvx, float area_scaled, vector dPdu,
                             float eccentricity[4], float angle[4], float Kx[4], float Ky[4],
                             float Sx[4], float Sy[4]) BUILTIN;
closure color fakefur_diffuse(normal N, vector T, float fur_reflectivity, float fur_transmission,
                              float shadow_start, float shadow_end, float fur_attenuation, float fur_density,
                              float fur_avg_radius, float fur_length, float fur_shadow_fraction) BUILTIN;
closure color fakefur_specular(normal N, vector T, float offset, float exp, float fur_reflectivity,
                               float fur_transmission, float shadow_start, float shadow_end,
                               float fur_attenuation, float fur_density, float fur_avg_radius,
                               float fur_length, float fur_shadow_fraction) BUILTIN;

closure color fakefur_skin(vector N, vector T, float fur_reflectivity, float fur_transmission,
                           float shadow_start, float shadow_end, float fur_attenuation, float fur_density,
                           float fur_avg_radius, float fur_length) BUILTIN;


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
