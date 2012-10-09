/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*

This file contains implementations of shadeops that will be used to JIT
LLVM code.  This file will be compiled by llvm-gcc, turned into LLVM IR,
which will be used to "seed" the LLVM JIT engine at runtime.  This is
*much* easier than creating LLVM IR directly (see llvm_instance.cpp for
examples), as you are just coding in C++, but there are some rules:

* Shadeop implementations MUST be named: osl_NAME_{args} where NAME is
  the traditional name of the oso shadeop, and {args} is the
  concatenation of type codes for all args including the return value --
  f/i/v/m/s for float/int/triple/matrix/string that don't have
  derivatives, and df/dv/dm for duals (values with derivatives).
  (Special case: x for 'void' return value.)

* Shadeops that return a string, int, or float without derivatives, just
  return the value directly.  Shadeops that "return" a float with
  derivatives, or an aggregate type (color, point/vector/normal, or
  matrix), will be "void" functions, and their first argument is a
  pointer to where the "return value" should go.

* Argument passing: int and float (without derivs) are passed as int and
  float.  Aggregates (color/point/vector/normal/matrix), arrays of any
  types, or floats with derivatives are passed as a void* and to their
  memory location you need to cast appropriately.  Strings are passed as
  char*, but they are always the characters of 'ustring' objects, so are
  unique.  See the handy USTR, MAT, VEC, DFLOAT, DVEC macros for
  handy/cheap casting of those void*'s to references to ustring&,
  Matrix44&, Vec3&, Dual2<float>&, and Dual2<Vec3>, respectively.

* You must provide all allowable polymorphic and derivative combinations!
  Remember that string, int, and matrix can't have a derivative, so
  there's no need to do the dm/ds/di combinations.  Furthermore, if the
  function returns an int, string, or matrix, there's no need to worry
  about derivs of the arguments, either.  (Upstream it will recognize
  that if the results can't have derivs, there's no need to pass derivs
  of arguments.)

* For the special case of simple functions that operate per-component
  and have only 1 or 2 arguments and a return value of the same type,
  note the MAKE_UNARY_PERCOMPONENT_OP and MAKE_BINARY_PERCOMPONENT_OP
  macros that will populate all the polymorphic and derivative cases
  for you.

* Shadeop implementations must have 'extern "C"' declaration, through
  the OSL_SHADEOP define, that's the only way they can be "seen" by
  LLVM, given the mangling that would occur otherwise.  (This is why
  we use the _{args} suffixes to  distinguish polymorphic and
  deiv/noderiv versions.) On Windows, this defined also ensures the
  symbols are exported for LLVM to find them.

* You may use full C++, including standard library.  You may have calls
  to any other part of the OSL library software.  You may use Boost,
  Ilmbase (Vec3, Matrix44, etc.) or any other external routines.  You
  may write templates or helper functions (which do NOT need to use
  OSL_SHADEOP, since they don't need to be runtime-discoverable by LLVM.

* If you need to access non-passed globals (P, N, etc.) or make renderer
  callbacks, just make the first argument to the function a void* that
  you cast to a ShaderGlobals* and access the globals, shading
  context (sg->context), opaque renderer state (sg->renderstate), etc.

*/


#include <string>
#include <cstdio>

#include "oslconfig.h"
#include "oslexec_pvt.h"
#include "dual.h"
using namespace OSL;
using namespace OSL::pvt;

#include <dual.h>
#include <dual_vec.h>
#include <OpenEXR/ImathFun.h>
#include <OpenImageIO/fmath.h>

using OIIO::safe_asinf;
using OIIO::safe_acosf;
using OIIO::isinf;

#ifdef _WIN32
using OIIO::roundf;
using OIIO::truncf;
using OIIO::expm1f;
using OIIO::erff;
using OIIO::erfcf;
using OIIO::log2f;
using OIIO::logbf;
using OIIO::exp2f;
#else
using OIIO::isnan;
using OIIO::isfinite;
#endif

// Handy re-casting macros
#define USTR(cstr) (*((ustring *)&cstr))
#define MAT(m) (*(Matrix44 *)m)
#define VEC(v) (*(Vec3 *)v)
#define DFLOAT(x) (*(Dual2<Float> *)x)
#define DVEC(x) (*(Dual2<Vec3> *)x)
#define COL(x) (*(Color3 *)x)
#define DCOL(x) (*(Dual2<Color3> *)x)
#define TYPEDESC(x) (*(TypeDesc *)&x)


OSL_SHADEOP void
osl_assert_nonnull (void *x, const char *msg)
{
    if (!x && msg)
        printf ("found null %s\n", msg);
    ASSERT (x && "should be non-null");
}



#define MAKE_UNARY_PERCOMPONENT_OP(name,floatfunc,dualfunc)         \
OSL_SHADEOP float                                                   \
osl_##name##_ff (float a)                                           \
{                                                                   \
    return floatfunc(a);                                            \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_dfdf (void *r, void *a)                                \
{                                                                   \
    DFLOAT(r) = dualfunc (DFLOAT(a));                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_vv (void *r_, void *a_)                                \
{                                                                   \
    Vec3 &r (VEC(r_));                                              \
    Vec3 &a (VEC(a_));                                              \
    r[0] = floatfunc (a[0]);                                        \
    r[1] = floatfunc (a[1]);                                        \
    r[2] = floatfunc (a[2]);                                        \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_dvdv (void *r_, void *a_)                              \
{                                                                   \
    Dual2<Vec3> &r (DVEC(r_));                                      \
    Dual2<Vec3> &a (DVEC(a_));                                      \
    /* Swizzle the Dual2<Vec3>'s into 3 Dual2<float>'s */           \
    Dual2<float> ax, ay, az;                                        \
    ax = dualfunc (Dual2<float> (a.val().x, a.dx().x, a.dy().x));   \
    ay = dualfunc (Dual2<float> (a.val().y, a.dx().y, a.dy().y));   \
    az = dualfunc (Dual2<float> (a.val().z, a.dx().z, a.dy().z));   \
    /* Now swizzle back */                                          \
    r.set (Vec3( ax.val(), ay.val(), az.val()),                     \
           Vec3( ax.dx(),  ay.dx(),  az.dx() ),                     \
           Vec3( ax.dy(),  ay.dy(),  az.dy() ));                    \
}


#define MAKE_BINARY_PERCOMPONENT_OP(name,floatfunc,dualfunc)        \
OSL_SHADEOP float osl_##name##_fff (float a, float b) {             \
    return floatfunc(a,b);                                          \
}                                                                   \
                                                                    \
OSL_SHADEOP void osl_##name##_dfdfdf (void *r, void *a, void *b) {  \
    DFLOAT(r) = dualfunc (DFLOAT(a),DFLOAT(b));                     \
}                                                                   \
                                                                    \
OSL_SHADEOP void osl_##name##_dffdf (void *r, float a, void *b) {   \
    DFLOAT(r) = dualfunc (Dual2<float>(a),DFLOAT(b));               \
}                                                                   \
                                                                    \
OSL_SHADEOP void osl_##name##_dfdff (void *r, void *a, float b) {   \
    DFLOAT(r) = dualfunc (DFLOAT(a),Dual2<float>(b));               \
}                                                                   \
                                                                    \
OSL_SHADEOP void osl_##name##_vvv (void *r_, void *a_, void *b_) {  \
    Vec3 &r (VEC(r_));                                              \
    Vec3 &a (VEC(a_));                                              \
    Vec3 &b (VEC(b_));                                              \
    r[0] = floatfunc (a[0], b[0]);                                  \
    r[1] = floatfunc (a[1], b[1]);                                  \
    r[2] = floatfunc (a[2], b[2]);                                  \
}                                                                   \
                                                                    \
OSL_SHADEOP void osl_##name##_dvdvdv (void *r_, void *a_, void *b_) \
{                                                                   \
    Dual2<Vec3> &r (DVEC(r_));                                      \
    Dual2<Vec3> &a (DVEC(a_));                                      \
    Dual2<Vec3> &b (DVEC(b_));                                      \
    /* Swizzle the Dual2<Vec3>'s into 3 Dual2<float>'s */           \
    Dual2<float> ax, ay, az;                                        \
    ax = dualfunc (Dual2<float> (a.val().x, a.dx().x, a.dy().x),    \
                   Dual2<float> (b.val().x, b.dx().x, b.dy().x));   \
    ay = dualfunc (Dual2<float> (a.val().y, a.dx().y, a.dy().y),    \
                   Dual2<float> (b.val().y, b.dx().y, b.dy().y));   \
    az = dualfunc (Dual2<float> (a.val().z, a.dx().z, a.dy().z),    \
                   Dual2<float> (b.val().z, b.dx().z, b.dy().z));   \
    /* Now swizzle back */                                          \
    r.set (Vec3( ax.val(), ay.val(), az.val()),                     \
           Vec3( ax.dx(),  ay.dx(),  az.dx() ),                     \
           Vec3( ax.dy(),  ay.dy(),  az.dy() ));                    \
}                                                                   \
                                                                    \
OSL_SHADEOP void osl_##name##_dvvdv (void *r_, void *a_, void *b_)  \
{                                                                   \
    Dual2<Vec3> &r (DVEC(r_));                                      \
    Dual2<Vec3> a (VEC(a_), Vec3(0,0,0), Vec3(0,0,0));              \
    Dual2<Vec3> &b (DVEC(b_));                                      \
    /* Swizzle the Dual2<Vec3>'s into 3 Dual2<float>'s */           \
    Dual2<float> ax, ay, az;                                        \
    ax = dualfunc (Dual2<float> (a.val().x, a.dx().x, a.dy().x),    \
                   Dual2<float> (b.val().x, b.dx().x, b.dy().x));   \
    ay = dualfunc (Dual2<float> (a.val().y, a.dx().y, a.dy().y),    \
                   Dual2<float> (b.val().y, b.dx().y, b.dy().y));   \
    az = dualfunc (Dual2<float> (a.val().z, a.dx().z, a.dy().z),    \
                   Dual2<float> (b.val().z, b.dx().z, b.dy().z));   \
    /* Now swizzle back */                                          \
    r.set (Vec3( ax.val(), ay.val(), az.val()),                     \
           Vec3( ax.dx(),  ay.dx(),  az.dx() ),                     \
           Vec3( ax.dy(),  ay.dy(),  az.dy() ));                    \
}                                                                   \
                                                                    \
OSL_SHADEOP void osl_##name##_dvdvv (void *r_, void *a_, void *b_)  \
{                                                                   \
    Dual2<Vec3> &r (DVEC(r_));                                      \
    Dual2<Vec3> &a (DVEC(a_));                                      \
    Dual2<Vec3> b (VEC(b_), Vec3(0,0,0), Vec3(0,0,0));              \
    /* Swizzle the Dual2<Vec3>'s into 3 Dual2<float>'s */           \
    Dual2<float> ax, ay, az;                                        \
    ax = dualfunc (Dual2<float> (a.val().x, a.dx().x, a.dy().x),    \
                   Dual2<float> (b.val().x, b.dx().x, b.dy().x));   \
    ay = dualfunc (Dual2<float> (a.val().y, a.dx().y, a.dy().y),    \
                   Dual2<float> (b.val().y, b.dx().y, b.dy().y));   \
    az = dualfunc (Dual2<float> (a.val().z, a.dx().z, a.dy().z),    \
                   Dual2<float> (b.val().z, b.dx().z, b.dy().z));   \
    /* Now swizzle back */                                          \
    r.set (Vec3( ax.val(), ay.val(), az.val()),                     \
           Vec3( ax.dx(),  ay.dx(),  az.dx() ),                     \
           Vec3( ax.dy(),  ay.dy(),  az.dy() ));                    \
}


MAKE_UNARY_PERCOMPONENT_OP (sin, sinf, sin)
MAKE_UNARY_PERCOMPONENT_OP (cos, cosf, cos)
MAKE_UNARY_PERCOMPONENT_OP (tan, tanf, tan)
MAKE_UNARY_PERCOMPONENT_OP (asin, safe_asinf, asin)
MAKE_UNARY_PERCOMPONENT_OP (acos, safe_acosf, acos)
MAKE_UNARY_PERCOMPONENT_OP (atan, std::atan, atan)
MAKE_BINARY_PERCOMPONENT_OP (atan2, std::atan2, atan2)
MAKE_UNARY_PERCOMPONENT_OP (sinh, std::sinh, sinh)
MAKE_UNARY_PERCOMPONENT_OP (cosh, std::cosh, cosh)
MAKE_UNARY_PERCOMPONENT_OP (tanh, std::tanh, tanh)

OSL_SHADEOP void osl_sincos_fff(float x, void *s_, void *c_)
{
    OIIO::sincos(x, (float *)s_, (float *)c_);
}

OSL_SHADEOP void osl_sincos_dfdff(void *x_, void *s_, void *c_)
{
    Dual2<float> &x      = DFLOAT(x_);
    Dual2<float> &sine   = DFLOAT(s_);
    float        &cosine = *(float *)c_;

    float s_f, c_f;
    OIIO::sincos(x.val(), &s_f, &c_f);
    float xdx = x.dx(), xdy = x.dy(); // x might be aliased
    sine   = Dual2<float>(s_f,  c_f * xdx,  c_f * xdy);
    cosine = c_f;
}

OSL_SHADEOP void osl_sincos_dffdf(void *x_, void *s_, void *c_)
{
    Dual2<float> &x      = DFLOAT(x_);
    float        &sine   = *(float *)s_;
    Dual2<float> &cosine = DFLOAT(c_);

    float s_f, c_f;
    OIIO::sincos(x.val(), &s_f, &c_f);
    float xdx = x.dx(), xdy = x.dy(); // x might be aliased
    sine   = s_f;
    cosine = Dual2<float>(c_f, -s_f * xdx, -s_f * xdy);
}

OSL_SHADEOP void osl_sincos_dfdfdf(void *x_, void *s_, void *c_)
{
    Dual2<float> &x      = DFLOAT(x_);
    Dual2<float> &sine   = DFLOAT(s_);
    Dual2<float> &cosine = DFLOAT(c_);

    float s_f, c_f;
    OIIO::sincos(x.val(), &s_f, &c_f);
    float xdx = x.dx(), xdy = x.dy(); // x might be aliased
    sine   = Dual2<float>(s_f,  c_f * xdx,  c_f * xdy);
    cosine = Dual2<float>(c_f, -s_f * xdx, -s_f * xdy);
}

OSL_SHADEOP void osl_sincos_vvv(void *x_, void *s_, void *c_)
{
    for (int i = 0; i < 3; i++)
        OIIO::sincos(VEC(x_)[i], &VEC(s_)[i], &VEC(c_)[i]);
}

OSL_SHADEOP void osl_sincos_dvdvv(void *x_, void *s_, void *c_)
{
    Dual2<Vec3> &x      = DVEC(x_);
    Dual2<Vec3> &sine   = DVEC(s_);
    Vec3        &cosine = VEC(c_);

    for (int i = 0; i < 3; i++) {
        float s_f, c_f;
        OIIO::sincos(x.val()[i], &s_f, &c_f);
        float xdx = x.dx()[i], xdy = x.dy()[i]; // x might be aliased
        sine.val()[i] = s_f; sine.dx()[i] =  c_f * xdx; sine.dy()[i] =  c_f * xdy;
        cosine[i] = c_f;
    }
}

OSL_SHADEOP void osl_sincos_dvvdv(void *x_, void *s_, void *c_)
{
    Dual2<Vec3> &x      = DVEC(x_);
    Vec3        &sine   = VEC(s_);
    Dual2<Vec3> &cosine = DVEC(c_);

    for (int i = 0; i < 3; i++) {
        float s_f, c_f;
        OIIO::sincos(x.val()[i], &s_f, &c_f);
        float xdx = x.dx()[i], xdy = x.dy()[i]; // x might be aliased
        sine[i] = s_f;
        cosine.val()[i] = c_f; cosine.dx()[i] = -s_f * xdx; cosine.dy()[i] = -s_f * xdy;
    }
}

OSL_SHADEOP void osl_sincos_dvdvdv(void *x_, void *s_, void *c_)
{
    Dual2<Vec3> &x      = DVEC(x_);
    Dual2<Vec3> &sine   = DVEC(s_);
    Dual2<Vec3> &cosine = DVEC(c_);

    for (int i = 0; i < 3; i++) {
        float s_f, c_f;
        OIIO::sincos(x.val()[i], &s_f, &c_f);
        float xdx = x.dx()[i], xdy = x.dy()[i]; // x might be aliased
          sine.val()[i] = s_f;   sine.dx()[i] =  c_f * xdx;   sine.dy()[i] =  c_f * xdy;
        cosine.val()[i] = c_f; cosine.dx()[i] = -s_f * xdx; cosine.dy()[i] = -s_f * xdy;
    }
}



inline float safe_log (float f) {
    if (f <= 0.0f)
        return -std::numeric_limits<float>::max();
    else
        return std::log (f);
}

inline float safe_log2(float x) {
    if (x <= 0.0f)
        return -std::numeric_limits<float>::max();
    else
        return log2f(x);
}

inline float safe_log10(float x) {
    if (x <= 0.0f)
        return -std::numeric_limits<float>::max();
    else
        return log10f(x);
}

inline float safe_logb (float f) {
    if (f == 0.0f) {
        // m_exec->error ("attempted to compute logb(%g)", f);
        return -std::numeric_limits<float>::max();
    } else {
        return logbf (f);
    }
}

inline Dual2<float> logb (const Dual2<float> &f) {
    // FIXME - punt on derivs
    return Dual2<float> (safe_logb(f.val()), 0.0, 0.0);
}


MAKE_UNARY_PERCOMPONENT_OP (log, safe_log, log)
MAKE_UNARY_PERCOMPONENT_OP (log2, safe_log2, log2)
MAKE_UNARY_PERCOMPONENT_OP (log10, safe_log10, log10)
MAKE_UNARY_PERCOMPONENT_OP (logb, safe_logb, logb)
MAKE_UNARY_PERCOMPONENT_OP (exp, std::exp, exp)
MAKE_UNARY_PERCOMPONENT_OP (exp2, exp2f, exp2)
MAKE_UNARY_PERCOMPONENT_OP (expm1, expm1f, expm1)
MAKE_BINARY_PERCOMPONENT_OP (pow, safe_pow, pow)
MAKE_UNARY_PERCOMPONENT_OP (erf, erff, erf)
MAKE_UNARY_PERCOMPONENT_OP (erfc, erfcf, erfc)

// Mixed vec pow(vec,float)
OSL_SHADEOP void osl_pow_vvf (void *r_, void *a_, float b) {
    Vec3 &r (VEC(r_));
    Vec3 &a (VEC(a_));
    r[0] = safe_pow (a[0], b);
    r[1] = safe_pow (a[1], b);
    r[2] = safe_pow (a[2], b);
}

OSL_SHADEOP void osl_pow_dvdvdf (void *r_, void *a_, void *b_)
{
    Dual2<Vec3> &r (DVEC(r_));
    Dual2<Vec3> &a (DVEC(a_));
    Dual2<float> &b (DFLOAT(b_));
    Dual2<float> ax, ay, az;
    ax = pow (Dual2<float> (a.val().x, a.dx().x, a.dy().x),
                   Dual2<float> (b.val(), b.dx(), b.dy()));
    ay = pow (Dual2<float> (a.val().y, a.dx().y, a.dy().y),
                   Dual2<float> (b.val(), b.dx(), b.dy()));
    az = pow (Dual2<float> (a.val().z, a.dx().z, a.dy().z),
                   Dual2<float> (b.val(), b.dx(), b.dy()));
    /* Now swizzle back */
    r.set (Vec3( ax.val(), ay.val(), az.val()),
           Vec3( ax.dx(),  ay.dx(),  az.dx() ),
           Vec3( ax.dy(),  ay.dy(),  az.dy() ));
}

OSL_SHADEOP void osl_pow_dvvdf (void *r_, void *a_, void *b_)
{
    Dual2<Vec3> &r (DVEC(r_));
    Vec3 &a (VEC(a_));
    Dual2<float> &b (DFLOAT(b_));
    /* Swizzle the Dual2<Vec3>'s into 3 Dual2<float>'s */
    Dual2<float> ax, ay, az;
    ax = pow (Dual2<float> (a.x),
                   Dual2<float> (b.val(), b.dx(), b.dy()));
    ay = pow (Dual2<float> (a.y),
                   Dual2<float> (b.val(), b.dx(), b.dy()));
    az = pow (Dual2<float> (a.z),
                   Dual2<float> (b.val(), b.dx(), b.dy()));
    /* Now swizzle back */
    r.set (Vec3( ax.val(), ay.val(), az.val()),
           Vec3( ax.dx(),  ay.dx(),  az.dx() ),
           Vec3( ax.dy(),  ay.dy(),  az.dy() ));
}

OSL_SHADEOP void osl_pow_dvdvf (void *r_, void *a_, float b_)
{
    Dual2<Vec3> &r (DVEC(r_));
    Dual2<Vec3> &a (DVEC(a_));
    Dual2<float> b (b_);
    /* Swizzle the Dual2<Vec3>'s into 3 Dual2<float>'s */
    Dual2<float> ax, ay, az;
    ax = pow (Dual2<float> (a.val().x, a.dx().x, a.dy().x), b);
    ay = pow (Dual2<float> (a.val().y, a.dx().y, a.dy().y), b);
    az = pow (Dual2<float> (a.val().z, a.dx().z, a.dy().z), b);
    /* Now swizzle back */
    r.set (Vec3( ax.val(), ay.val(), az.val()),
           Vec3( ax.dx(),  ay.dx(),  az.dx() ),
           Vec3( ax.dy(),  ay.dy(),  az.dy() ));
}



inline float safe_sqrt (float f) {
    if (f <= 0.0f) {
        return 0.0f;
    } else {
        return std::sqrt (f);
    }
}

inline float safe_inversesqrt (float f) {
    if (f <= 0.0f) {
        return 0.0f;
    } else {
        return 1.0f/sqrtf (f);
    }
}

MAKE_UNARY_PERCOMPONENT_OP (sqrt, safe_sqrt, sqrt)
MAKE_UNARY_PERCOMPONENT_OP (inversesqrt, safe_inversesqrt, inversesqrt)

OSL_SHADEOP float osl_floor_ff (float x) { return floorf(x); }
OSL_SHADEOP void osl_floor_vv (void *r, void *x_) {
    const Vec3 &x (VEC(x_));
    VEC(r).setValue (floorf(x[0]), floorf(x[1]), floorf(x[2]));
}
OSL_SHADEOP float osl_ceil_ff (float x) { return ceilf(x); }
OSL_SHADEOP void osl_ceil_vv (void *r, void *x_) {
    const Vec3 &x (VEC(x_));
    VEC(r).setValue (ceilf(x[0]), ceilf(x[1]), ceilf(x[2]));
}
OSL_SHADEOP float osl_round_ff (float x) { return roundf(x); }
OSL_SHADEOP void osl_round_vv (void *r, void *x_) {
    const Vec3 &x (VEC(x_));
    VEC(r).setValue (roundf(x[0]), roundf(x[1]), roundf(x[2]));
}
OSL_SHADEOP float osl_trunc_ff (float x) { return truncf(x); }
OSL_SHADEOP void osl_trunc_vv (void *r, void *x_) {
    const Vec3 &x (VEC(x_));
    VEC(r).setValue (truncf(x[0]), truncf(x[1]), truncf(x[2]));
}
OSL_SHADEOP float osl_sign_ff (float x) {
    return x < 0.0f ? -1.0f : (x==0.0f ? 0.0f : 1.0f);
}
OSL_SHADEOP void osl_sign_vv (void *r, void *x_) {
    const Vec3 &x (VEC(x_));
    VEC(r).setValue (osl_sign_ff(x[0]), osl_sign_ff(x[1]), osl_sign_ff(x[2]));
}
OSL_SHADEOP float osl_step_fff (float edge, float x) {
    return x < edge ? 0.0f : 1.0f;
}
OSL_SHADEOP void osl_step_vvv (void *result, void *edge, void *x) {
    VEC(result).setValue (((float *)x)[0] < ((float *)edge)[0] ? 0.0f : 1.0f,
                          ((float *)x)[1] < ((float *)edge)[1] ? 0.0f : 1.0f,
                          ((float *)x)[2] < ((float *)edge)[2] ? 0.0f : 1.0f);

}

OSL_SHADEOP int osl_isnan_if (float f) { return isnan (f); }
OSL_SHADEOP int osl_isinf_if (float f) { return isinf (f); }
OSL_SHADEOP int osl_isfinite_if (float f) { return isfinite (f); }


OSL_SHADEOP int osl_abs_ii (int x) { return abs(x); }
OSL_SHADEOP int osl_fabs_ii (int x) { return abs(x); }

inline Dual2<float> fabsf (const Dual2<float> &x) {
    return x.val() >= 0 ? x : -x;
}

MAKE_UNARY_PERCOMPONENT_OP (abs, fabsf, fabsf);
MAKE_UNARY_PERCOMPONENT_OP (fabs, fabsf, fabsf);


OSL_SHADEOP float osl_smoothstep_ffff(float e0, float e1, float x) { return smoothstep(e0, e1, x); }

OSL_SHADEOP void osl_smoothstep_dfffdf(void *result, float e0_, float e1_, void *x_)
{
   Dual2<float> e0 (e0_);
   Dual2<float> e1 (e1_);
   Dual2<float> x = DFLOAT(x_);

   DFLOAT(result) = smoothstep(e0, e1, x);
}

OSL_SHADEOP void osl_smoothstep_dffdff(void *result, float e0_, void* e1_, float x_)
{
   Dual2<float> e0 (e0_);
   Dual2<float> e1 = DFLOAT(e1_);
   Dual2<float> x  (x_ );

   DFLOAT(result) = smoothstep(e0, e1, x);
}

OSL_SHADEOP void osl_smoothstep_dffdfdf(void *result, float e0_, void* e1_, void* x_)
{
   Dual2<float> e0 (e0_);
   Dual2<float> e1 = DFLOAT(e1_);
   Dual2<float> x  = DFLOAT(x_ );

   DFLOAT(result) = smoothstep(e0, e1, x);
}

OSL_SHADEOP void osl_smoothstep_dfdfff(void *result, void* e0_, float e1_, float x_)
{
   Dual2<float> e0 = DFLOAT(e0_);
   Dual2<float> e1 (e1_);
   Dual2<float> x  (x_ );

   DFLOAT(result) = smoothstep(e0, e1, x);
}

OSL_SHADEOP void osl_smoothstep_dfdffdf(void *result, void* e0_, float e1_, void* x_)
{
   Dual2<float> e0 = DFLOAT(e0_);
   Dual2<float> e1 (e1_);
   Dual2<float> x  = DFLOAT(x_ );

   DFLOAT(result) = smoothstep(e0, e1, x);
}

OSL_SHADEOP void osl_smoothstep_dfdfdff(void *result, void* e0_, void* e1_, float x_)
{
   Dual2<float> e0 = DFLOAT(e0_);
   Dual2<float> e1 = DFLOAT(e1_);
   Dual2<float> x  (x_ );

   DFLOAT(result) = smoothstep(e0, e1, x);
}

OSL_SHADEOP void osl_smoothstep_dfdfdfdf(void *result, void* e0_, void* e1_, void* x_)
{
   Dual2<float> e0 = DFLOAT(e0_);
   Dual2<float> e1 = DFLOAT(e1_);
   Dual2<float> x  = DFLOAT(x_ );

   DFLOAT(result) = smoothstep(e0, e1, x);
}


// point = M * point
OSL_SHADEOP void osl_transform_vmv(void *result, void* M_, void* v_)
{
   Vec3 v = VEC(v_);
   Matrix44 M = MAT(M_);
   M.multVecMatrix (v, VEC(result));
}

OSL_SHADEOP void osl_transform_dvmdv(void *result, void* M_, void* v_)
{
   Dual2<Vec3> v = DVEC(v_);
   Matrix44    M = MAT(M_);
   multVecMatrix (M, v, DVEC(result));
}

// vector = M * vector
OSL_SHADEOP void osl_transformv_vmv(void *result, void* M_, void* v_)
{
   Vec3 v = VEC(v_);
   Matrix44 M = MAT(M_);
   M.multDirMatrix (v, VEC(result));
}

OSL_SHADEOP void osl_transformv_dvmdv(void *result, void* M_, void* v_)
{
   Dual2<Vec3> v = DVEC(v_);
   Matrix44    M = MAT(M_);
   multDirMatrix (M, v, DVEC(result));
}

// normal = M * normal
OSL_SHADEOP void osl_transformn_vmv(void *result, void* M_, void* v_)
{
   Vec3 v = VEC(v_);
   Matrix44 M = MAT(M_);
   M.inverse().transpose().multDirMatrix (v, VEC(result));
}

OSL_SHADEOP void osl_transformn_dvmdv(void *result, void* M_, void* v_)
{
   Dual2<Vec3> v = DVEC(v_);
   Matrix44    M = MAT(M_);
   multDirMatrix (M.inverse().transpose(), v, DVEC(result));
}



// Matrix ops

OSL_SHADEOP void
osl_mul_mm (void *r, void *a, void *b)
{
    MAT(r) = MAT(a) * MAT(b);
}

OSL_SHADEOP void
osl_mul_mf (void *r, void *a, float b)
{
    MAT(r) = MAT(a) * b;
}

OSL_SHADEOP void
osl_mul_m_ff (void *r, float a, float b)
{
    float f = a * b;
    MAT(r) = Matrix44 (f,0,0,0, 0,f,0,0, 0,0,f,0, 0,0,0,f);
}

OSL_SHADEOP void
osl_div_mm (void *r, void *a, void *b)
{
    MAT(r) = MAT(a) * MAT(b).inverse();
}

OSL_SHADEOP void
osl_div_mf (void *r, void *a, float b)
{
    MAT(r) = MAT(a) * (1.0f/b);
}

OSL_SHADEOP void
osl_div_fm (void *r, float a, void *b)
{
    MAT(r) = a * MAT(b).inverse();
}

OSL_SHADEOP void
osl_div_m_ff (void *r, float a, float b)
{
    float f = (b == 0) ? 0.0f : (a / b);
    MAT(r) = Matrix44 (f,0,0,0, 0,f,0,0, 0,0,f,0, 0,0,0,f);
}

bool
osl_get_matrix (ShaderGlobals *sg, Matrix44 *r, const char *from)
{
    ShadingContext *ctx = (ShadingContext *)sg->context;
    if (USTR(from) == Strings::common ||
            USTR(from) == ctx->shadingsys().commonspace_synonym()) {
        r->makeIdentity ();
        return true;
    }
    if (USTR(from) == Strings::shader) {
        ctx->renderer()->get_matrix (*r, sg->shader2common, sg->time);
        return true;
    }
    if (USTR(from) == Strings::object) {
        ctx->renderer()->get_matrix (*r, sg->object2common, sg->time);
        return true;
    }
    bool ok = ctx->renderer()->get_matrix (*r, USTR(from), sg->time);
    if (! ok) {
        r->makeIdentity();
        ShadingContext *ctx = (ShadingContext *)((ShaderGlobals *)sg)->context;
        if (ctx->shadingsys().unknown_coordsys_error())
            ctx->shadingsys().error ("Unknown transformation \"%s\"", from);
    }
    return ok;
}

bool
osl_get_inverse_matrix (ShaderGlobals *sg, Matrix44 *r, const char *to)
{
    ShadingContext *ctx = (ShadingContext *)sg->context;
    if (USTR(to) == Strings::common ||
            USTR(to) == ctx->shadingsys().commonspace_synonym()) {
        r->makeIdentity ();
        return true;
    }
    if (USTR(to) == Strings::shader) {
        ctx->renderer()->get_inverse_matrix (*r, sg->shader2common, sg->time);
        return true;
    }
    if (USTR(to) == Strings::object) {
        ctx->renderer()->get_inverse_matrix (*r, sg->object2common, sg->time);
        return true;
    }
    bool ok = ctx->renderer()->get_inverse_matrix (*r, USTR(to), sg->time);
    if (! ok) {
        r->makeIdentity ();
        ShadingContext *ctx = (ShadingContext *)((ShaderGlobals *)sg)->context;
        if (ctx->shadingsys().unknown_coordsys_error())
            ctx->shadingsys().error ("Unknown transformation \"%s\"", to);
    }
    return ok;
}

OSL_SHADEOP int
osl_prepend_matrix_from (void *sg, void *r, const char *from)
{
    Matrix44 m;
    bool ok = osl_get_matrix ((ShaderGlobals *)sg, &m, from);
    if (ok)
        MAT(r) = m * MAT(r);
    else {
        ShadingContext *ctx = (ShadingContext *)((ShaderGlobals *)sg)->context;
        if (ctx->shadingsys().unknown_coordsys_error())
            ctx->shadingsys().error ("Unknown transformation \"%s\"", from);
    }
    return ok;
}

OSL_SHADEOP int
osl_get_from_to_matrix (void *sg, void *r, const char *from, const char *to)
{
    Matrix44 Mfrom, Mto;
    bool ok = osl_get_matrix ((ShaderGlobals *)sg, &Mfrom, from);
    ok &= osl_get_inverse_matrix ((ShaderGlobals *)sg, &Mto, to);
    MAT(r) = Mfrom * Mto;
    return ok;
}



OSL_SHADEOP int
osl_transform_triple (void *sg_, void *Pin, int Pin_derivs,
                      void *Pout, int Pout_derivs,
                      void *from, void *to, int vectype)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    Matrix44 M;
    bool ok;
    Pin_derivs &= Pout_derivs;   // ignore derivs if output doesn't need it
    if (USTR(from) == Strings::common)
        ok = osl_get_inverse_matrix (sg, &M, (const char *)to);
    else if (USTR(to) == Strings::common)
        ok = osl_get_matrix (sg, &M, (const char *)from);
    else
        ok = osl_get_from_to_matrix (sg, &M, (const char *)from,
                                     (const char *)to);
    if (ok) {
        if (vectype == TypeDesc::POINT) {
            if (Pin_derivs)
                osl_transform_dvmdv(Pout, &M, Pin);
            else
                osl_transform_vmv(Pout, &M, Pin);
        } else if (vectype == TypeDesc::VECTOR) {
            if (Pin_derivs)
                osl_transformv_dvmdv(Pout, &M, Pin);
            else
                osl_transformv_vmv(Pout, &M, Pin);
        } else if (vectype == TypeDesc::NORMAL) {
            if (Pin_derivs)
                osl_transformn_dvmdv(Pout, &M, Pin);
            else
                osl_transformn_vmv(Pout, &M, Pin);
        }
        else ASSERT(0);
    } else {
        *(Vec3 *)Pout = *(Vec3 *)Pin;
        if (Pin_derivs) {
            ((Vec3 *)Pout)[1] = ((Vec3 *)Pin)[1];
            ((Vec3 *)Pout)[2] = ((Vec3 *)Pin)[2];
        }
    }
    if (Pout_derivs && !Pin_derivs) {
        ((Vec3 *)Pout)[1].setValue (0.0f, 0.0f, 0.0f);
        ((Vec3 *)Pout)[2].setValue (0.0f, 0.0f, 0.0f);
    }
    return ok;
}



OSL_SHADEOP int
osl_transform_triple_nonlinear (void *sg_, void *Pin, int Pin_derivs,
                                void *Pout, int Pout_derivs,
                                void *from, void *to,
                                int vectype)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    ShadingContext *ctx = (ShadingContext *)sg->context;
    RendererServices *rend = ctx->renderer();
    if (rend->transform_points (sg, USTR(from), USTR(to), sg->time,
                                (const Vec3 *)Pin, (Vec3 *)Pout, 1,
                                (TypeDesc::VECSEMANTICS)vectype)) {
        // Renderer had a direct way to transform the points between the
        // two spaces.
        if (Pout_derivs) {
            if (Pin_derivs) {
                rend->transform_points (sg, USTR(from), USTR(to), sg->time,
                                        (const Vec3 *)Pin+1,
                                        (Vec3 *)Pout+1, 2, TypeDesc::VECTOR);
            } else {
                ((Vec3 *)Pout)[1].setValue (0.0f, 0.0f, 0.0f);
                ((Vec3 *)Pout)[2].setValue (0.0f, 0.0f, 0.0f);
            }
        }
        return true;
    }

    // Renderer couldn't or wouldn't transform directly
    return osl_transform_triple (sg, Pin, Pin_derivs, Pout, Pout_derivs,
                                 from, to, vectype);
}



OSL_SHADEOP void
osl_transpose_mm (void *r, void *m)
{
    MAT(r) = MAT(m).transposed();
}

// Calculate the determinant of a 2x2 matrix.
template <typename F>
inline F det2x2(F a, F b, F c, F d)
{
    return a * d - b * c;
}

// calculate the determinant of a 3x3 matrix in the form:
//     | a1,  b1,  c1 |
//     | a2,  b2,  c2 |
//     | a3,  b3,  c3 |
template <typename F>
inline F det3x3(F a1, F a2, F a3, F b1, F b2, F b3, F c1, F c2, F c3)
{
    return a1 * det2x2( b2, b3, c2, c3 )
         - b1 * det2x2( a2, a3, c2, c3 )
         + c1 * det2x2( a2, a3, b2, b3 );
}

// calculate the determinant of a 4x4 matrix.
template <typename F>
inline F det4x4(const Imath::Matrix44<F> &m)
{
    // assign to individual variable names to aid selecting correct elements
    F a1 = m[0][0], b1 = m[0][1], c1 = m[0][2], d1 = m[0][3];
    F a2 = m[1][0], b2 = m[1][1], c2 = m[1][2], d2 = m[1][3];
    F a3 = m[2][0], b3 = m[2][1], c3 = m[2][2], d3 = m[2][3];
    F a4 = m[3][0], b4 = m[3][1], c4 = m[3][2], d4 = m[3][3];
    return a1 * det3x3( b2, b3, b4, c2, c3, c4, d2, d3, d4)
         - b1 * det3x3( a2, a3, a4, c2, c3, c4, d2, d3, d4)
         + c1 * det3x3( a2, a3, a4, b2, b3, b4, d2, d3, d4)
         - d1 * det3x3( a2, a3, a4, b2, b3, b4, c2, c3, c4);
}

OSL_SHADEOP float
osl_determinant_fm (void *m)
{
    return det4x4 (MAT(m));
}



// Vector ops

OSL_SHADEOP float
osl_dot_fvv (void *a, void *b)
{
    return VEC(a).dot (VEC(b));
}

OSL_SHADEOP void
osl_dot_dfdvdv (void *result, void *a, void *b)
{
    DFLOAT(result) = dot (DVEC(a), DVEC(b));
}

OSL_SHADEOP void
osl_dot_dfdvv (void *result, void *a, void *b_)
{
    Dual2<Vec3> b (VEC(b_));
    osl_dot_dfdvdv (result, a, &b);
}

OSL_SHADEOP void
osl_dot_dfvdv (void *result, void *a_, void *b)
{
    Dual2<Vec3> a (VEC(a_));
    osl_dot_dfdvdv (result, &a, b);
}


OSL_SHADEOP void
osl_cross_vvv (void *result, void *a, void *b)
{
    VEC(result) = VEC(a).cross (VEC(b));
}

OSL_SHADEOP void
osl_cross_dvdvdv (void *result, void *a, void *b)
{
    DVEC(result) = cross (DVEC(a), DVEC(b));
}

OSL_SHADEOP void
osl_cross_dvdvv (void *result, void *a, void *b_)
{
    Dual2<Vec3> b (VEC(b_));
    osl_cross_dvdvdv (result, a, &b);
}

OSL_SHADEOP void
osl_cross_dvvdv (void *result, void *a_, void *b)
{
    Dual2<Vec3> a (VEC(a_));
    osl_cross_dvdvdv (result, &a, b);
}


OSL_SHADEOP float
osl_length_fv (void *a)
{
    return VEC(a).length();
}

OSL_SHADEOP void
osl_length_dfdv (void *result, void *a)
{
    DFLOAT(result) = length(DVEC(a));
}


OSL_SHADEOP float
osl_distance_fvv (void *a_, void *b_)
{
    const Vec3 &a (VEC(a_));
    const Vec3 &b (VEC(b_));
    float x = a[0] - b[0];
    float y = a[1] - b[1];
    float z = a[2] - b[2];
    return sqrtf (x*x + y*y + z*z);
}

OSL_SHADEOP void
osl_distance_dfdvdv (void *result, void *a, void *b)
{
    DFLOAT(result) = distance (DVEC(a), DVEC(b));
}

OSL_SHADEOP void
osl_distance_dfdvv (void *result, void *a, void *b)
{
    DFLOAT(result) = distance (DVEC(a), VEC(b));
}

OSL_SHADEOP void
osl_distance_dfvdv (void *result, void *a, void *b)
{
    DFLOAT(result) = distance (VEC(a), DVEC(b));
}


OSL_SHADEOP void
osl_normalize_vv (void *result, void *a)
{
    VEC(result) = VEC(a).normalized();
}

OSL_SHADEOP void
osl_normalize_dvdv (void *result, void *a)
{
    DVEC(result) = normalize(DVEC(a));
}



OSL_SHADEOP void
osl_prepend_color_from (void *sg, void *c_, const char *from)
{
    ShadingContext *ctx (((ShaderGlobals *)sg)->context);
    Color3 &c (COL(c_));
    c = ctx->shadingsys().to_rgb (USTR(from), c[0], c[1], c[2]);
}




// String ops

// Only define 2-arg version of concat, sort it out upstream
OSL_SHADEOP const char *
osl_concat_sss (const char *s, const char *t)
{
    return ustring::format("%s%s", s, t).c_str();
}

OSL_SHADEOP int
osl_strlen_is (const char *s)
{
    return (int) USTR(s).length();
}

OSL_SHADEOP int
osl_startswith_iss (const char *s, const char *substr)
{
    return strncmp (s, substr, USTR(substr).length()) == 0;
}

OSL_SHADEOP int
osl_endswith_iss (const char *s, const char *substr)
{
    size_t len = USTR(substr).length();
    if (len > USTR(s).length())
        return 0;
    else
        return strncmp (s+USTR(s).length()-len, substr, len) == 0;
}

OSL_SHADEOP const char *
osl_substr_ssii (const char *s, int start, int length)
{
    int slen = (int) USTR(s).length();
    int b = start;
    if (b < 0)
        b += slen;
    b = Imath::clamp (b, 0, slen);
    return ustring(s, b, Imath::clamp (length, 0, slen)).c_str();
}

OSL_SHADEOP int
osl_regex_impl (void *sg_, const char *subject_, void *results, int nresults,
                const char *pattern, int fullmatch)
{
    extern int osl_regex_impl2 (OSL::ShadingContext *ctx, ustring subject,
                               int *results, int nresults, ustring pattern,
                               int fullmatch);

    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    return osl_regex_impl2 (sg->context, USTR(subject_),
                            (int *)results, nresults,
                            USTR(pattern), fullmatch);
}




/***********************************************************************
 * texture routines
 */

OSL_SHADEOP void
osl_texture_clear (void *opt)
{
    // Use "placement new" to clear the texture options
    new (opt) TextureOpt;
}


OSL_SHADEOP void
osl_texture_set_firstchannel (void *opt, int x)
{
    ((TextureOpt *)opt)->firstchannel = x;
}


OSL_SHADEOP void
osl_texture_set_swrap (void *opt, const char *x)
{
    ((TextureOpt *)opt)->swrap = TextureOpt::decode_wrapmode(USTR(x));
}

OSL_SHADEOP void
osl_texture_set_twrap (void *opt, const char *x)
{
    ((TextureOpt *)opt)->twrap = TextureOpt::decode_wrapmode(USTR(x));
}

OSL_SHADEOP void
osl_texture_set_rwrap (void *opt, const char *x)
{
    ((TextureOpt *)opt)->rwrap = TextureOpt::decode_wrapmode(USTR(x));
}

OSL_SHADEOP void
osl_texture_set_swrap_code (void *opt, int mode)
{
    ((TextureOpt *)opt)->swrap = (TextureOpt::Wrap)mode;
}

OSL_SHADEOP void
osl_texture_set_twrap_code (void *opt, int mode)
{
    ((TextureOpt *)opt)->twrap = (TextureOpt::Wrap)mode;
}

OSL_SHADEOP void
osl_texture_set_rwrap_code (void *opt, int mode)
{
    ((TextureOpt *)opt)->rwrap = (TextureOpt::Wrap)mode;
}

OSL_SHADEOP void
osl_texture_set_sblur (void *opt, float x)
{
    ((TextureOpt *)opt)->sblur = x;
}

OSL_SHADEOP void
osl_texture_set_tblur (void *opt, float x)
{
    ((TextureOpt *)opt)->tblur = x;
}

OSL_SHADEOP void
osl_texture_set_rblur (void *opt, float x)
{
    ((TextureOpt *)opt)->rblur = x;
}

OSL_SHADEOP void
osl_texture_set_swidth (void *opt, float x)
{
    ((TextureOpt *)opt)->swidth = x;
}

OSL_SHADEOP void
osl_texture_set_twidth (void *opt, float x)
{
    ((TextureOpt *)opt)->twidth = x;
}

OSL_SHADEOP void
osl_texture_set_rwidth (void *opt, float x)
{
    ((TextureOpt *)opt)->rwidth = x;
}

OSL_SHADEOP void
osl_texture_set_fill (void *opt, float x)
{
    ((TextureOpt *)opt)->fill = x;
}

OSL_SHADEOP void
osl_texture_set_time (void *opt, float x)
{
    ((TextureOpt *)opt)->time = x;
}

OSL_SHADEOP void
osl_texture_set_interp_name (void *opt, const char *modename)
{
    int mode = tex_interp_to_code (USTR(modename));
    if (mode >= 0)
        ((TextureOpt *)opt)->interpmode = (TextureOpt::InterpMode)mode;
}


OSL_SHADEOP void
osl_texture_set_interp_code (void *opt, int mode)
{
    ((TextureOpt *)opt)->interpmode = (TextureOpt::InterpMode)mode;
}


OSL_SHADEOP void
osl_texture_set_subimage (void *opt, int subimage)
{
    ((TextureOpt *)opt)->subimage = subimage;
}


OSL_SHADEOP void
osl_texture_set_subimagename (void *opt, const char *subimagename)
{
    ((TextureOpt *)opt)->subimagename = USTR(subimagename);
}



OSL_SHADEOP int
osl_texture (void *sg_, const char *name, void *opt_, float s, float t,
             float dsdx, float dtdx, float dsdy, float dtdy, int chans,
             void *result, void *dresultdx, void *dresultdy)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    RendererServices *renderer (sg->context->renderer());
    TextureOpt *opt = (TextureOpt *)opt_;
    opt->nchannels = chans;
    float dresultds[3], dresultdt[3];
    opt->dresultds = dresultdx ? dresultds : NULL;
    opt->dresultdt = dresultdy ? dresultdt : NULL;

    bool ok = renderer->texture (USTR(name), *opt, sg, s, t,
                                 dsdx, dtdx, dsdy, dtdy, (float *)result);

    // Correct our st texture space gradients into xy-space gradients
    if (dresultdx)
        for (int i = 0;  i < chans;  ++i)
            ((float *)dresultdx)[i] = dresultds[i] * dsdx + dresultdt[i] * dtdx;
    if (dresultdy)
        for (int i = 0;  i < chans;  ++i)
            ((float *)dresultdy)[i] = dresultds[i] * dsdy + dresultdt[i] * dtdy;
    return ok;
}

OSL_SHADEOP int
osl_texture_alpha (void *sg_, const char *name, void *opt_, float s, float t,
             float dsdx, float dtdx, float dsdy, float dtdy, int chans,
             void *result, void *dresultdx, void *dresultdy,
             void *alpha, void *dalphadx, void *dalphady)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    RendererServices *renderer (sg->context->renderer());
    TextureOpt *opt = (TextureOpt *)opt_;
    opt->nchannels = chans + 1;
    float local_result[4], dresultds[4], dresultdt[4];
    opt->dresultds = (dresultdx || dalphadx) ? dresultds : NULL;
    opt->dresultdt = (dresultdy || dalphady) ? dresultdt : NULL;

    bool ok = renderer->texture (USTR(name), *opt, sg, s, t,
                                 dsdx, dtdx, dsdy, dtdy, local_result);

    for (int i = 0;  i < chans;  ++i)
        ((float *)result)[i] = local_result[i];
    ((float *)alpha)[0] = local_result[chans];

    // Correct our st texture space gradients into xy-space gradients
    if (dresultdx)
        for (int i = 0;  i < chans;  ++i)
            ((float *)dresultdx)[i] = dresultds[i] * dsdx + dresultdt[i] * dtdx;
    if (dresultdy)
        for (int i = 0;  i < chans;  ++i)
            ((float *)dresultdy)[i] = dresultds[i] * dsdy + dresultdt[i] * dtdy;
    if (dalphadx)
        ((float *)dalphadx)[0] = dresultds[chans] * dsdx + dresultdt[chans] * dtdx;
    if (dalphady)
        ((float *)dalphady)[0] = dresultds[chans] * dsdy + dresultdt[chans] * dtdy;

    return ok;
}



OSL_SHADEOP int
osl_texture3d (void *sg_, const char *name, void *opt_, void *P_,
               void *dPdx_, void *dPdy_, void *dPdz_, int chans,
               void *result, void *dresultdx, void *dresultdy, void *dresultdz)
{
    const Vec3 &P (*(Vec3 *)P_);
    const Vec3 &dPdx (*(Vec3 *)dPdx_);
    const Vec3 &dPdy (*(Vec3 *)dPdy_);
    const Vec3 &dPdz (*(Vec3 *)dPdz_);
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    RendererServices *renderer (sg->context->renderer());
    TextureOpt *opt = (TextureOpt *)opt_;
    opt->nchannels = chans;
    float dresultds[3], dresultdt[3], dresultdr[3];
    opt->dresultds = dresultdx ? dresultds : NULL;
    opt->dresultdt = dresultdy ? dresultdt : NULL;
    opt->dresultdr = dresultdz ? dresultdr : NULL;

    bool ok = renderer->texture3d (USTR(name), *opt, sg, P,
                                   dPdx, dPdy, dPdz, (float *)result);

    // Correct our str texture space gradients into xyz-space gradients
    if (dresultdx)
        for (int i = 0;  i < chans;  ++i)
            ((float *)dresultdx)[i] = dresultds[i] * dPdx[0] + dresultdt[i] * dPdx[1] + dresultdr[i] * dPdx[2];
    if (dresultdy)
        for (int i = 0;  i < chans;  ++i)
            ((float *)dresultdy)[i] = dresultds[i] * dPdy[0] + dresultdt[i] * dPdy[1] + dresultdr[i] * dPdy[2];
    if (dresultdz)
        for (int i = 0;  i < chans;  ++i)
            ((float *)dresultdz)[i] = dresultds[i] * dPdz[0] + dresultdt[i] * dPdz[1] + dresultdr[i] * dPdz[2];
    return ok;
}


OSL_SHADEOP int
osl_texture3d_alpha (void *sg_, const char *name, void *opt_, void *P_,
                     void *dPdx_, void *dPdy_, void *dPdz_, int chans,
                     void *result, void *dresultdx,
                     void *dresultdy, void *dresultdz,
                     void *alpha, void *dalphadx,
                     void *dalphady, void *dalphadz)
{
    const Vec3 &P (*(Vec3 *)P_);
    const Vec3 &dPdx (*(Vec3 *)dPdx_);
    const Vec3 &dPdy (*(Vec3 *)dPdy_);
    const Vec3 &dPdz (*(Vec3 *)dPdz_);
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    RendererServices *renderer (sg->context->renderer());
    TextureOpt *opt = (TextureOpt *)opt_;
    opt->nchannels = chans + 1;
    float local_result[4], dresultds[4], dresultdt[4], dresultdr[4];
    opt->dresultds = (dresultdx || dalphadx) ? dresultds : NULL;
    opt->dresultdt = (dresultdy || dalphady) ? dresultdt : NULL;
    opt->dresultdr = (dresultdz || dalphadz) ? dresultdr : NULL;

    bool ok = renderer->texture3d (USTR(name), *opt, sg, P,
                                   dPdx, dPdy, dPdz, (float *)local_result);

    for (int i = 0;  i < chans;  ++i)
        ((float *)result)[i] = local_result[i];
    ((float *)alpha)[0] = local_result[chans];

    // Correct our str texture space gradients into xyz-space gradients
    if (dresultdx)
        for (int i = 0;  i < chans;  ++i)
            ((float *)dresultdx)[i] = dresultds[i] * dPdx[0] + dresultdt[i] * dPdx[1] + dresultdr[i] * dPdx[2];
    if (dresultdy)
        for (int i = 0;  i < chans;  ++i)
            ((float *)dresultdy)[i] = dresultds[i] * dPdy[0] + dresultdt[i] * dPdy[1] + dresultdr[i] * dPdy[2];
    if (dresultdz)
        for (int i = 0;  i < chans;  ++i)
            ((float *)dresultdz)[i] = dresultds[i] * dPdz[0] + dresultdt[i] * dPdz[1] + dresultdr[i] * dPdz[2];
    if (dalphadx)
        ((float *)dalphadx)[0] = dresultds[chans] * dPdx[0] + dresultdt[chans] * dPdx[1] + dresultdr[chans] * dPdx[2];
    if (dalphady)
        ((float *)dalphady)[0] = dresultds[chans] * dPdy[0] + dresultdt[chans] * dPdy[1] + dresultdr[chans] * dPdy[2];
    if (dalphadz)
        ((float *)dalphadz)[0] = dresultds[chans] * dPdz[0] + dresultdt[chans] * dPdz[1] + dresultdr[chans] * dPdz[2];

    return ok;
}



OSL_SHADEOP int
osl_environment (void *sg_, const char *name, void *opt_, void *R_,
                 void *dRdx_, void *dRdy_, int chans,
                 void *result, void *dresultdx, void *dresultdy,
                 void *alpha, void *dalphadx, void *dalphady)
{
    const Vec3 &R (*(Vec3 *)R_);
    const Vec3 &dRdx (*(Vec3 *)dRdx_);
    const Vec3 &dRdy (*(Vec3 *)dRdy_);
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    RendererServices *renderer (sg->context->renderer());
    TextureOpt *opt = (TextureOpt *)opt_;
    opt->nchannels = chans + (alpha ? 1 : 0);
    float dresultds[4], dresultdt[4];
    opt->dresultds = dresultdx ? dresultds : NULL;
    opt->dresultdt = dresultdy ? dresultdt : NULL;
    float local_result[4];

    bool ok = renderer->environment (USTR(name), *opt, sg, R,
                                     dRdx, dRdy, (float *)local_result);

    for (int i = 0;  i < chans;  ++i)
        ((float *)result)[i] = local_result[i];

    // For now, just zero out the result derivatives.  If somebody needs
    // derivatives of environment lookups, we'll fix it.  The reason
    // that this is a pain is that OIIO's environment call (unwisely?)
    // returns the st gradients, but we want the xy gradients, which is
    // tricky because we (this function you're reading) don't know which
    // projection is used to generate st from R.  Ugh.  Sweep under the
    // rug for a day when somebody is really asking for it.
    if (dresultdx)
        ((float *)dresultdx)[0] = 0.0f;
    if (dresultdy)
        ((float *)dresultdy)[0] = 0.0f;

    if (alpha) {
        ((float *)alpha)[0] = local_result[chans];
        // Zero out the alpha derivatives, for the same reason as above.
        if (dalphadx)
            ((float *)dalphadx)[0] = 0.0f;
        if (dalphady)
            ((float *)dalphady)[0] = 0.0f;
    }

    return ok;
}



OSL_SHADEOP int osl_get_textureinfo(void *sg_,    void *fin_, 
                                   void *dnam_,  int type, 
                                   int arraylen, int aggregate, void *data)
{
    // recreate TypeDesc
    TypeDesc typedesc;
    typedesc.basetype  = type;
    typedesc.arraylen  = arraylen;
    typedesc.aggregate = aggregate;
 
    ShaderGlobals *sg   = (ShaderGlobals *)sg_;
    RendererServices *renderer (sg->context->renderer());

    const ustring &filename  = USTR(fin_);
    const ustring &dataname  = USTR(dnam_);

    return renderer->get_texture_info (filename, 0 /*FIXME-ptex*/,
                                       dataname, typedesc, data);
}



// Noise helper functions
OSL_SHADEOP void
osl_noiseparams_clear (void *opt)
{
    // Use "placement new" to clear the noise options
    new (opt) NoiseParams;
}



OSL_SHADEOP void
osl_noiseparams_set_anisotropic (void *opt, int a)
{
    ((NoiseParams *)opt)->anisotropic = a;
}



OSL_SHADEOP void
osl_noiseparams_set_do_filter (void *opt, int a)
{
    ((NoiseParams *)opt)->do_filter = a;
}



OSL_SHADEOP void
osl_noiseparams_set_direction (void *opt, void *dir)
{
    ((NoiseParams *)opt)->direction = VEC(dir);
}



OSL_SHADEOP void
osl_noiseparams_set_bandwidth (void *opt, float b)
{
    ((NoiseParams *)opt)->bandwidth = b;
}



OSL_SHADEOP void
osl_noiseparams_set_impulses (void *opt, float i)
{
    ((NoiseParams *)opt)->impulses = i;
}



// Trace

OSL_SHADEOP void
osl_trace_clear (void *opt)
{
    new ((RendererServices::TraceOpt *)opt) RendererServices::TraceOpt;
}

OSL_SHADEOP void
osl_trace_set_mindist (void *opt, float x)
{
    ((RendererServices::TraceOpt *)opt)->mindist = x;
}

OSL_SHADEOP void
osl_trace_set_maxdist (void *opt, float x)
{
    ((RendererServices::TraceOpt *)opt)->maxdist = x;
}

OSL_SHADEOP void
osl_trace_set_shade (void *opt, int x)
{
    ((RendererServices::TraceOpt *)opt)->shade = x;
}


OSL_SHADEOP void
osl_trace_set_traceset (void *opt, const char *x)
{
    ((RendererServices::TraceOpt *)opt)->traceset = USTR(x);
}


OSL_SHADEOP int
osl_trace (void *sg_, void *opt_, void *Pos_, void *dPosdx_, void *dPosdy_,
           void *Dir_, void *dDirdx_, void *dDirdy_)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    RendererServices *renderer (sg->context->renderer());
    RendererServices::TraceOpt *opt = (RendererServices::TraceOpt *)opt_;
    static const Vec3 Zero (0.0f, 0.0f, 0.0f);
    const Vec3 *Pos = (Vec3 *)Pos_;
    const Vec3 *dPosdx = dPosdx_ ? (Vec3 *)dPosdx_ : &Zero;
    const Vec3 *dPosdy = dPosdy_ ? (Vec3 *)dPosdy_ : &Zero;
    const Vec3 *Dir = (Vec3 *)Dir_;
    const Vec3 *dDirdx = dDirdx_ ? (Vec3 *)dDirdx_ : &Zero;
    const Vec3 *dDirdy = dDirdy_ ? (Vec3 *)dDirdy_ : &Zero;
    return renderer->trace (*opt, sg, *Pos, *dPosdx, *dPosdy,
                            *Dir, *dDirdx, *dDirdy);
}



OSL_SHADEOP int osl_get_attribute(void *sg_,
                             int   dest_derivs,
                             void *obj_name_,
                             void *attr_name_,
                             int   array_lookup,
                             int   index,
                             const void *attr_type,
                             void *attr_dest)
{
    ShaderGlobals *sg   = (ShaderGlobals *)sg_;
    const ustring &obj_name  = USTR(obj_name_);
    const ustring &attr_name = USTR(attr_name_);

    return sg->context->osl_get_attribute (sg->renderstate, sg->objdata,
                                           dest_derivs, obj_name, attr_name,
                                           array_lookup, index,
                                           *(const TypeDesc *)attr_type,
                                           attr_dest);
}



inline Vec3 calculatenormal(void *P_, bool flipHandedness)
{
    Dual2<Vec3> &tmpP (DVEC(P_));
    if (flipHandedness)
        return tmpP.dy().cross( tmpP.dx());
    else
        return tmpP.dx().cross( tmpP.dy());
}

OSL_SHADEOP void osl_calculatenormal(void *out, void *sg_, void *P_)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    Vec3 N = calculatenormal(P_, sg->flipHandedness);
    // Don't normalize N
    VEC(out) = N;
}

OSL_SHADEOP float osl_area(void *P_)
{
    Vec3 N = calculatenormal(P_, false);
    return N.length();
}



inline float filter_width(float dx, float dy)
{
    return sqrtf(dx*dx + dy*dy);
}

OSL_SHADEOP float osl_filterwidth_fdf(void *x_)
{
    Dual2<float> &x = DFLOAT(x_);
    return filter_width(x.dx(), x.dy());
}

OSL_SHADEOP void osl_filterwidth_vdv(void *out, void *x_)
{
    Dual2<Vec3> &x = DVEC(x_);

    VEC(out).x = filter_width (x.dx().x, x.dy().x);   
    VEC(out).y = filter_width (x.dx().y, x.dy().y);   
    VEC(out).z = filter_width (x.dx().z, x.dy().z);   
}



OSL_SHADEOP int osl_dict_find_iis (void *sg_, int nodeID, void *query)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_; 
    return sg->context->dict_find (nodeID, USTR(query));
}


OSL_SHADEOP int osl_dict_find_iss (void *sg_, void *dictionary, void *query)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_; 
    return sg->context->dict_find (USTR(dictionary), USTR(query));
}


OSL_SHADEOP int osl_dict_next (void *sg_, int nodeID)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_; 
    return sg->context->dict_next (nodeID);
}


OSL_SHADEOP int osl_dict_value (void *sg_, int nodeID, void *attribname,
                               long long type, void *data)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_; 
    return sg->context->dict_value (nodeID, USTR(attribname), TYPEDESC(type), data);
}



// Asked if the raytype is a name we can't know until mid-shader.
OSL_SHADEOP int osl_raytype_name (void *sg_, void *name)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_; 
    int bit = sg->context->shadingsys().raytype_bit (USTR(name));
    return (sg->raytype & bit) != 0;
}

// Asked if the raytype includes a bit pattern.
OSL_SHADEOP int osl_raytype_bit (void *sg_, int bit)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_; 
    return (sg->raytype & bit) != 0;
}


OSL_SHADEOP float osl_rayroughness (void *sg_)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_; 
    return sg->rayroughness;
}



/***********************************************************************
 * Utility routines
 */

OSL_SHADEOP int
osl_bind_interpolated_param (void *sg_, const void *name, long long type,
                             int has_derivs, void *result)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    RendererServices *renderer (sg->context->renderer());

    return renderer->get_userdata (has_derivs, USTR(name), TYPEDESC(type),
                                   sg->renderstate, result);
}



OSL_SHADEOP int
osl_range_check (int indexvalue, int length,
                 void *sg, const void *sourcefile, int sourceline)
{
    if (indexvalue < 0 || indexvalue >= length) {
        ShadingContext *ctx = (ShadingContext *)((ShaderGlobals *)sg)->context;
        ctx->shadingsys().error ("Index [%d] out of range [0..%d]: %s:%d",
                                 indexvalue, length-1,
                                 USTR(sourcefile).c_str(), sourceline);
        if (indexvalue >= length)
            indexvalue = length-1;
        else
            indexvalue = 0;
    }
    return indexvalue;
}



OSL_SHADEOP void
osl_naninf_check (int ncomps, const void *vals_, int has_derivs,
                  void *sg, const void *sourcefile, int sourceline,
                  void *symbolname)
{
    const float *vals = (const float *)vals_;
    for (int i = 0, e = has_derivs ? 3*ncomps : ncomps;  i < e;  ++i)
        if (! isfinite(vals[i])) {
            ShadingContext *ctx = (ShadingContext *)((ShaderGlobals *)sg)->context;
            ctx->shadingsys().error ("Detected %g value in %s%s at %s:%d",
                                     vals[i],
                                     i>=ncomps ? "the derivatives of " : "",
                                     USTR(symbolname).c_str(),
                                     USTR(sourcefile).c_str(), sourceline);
            return;
        }
}
