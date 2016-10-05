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

#include "llvm_ops_math.h"
#include "llvm_ops_vec.h"
#include "llvm_ops_dual.h"
#include "llvm_ops_dual_vec.h"


#ifndef OSL_NAMESPACE_ENTER
#define OSL_NAMESPACE_ENTER
#endif
#ifndef OSL_NAMESPACE_EXIT
#define OSL_NAMESPACE_EXIT
#endif
#include "OSL/shaderglobals.h"


#ifdef OSL_COMPILING_TO_BITCODE
void * __dso_handle = 0; // necessary to avoid linkage issues in bitcode
#endif


// Handy re-casting macros
#define USTR(cstr) (*((ustring *)&cstr))
#define MAT(m) (*(Matrix44 *)m)
#define VEC(v) (*(Vec3 *)v)
#define DFLOAT(x) (*(Dual2<float> *)x)
#define DVEC(x) (*(Dual2<Vec3> *)x)
#define COL(x) (*(Color3 *)x)
#define DCOL(x) (*(Dual2<Color3> *)x)
#define TYPEDESC(x) (*(TypeDesc *)&x)


#ifndef OSL_LLVM_EXPORT
#ifdef _MSC_VER
#define OSL_LLVM_EXPORT __declspec(dllexport)
#else
#define OSL_LLVM_EXPORT __attribute__ ((visibility ("default")))
#endif
#endif

#ifndef OSL_SHADEOP
#define OSL_SHADEOP extern "C" OSL_LLVM_EXPORT
#endif


#define MAKE_UNARY_PERCOMPONENT_OP(name,floatfunc,dualfunc)         \
OSL_SHADEOP float                                                   \
ei_osl_##name##_ff (float a)                                        \
{                                                                   \
    return floatfunc(a);                                            \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
ei_osl_##name##_dfdf (void *r, void *a)                             \
{                                                                   \
    DFLOAT(r) = dualfunc (DFLOAT(a));                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
ei_osl_##name##_vv (void *r_, void *a_)                             \
{                                                                   \
    Vec3 &r (VEC(r_));                                              \
    Vec3 &a (VEC(a_));                                              \
    r[0] = floatfunc (a[0]);                                        \
    r[1] = floatfunc (a[1]);                                        \
    r[2] = floatfunc (a[2]);                                        \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
ei_osl_##name##_dvdv (void *r_, void *a_)                           \
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
OSL_SHADEOP float ei_osl_##name##_fff (float a, float b) {          \
    return floatfunc(a,b);                                          \
}                                                                   \
                                                                    \
OSL_SHADEOP void ei_osl_##name##_dfdfdf (void *r, void *a, void *b) {\
    DFLOAT(r) = dualfunc (DFLOAT(a),DFLOAT(b));                     \
}                                                                   \
                                                                    \
OSL_SHADEOP void ei_osl_##name##_dffdf (void *r, float a, void *b) {\
    DFLOAT(r) = dualfunc (Dual2<float>(a),DFLOAT(b));               \
}                                                                   \
                                                                    \
OSL_SHADEOP void ei_osl_##name##_dfdff (void *r, void *a, float b) {\
    DFLOAT(r) = dualfunc (DFLOAT(a),Dual2<float>(b));               \
}                                                                   \
                                                                    \
OSL_SHADEOP void ei_osl_##name##_vvv (void *r_, void *a_, void *b_) {\
    Vec3 &r (VEC(r_));                                              \
    Vec3 &a (VEC(a_));                                              \
    Vec3 &b (VEC(b_));                                              \
    r[0] = floatfunc (a[0], b[0]);                                  \
    r[1] = floatfunc (a[1], b[1]);                                  \
    r[2] = floatfunc (a[2], b[2]);                                  \
}                                                                   \
                                                                    \
OSL_SHADEOP void ei_osl_##name##_dvdvdv (void *r_, void *a_, void *b_)\
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
OSL_SHADEOP void ei_osl_##name##_dvvdv (void *r_, void *a_, void *b_)\
{                                                                   \
    Dual2<Vec3> a (VEC(a_));                                        \
    ei_osl_##name##_dvdvdv (r_, &a, b_);                            \
}                                                                   \
                                                                    \
OSL_SHADEOP void ei_osl_##name##_dvdvv (void *r_, void *a_, void *b_)\
{                                                                   \
    Dual2<Vec3> b (VEC(b_));                                        \
    ei_osl_##name##_dvdvdv (r_, a_, &b);                            \
}


// Mixed vec func(vec,float)
#define MAKE_BINARY_PERCOMPONENT_VF_OP(name,floatfunc,dualfunc)         \
OSL_SHADEOP void ei_osl_##name##_vvf (void *r_, void *a_, float b) {    \
    Vec3 &r (VEC(r_));                                                  \
    Vec3 &a (VEC(a_));                                                  \
    r[0] = floatfunc (a[0], b);                                         \
    r[1] = floatfunc (a[1], b);                                         \
    r[2] = floatfunc (a[2], b);                                         \
}                                                                       \
                                                                        \
OSL_SHADEOP void ei_osl_##name##_dvdvdf (void *r_, void *a_, void *b_)  \
{                                                                       \
    Dual2<Vec3> &r (DVEC(r_));                                          \
    Dual2<Vec3> &a (DVEC(a_));                                          \
    Dual2<float> &b (DFLOAT(b_));                                       \
    Dual2<float> ax, ay, az;                                            \
    ax = dualfunc (Dual2<float> (a.val().x, a.dx().x, a.dy().x), b);    \
    ay = dualfunc (Dual2<float> (a.val().y, a.dx().y, a.dy().y), b);    \
    az = dualfunc (Dual2<float> (a.val().z, a.dx().z, a.dy().z), b);    \
    /* Now swizzle back */                                              \
    r.set (Vec3( ax.val(), ay.val(), az.val()),                         \
           Vec3( ax.dx(),  ay.dx(),  az.dx() ),                         \
           Vec3( ax.dy(),  ay.dy(),  az.dy() ));                        \
}                                                                       \
                                                                        \
OSL_SHADEOP void ei_osl_##name##_dvvdf (void *r_, void *a_, void *b_)   \
{                                                                       \
    Dual2<Vec3> a (VEC(a_));                                            \
    ei_osl_##name##_dvdvdf (r_, &a, b_);                                \
}                                                                       \
                                                                        \
OSL_SHADEOP void ei_osl_##name##_dvdvf (void *r_, void *a_, float b_)   \
{                                                                       \
    Dual2<float> b (b_);                                                \
    ei_osl_##name##_dvdvdf (r_, a_, &b);                                \
}


MAKE_UNARY_PERCOMPONENT_OP (sin  , fast_sin  , fast_sin )
MAKE_UNARY_PERCOMPONENT_OP (cos  , fast_cos  , fast_cos )
MAKE_UNARY_PERCOMPONENT_OP (tan  , fast_tan  , fast_tan )
MAKE_UNARY_PERCOMPONENT_OP (asin , fast_asin , fast_asin)
MAKE_UNARY_PERCOMPONENT_OP (acos , fast_acos , fast_acos)
MAKE_UNARY_PERCOMPONENT_OP (atan , fast_atan , fast_atan)
MAKE_BINARY_PERCOMPONENT_OP(atan2, fast_atan2, fast_atan2)
MAKE_UNARY_PERCOMPONENT_OP (sinh , fast_sinh , fast_sinh)
MAKE_UNARY_PERCOMPONENT_OP (cosh , fast_cosh , fast_cosh)
MAKE_UNARY_PERCOMPONENT_OP (tanh , fast_tanh , fast_tanh)


OSL_SHADEOP void ei_osl_sincos_fff(float x, void *s_, void *c_)
{
    fast_sincos(x, (float *)s_, (float *)c_);
}

OSL_SHADEOP void ei_osl_sincos_dfdff(void *x_, void *s_, void *c_)
{
    Dual2<float> &x      = DFLOAT(x_);
    Dual2<float> &sine   = DFLOAT(s_);
    float        &cosine = *(float *)c_;

    float s_f, c_f;
    fast_sincos(x.val(), &s_f, &c_f);

    float xdx = x.dx(), xdy = x.dy(); // x might be aliased
    sine   = Dual2<float>(s_f,  c_f * xdx,  c_f * xdy);
    cosine = c_f;
}

OSL_SHADEOP void ei_osl_sincos_dffdf(void *x_, void *s_, void *c_)
{
    Dual2<float> &x      = DFLOAT(x_);
    float        &sine   = *(float *)s_;
    Dual2<float> &cosine = DFLOAT(c_);

    float s_f, c_f;
    fast_sincos(x.val(), &s_f, &c_f);
    float xdx = x.dx(), xdy = x.dy(); // x might be aliased
    sine   = s_f;
    cosine = Dual2<float>(c_f, -s_f * xdx, -s_f * xdy);
}

OSL_SHADEOP void ei_osl_sincos_dfdfdf(void *x_, void *s_, void *c_)
{
    Dual2<float> &x      = DFLOAT(x_);
    Dual2<float> &sine   = DFLOAT(s_);
    Dual2<float> &cosine = DFLOAT(c_);

    float s_f, c_f;
    fast_sincos(x.val(), &s_f, &c_f);
    float xdx = x.dx(), xdy = x.dy(); // x might be aliased
    sine   = Dual2<float>(s_f,  c_f * xdx,  c_f * xdy);
    cosine = Dual2<float>(c_f, -s_f * xdx, -s_f * xdy);
}

OSL_SHADEOP void ei_osl_sincos_vvv(void *x_, void *s_, void *c_)
{
    for (int i = 0; i < 3; i++)
        fast_sincos(VEC(x_)[i], &VEC(s_)[i], &VEC(c_)[i]);
}

OSL_SHADEOP void ei_osl_sincos_dvdvv(void *x_, void *s_, void *c_)
{
    Dual2<Vec3> &x      = DVEC(x_);
    Dual2<Vec3> &sine   = DVEC(s_);
    Vec3        &cosine = VEC(c_);

    for (int i = 0; i < 3; i++) {
        float s_f, c_f;
        fast_sincos(x.val()[i], &s_f, &c_f);
        float xdx = x.dx()[i], xdy = x.dy()[i]; // x might be aliased
        sine.val()[i] = s_f; sine.dx()[i] =  c_f * xdx; sine.dy()[i] =  c_f * xdy;
        cosine[i] = c_f;
    }
}

OSL_SHADEOP void ei_osl_sincos_dvvdv(void *x_, void *s_, void *c_)
{
    Dual2<Vec3> &x      = DVEC(x_);
    Vec3        &sine   = VEC(s_);
    Dual2<Vec3> &cosine = DVEC(c_);

    for (int i = 0; i < 3; i++) {
        float s_f, c_f;
        fast_sincos(x.val()[i], &s_f, &c_f);
        float xdx = x.dx()[i], xdy = x.dy()[i]; // x might be aliased
        sine[i] = s_f;
        cosine.val()[i] = c_f; cosine.dx()[i] = -s_f * xdx; cosine.dy()[i] = -s_f * xdy;
    }
}

OSL_SHADEOP void ei_osl_sincos_dvdvdv(void *x_, void *s_, void *c_)
{
    Dual2<Vec3> &x      = DVEC(x_);
    Dual2<Vec3> &sine   = DVEC(s_);
    Dual2<Vec3> &cosine = DVEC(c_);

    for (int i = 0; i < 3; i++) {
        float s_f, c_f;
        fast_sincos(x.val()[i], &s_f, &c_f);
        float xdx = x.dx()[i], xdy = x.dy()[i]; // x might be aliased
          sine.val()[i] = s_f;   sine.dx()[i] =  c_f * xdx;   sine.dy()[i] =  c_f * xdy;
        cosine.val()[i] = c_f; cosine.dx()[i] = -s_f * xdx; cosine.dy()[i] = -s_f * xdy;
    }
}


MAKE_UNARY_PERCOMPONENT_OP     (log        , fast_log       , fast_log)
MAKE_UNARY_PERCOMPONENT_OP     (log2       , fast_log2      , fast_log2)
MAKE_UNARY_PERCOMPONENT_OP     (log10      , fast_log10     , fast_log10)
MAKE_UNARY_PERCOMPONENT_OP     (exp        , fast_exp       , fast_exp)
MAKE_UNARY_PERCOMPONENT_OP     (exp2       , fast_exp2      , fast_exp2)
MAKE_UNARY_PERCOMPONENT_OP     (expm1      , fast_expm1     , fast_expm1)
MAKE_BINARY_PERCOMPONENT_OP    (pow        , fast_safe_pow  , fast_safe_pow)
MAKE_BINARY_PERCOMPONENT_VF_OP (pow        , fast_safe_pow  , fast_safe_pow)
MAKE_UNARY_PERCOMPONENT_OP     (erf        , fast_erf       , fast_erf)
MAKE_UNARY_PERCOMPONENT_OP     (erfc       , fast_erfc      , fast_erfc)
MAKE_UNARY_PERCOMPONENT_OP     (sqrt       , safe_sqrt      , sqrt)
MAKE_UNARY_PERCOMPONENT_OP     (inversesqrt, safe_inversesqrt, inversesqrt)


OSL_SHADEOP float ei_osl_logb_ff (float x) { return fast_logb(x); }
OSL_SHADEOP void ei_osl_logb_vv (void *r, void *x_) {
    const Vec3 &x (VEC(x_));
    VEC(r).setValue (fast_logb(x[0]), fast_logb(x[1]), fast_logb(x[2]));
}

OSL_SHADEOP float ei_osl_floor_ff (float x) { return Imath::floor(x); }
OSL_SHADEOP void ei_osl_floor_vv (void *r, void *x_) {
    const Vec3 &x (VEC(x_));
    VEC(r).setValue (Imath::floor(x[0]), Imath::floor(x[1]), Imath::floor(x[2]));
}
OSL_SHADEOP float ei_osl_ceil_ff (float x) { return Imath::ceil(x); }
OSL_SHADEOP void ei_osl_ceil_vv (void *r, void *x_) {
    const Vec3 &x (VEC(x_));
    VEC(r).setValue (Imath::ceil(x[0]), Imath::ceil(x[1]), Imath::ceil(x[2]));
}
OSL_SHADEOP float ei_osl_round_ff (float x) { return Imath::round(x); }
OSL_SHADEOP void ei_osl_round_vv (void *r, void *x_) {
    const Vec3 &x (VEC(x_));
    VEC(r).setValue (Imath::round(x[0]), Imath::round(x[1]), Imath::round(x[2]));
}
OSL_SHADEOP float ei_osl_trunc_ff (float x) { return Imath::trunc(x); }
OSL_SHADEOP void ei_osl_trunc_vv (void *r, void *x_) {
    const Vec3 &x (VEC(x_));
    VEC(r).setValue (Imath::trunc(x[0]), Imath::trunc(x[1]), Imath::trunc(x[2]));
}
OSL_SHADEOP float ei_osl_sign_ff (float x) {
    return x < 0.0f ? -1.0f : (x==0.0f ? 0.0f : 1.0f);
}
OSL_SHADEOP void ei_osl_sign_vv (void *r, void *x_) {
    const Vec3 &x (VEC(x_));
    VEC(r).setValue (ei_osl_sign_ff(x[0]), ei_osl_sign_ff(x[1]), ei_osl_sign_ff(x[2]));
}
OSL_SHADEOP float ei_osl_step_fff (float edge, float x) {
    return x < edge ? 0.0f : 1.0f;
}
OSL_SHADEOP void ei_osl_step_vvv (void *result, void *edge, void *x) {
    VEC(result).setValue (((float *)x)[0] < ((float *)edge)[0] ? 0.0f : 1.0f,
                          ((float *)x)[1] < ((float *)edge)[1] ? 0.0f : 1.0f,
                          ((float *)x)[2] < ((float *)edge)[2] ? 0.0f : 1.0f);
}


OSL_SHADEOP int ei_osl_isnan_if (float f) { return Imath::isNaN(f); }
OSL_SHADEOP int ei_osl_isinf_if (float f) { return !Imath::finitef(f); }
OSL_SHADEOP int ei_osl_isfinite_if (float f) { return Imath::finitef(f); }


OSL_SHADEOP int ei_osl_abs_ii (int x) { return Imath::abs(x); }
OSL_SHADEOP int ei_osl_fabs_ii (int x) { return Imath::abs(x); }

inline Dual2<float> fast_fabs (const Dual2<float> &x) {
    return x.val() >= 0 ? x : -x;
}

MAKE_UNARY_PERCOMPONENT_OP (abs, fast_fabs, fast_fabs);
MAKE_UNARY_PERCOMPONENT_OP (fabs, fast_fabs, fast_fabs);

OSL_SHADEOP int ei_osl_safe_mod_iii (int a, int b) {
    return (b != 0) ? (a % b) : 0;
}

inline float safe_fmod (float a, float b) {
    return (b != 0.0f) ? Imath::fmod (a,b) : 0.0f;
}

inline Dual2<float> safe_fmod (const Dual2<float> &a, const Dual2<float> &b) {
    return Dual2<float> (safe_fmod (a.val(), b.val()), a.dx(), a.dy());
}

MAKE_BINARY_PERCOMPONENT_OP (fmod, safe_fmod, safe_fmod);
MAKE_BINARY_PERCOMPONENT_VF_OP (fmod, safe_fmod, safe_fmod)

OSL_SHADEOP float ei_osl_safe_div_fff (float a, float b) {
    return (b != 0.0f) ? (a / b) : 0.0f;
}

OSL_SHADEOP int ei_osl_safe_div_iii (int a, int b) {
    return (b != 0) ? (a / b) : 0;
}

OSL_SHADEOP float ei_osl_smoothstep_ffff(float e0, float e1, float x) { return smoothstep(e0, e1, x); }

OSL_SHADEOP void ei_osl_smoothstep_dfffdf(void *result, float e0_, float e1_, void *x_)
{
   Dual2<float> e0 (e0_);
   Dual2<float> e1 (e1_);
   Dual2<float> x = DFLOAT(x_);

   DFLOAT(result) = smoothstep(e0, e1, x);
}

OSL_SHADEOP void ei_osl_smoothstep_dffdff(void *result, float e0_, void* e1_, float x_)
{
   Dual2<float> e0 (e0_);
   Dual2<float> e1 = DFLOAT(e1_);
   Dual2<float> x  (x_ );

   DFLOAT(result) = smoothstep(e0, e1, x);
}

OSL_SHADEOP void ei_osl_smoothstep_dffdfdf(void *result, float e0_, void* e1_, void* x_)
{
   Dual2<float> e0 (e0_);
   Dual2<float> e1 = DFLOAT(e1_);
   Dual2<float> x  = DFLOAT(x_ );

   DFLOAT(result) = smoothstep(e0, e1, x);
}

OSL_SHADEOP void ei_osl_smoothstep_dfdfff(void *result, void* e0_, float e1_, float x_)
{
   Dual2<float> e0 = DFLOAT(e0_);
   Dual2<float> e1 (e1_);
   Dual2<float> x  (x_ );

   DFLOAT(result) = smoothstep(e0, e1, x);
}

OSL_SHADEOP void ei_osl_smoothstep_dfdffdf(void *result, void* e0_, float e1_, void* x_)
{
   Dual2<float> e0 = DFLOAT(e0_);
   Dual2<float> e1 (e1_);
   Dual2<float> x  = DFLOAT(x_ );

   DFLOAT(result) = smoothstep(e0, e1, x);
}

OSL_SHADEOP void ei_osl_smoothstep_dfdfdff(void *result, void* e0_, void* e1_, float x_)
{
   Dual2<float> e0 = DFLOAT(e0_);
   Dual2<float> e1 = DFLOAT(e1_);
   Dual2<float> x  (x_ );

   DFLOAT(result) = smoothstep(e0, e1, x);
}

OSL_SHADEOP void ei_osl_smoothstep_dfdfdfdf(void *result, void* e0_, void* e1_, void* x_)
{
   Dual2<float> e0 = DFLOAT(e0_);
   Dual2<float> e1 = DFLOAT(e1_);
   Dual2<float> x  = DFLOAT(x_ );

   DFLOAT(result) = smoothstep(e0, e1, x);
}


// point = M * point
OSL_SHADEOP void ei_osl_transform_vmv(void *result, void* M_, void* v_)
{
   const Vec3 &v = VEC(v_);
   const Matrix44 &M = MAT(M_);
   robust_multVecMatrix (M, v, VEC(result));
}

OSL_SHADEOP void ei_osl_transform_dvmdv(void *result, void* M_, void* v_)
{
   const Dual2<Vec3> &v = DVEC(v_);
   const Matrix44    &M = MAT(M_);
   robust_multVecMatrix (M, v, DVEC(result));
}

// vector = M * vector
OSL_SHADEOP void ei_osl_transformv_vmv(void *result, void* M_, void* v_)
{
   const Vec3 &v = VEC(v_);
   const Matrix44 &M = MAT(M_);
   M.multDirMatrix (v, VEC(result));
}

OSL_SHADEOP void ei_osl_transformv_dvmdv(void *result, void* M_, void* v_)
{
   const Dual2<Vec3> &v = DVEC(v_);
   const Matrix44    &M = MAT(M_);
   multDirMatrix (M, v, DVEC(result));
}

// normal = M * normal
OSL_SHADEOP void ei_osl_transformn_vmv(void *result, void* M_, void* v_)
{
   const Vec3 &v = VEC(v_);
   const Matrix44 &M = MAT(M_);
   M.inverse().transpose().multDirMatrix (v, VEC(result));
}

OSL_SHADEOP void ei_osl_transformn_dvmdv(void *result, void* M_, void* v_)
{
   const Dual2<Vec3> &v = DVEC(v_);
   const Matrix44    &M = MAT(M_);
   multDirMatrix (M.inverse().transpose(), v, DVEC(result));
}


// Vector ops
OSL_SHADEOP float
ei_osl_dot_fvv (void *a, void *b)
{
    return VEC(a).dot (VEC(b));
}

OSL_SHADEOP void
ei_osl_dot_dfdvdv (void *result, void *a, void *b)
{
    DFLOAT(result) = dot (DVEC(a), DVEC(b));
}

OSL_SHADEOP void
ei_osl_dot_dfdvv (void *result, void *a, void *b_)
{
    Dual2<Vec3> b (VEC(b_));
    ei_osl_dot_dfdvdv (result, a, &b);
}

OSL_SHADEOP void
ei_osl_dot_dfvdv (void *result, void *a_, void *b)
{
    Dual2<Vec3> a (VEC(a_));
    ei_osl_dot_dfdvdv (result, &a, b);
}


OSL_SHADEOP void
ei_osl_cross_vvv (void *result, void *a, void *b)
{
    VEC(result) = VEC(a).cross (VEC(b));
}

OSL_SHADEOP void
ei_osl_cross_dvdvdv (void *result, void *a, void *b)
{
    DVEC(result) = cross (DVEC(a), DVEC(b));
}

OSL_SHADEOP void
ei_osl_cross_dvdvv (void *result, void *a, void *b_)
{
    Dual2<Vec3> b (VEC(b_));
    ei_osl_cross_dvdvdv (result, a, &b);
}

OSL_SHADEOP void
ei_osl_cross_dvvdv (void *result, void *a_, void *b)
{
    Dual2<Vec3> a (VEC(a_));
    ei_osl_cross_dvdvdv (result, &a, b);
}


OSL_SHADEOP float
ei_osl_length_fv (void *a)
{
    return VEC(a).length();
}

OSL_SHADEOP void
ei_osl_length_dfdv (void *result, void *a)
{
    DFLOAT(result) = length(DVEC(a));
}


OSL_SHADEOP float
ei_osl_distance_fvv (void *a_, void *b_)
{
    const Vec3 &a (VEC(a_));
    const Vec3 &b (VEC(b_));
    float x = a[0] - b[0];
    float y = a[1] - b[1];
    float z = a[2] - b[2];
    return fast_sqrt (x*x + y*y + z*z);
}

OSL_SHADEOP void
ei_osl_distance_dfdvdv (void *result, void *a, void *b)
{
    DFLOAT(result) = distance (DVEC(a), DVEC(b));
}

OSL_SHADEOP void
ei_osl_distance_dfdvv (void *result, void *a, void *b)
{
    DFLOAT(result) = distance (DVEC(a), VEC(b));
}

OSL_SHADEOP void
ei_osl_distance_dfvdv (void *result, void *a, void *b)
{
    DFLOAT(result) = distance (VEC(a), DVEC(b));
}


OSL_SHADEOP void
ei_osl_normalize_vv (void *result, void *a)
{
    VEC(result) = VEC(a).normalized();
}

OSL_SHADEOP void
ei_osl_normalize_dvdv (void *result, void *a)
{
    DVEC(result) = normalize(DVEC(a));
}


inline Vec3 calculatenormal(void *P_, bool flipHandedness)
{
    Dual2<Vec3> &tmpP (DVEC(P_));
    if (flipHandedness)
        return tmpP.dy().cross( tmpP.dx());
    else
        return tmpP.dx().cross( tmpP.dy());
}

OSL_SHADEOP void ei_osl_calculatenormal(void *out, void *sg_, void *P_)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    Vec3 N = calculatenormal(P_, sg->flipHandedness);
    // Don't normalize N
    VEC(out) = N;
}

OSL_SHADEOP float ei_osl_area(void *P_)
{
    Vec3 N = calculatenormal(P_, false);
    return N.length();
}


inline float filter_width(float dx, float dy)
{
    return fast_sqrt(dx*dx + dy*dy);
}

OSL_SHADEOP float ei_osl_filterwidth_fdf(void *x_)
{
    Dual2<float> &x = DFLOAT(x_);
    return filter_width(x.dx(), x.dy());
}

OSL_SHADEOP void ei_osl_filterwidth_vdv(void *out, void *x_)
{
    Dual2<Vec3> &x = DVEC(x_);

    VEC(out).x = filter_width (x.dx().x, x.dy().x);
    VEC(out).y = filter_width (x.dx().y, x.dy().y);
    VEC(out).z = filter_width (x.dx().z, x.dy().z);
}


// Asked if the raytype includes a bit pattern.
OSL_SHADEOP int ei_osl_raytype_bit (void *sg_, int bit)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    return (sg->raytype & bit) != 0;
}

