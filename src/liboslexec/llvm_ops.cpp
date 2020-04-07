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


// Some gcc versions on some platforms seem to have max_align_t missing from
// their <cstddef>. Putting this here appears to make it build cleanly on
// those platforms while not hurting anything elsewhere.
namespace {
typedef long double max_align_t;
}

#include <iostream>
#include <cstddef>

#include <OSL/oslconfig.h>
#include <OSL/shaderglobals.h>
#include <OSL/dual.h>
#include <OSL/dual_vec.h>
using namespace OSL;

#include <OpenEXR/ImathFun.h>
#include <OpenImageIO/fmath.h>
#include <OpenImageIO/simd.h>

#if defined(_MSC_VER) && _MSC_VER < 1700
using OIIO::isinf;
#endif

#if defined(_MSC_VER) && _MSC_VER < 1800
using OIIO::roundf;
using OIIO::truncf;
using OIIO::erff;
using OIIO::erfcf;
#endif

#if defined(__FreeBSD__)
#include <sys/param.h>
#if __FreeBSD_version < 803000
// freebsd before 8.3 doesn't have log2f - use OIIO lib replacement
using OIIO::log2f;
#endif
#endif


#ifdef OSL_COMPILING_TO_BITCODE
void * __dso_handle = 0; // necessary to avoid linkage issues in bitcode
#endif


// Handy re-casting macros
#define USTR(cstr) (*((ustring *)&cstr))
#define MAT(m) (*(Matrix44 *)m)
#define VEC(v) (*(Vec3 *)v)
#define DFLOAT(x) (*(Dual2<Float> *)x)
#define DVEC(x) (*(Dual2<Vec3> *)x)
#define COL(x) (*(Color3 *)x)
#define DCOL(x) (*(Dual2<Color3> *)x)

#ifndef OSL_SHADEOP
#  ifdef __CUDACC__
#    define OSL_SHADEOP extern "C" __device__ OSL_LLVM_EXPORT __attribute__((always_inline))
#  elif defined(OSL_COMPILING_TO_BITCODE)
#    define OSL_SHADEOP extern "C" OSL_LLVM_EXPORT __attribute__((always_inline))
#  else
#    define OSL_SHADEOP extern "C" OSL_LLVM_EXPORT
#  endif
#endif

#ifndef OSL_SHADEOP_NOINLINE
#  define OSL_SHADEOP_NOINLINE extern "C" OSL_DEVICE OSL_LLVM_EXPORT
#endif



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
    r.x = floatfunc (a.x);                                          \
    r.y = floatfunc (a.y);                                          \
    r.z = floatfunc (a.z);                                          \
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
    r.x = floatfunc (a.x, b.x);                                     \
    r.y = floatfunc (a.y, b.y);                                     \
    r.z = floatfunc (a.z, b.z);                                     \
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
    Dual2<Vec3> a (VEC(a_));                                        \
    osl_##name##_dvdvdv (r_, &a, b_);                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void osl_##name##_dvdvv (void *r_, void *a_, void *b_)  \
{                                                                   \
    Dual2<Vec3> b (VEC(b_));                                        \
    osl_##name##_dvdvdv (r_, a_, &b);                               \
}


// Mixed vec func(vec,float)
#define MAKE_BINARY_PERCOMPONENT_VF_OP(name,floatfunc,dualfunc)         \
OSL_SHADEOP void osl_##name##_vvf (void *r_, void *a_, float b) {       \
    Vec3 &r (VEC(r_));                                                  \
    Vec3 &a (VEC(a_));                                                  \
    r.x = floatfunc (a.x, b);                                           \
    r.y = floatfunc (a.y, b);                                           \
    r.z = floatfunc (a.z, b);                                           \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_##name##_dvdvdf (void *r_, void *a_, void *b_)     \
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
OSL_SHADEOP void osl_##name##_dvvdf (void *r_, void *a_, void *b_)      \
{                                                                       \
    Dual2<Vec3> a (VEC(a_));                                            \
    osl_##name##_dvdvdf (r_, &a, b_);                                   \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_##name##_dvdvf (void *r_, void *a_, float b_)      \
{                                                                       \
    Dual2<float> b (b_);                                                \
    osl_##name##_dvdvdf (r_, a_, &b);                                   \
}


#if OSL_FAST_MATH
MAKE_UNARY_PERCOMPONENT_OP (sin  , OIIO::fast_sin  , fast_sin )
MAKE_UNARY_PERCOMPONENT_OP (cos  , OIIO::fast_cos  , fast_cos )
MAKE_UNARY_PERCOMPONENT_OP (tan  , OIIO::fast_tan  , fast_tan )
MAKE_UNARY_PERCOMPONENT_OP (asin , OIIO::fast_asin , fast_asin)
MAKE_UNARY_PERCOMPONENT_OP (acos , OIIO::fast_acos , fast_acos)
MAKE_UNARY_PERCOMPONENT_OP (atan , OIIO::fast_atan , fast_atan)
MAKE_BINARY_PERCOMPONENT_OP(atan2, OIIO::fast_atan2, fast_atan2)
MAKE_UNARY_PERCOMPONENT_OP (sinh , OIIO::fast_sinh , fast_sinh)
MAKE_UNARY_PERCOMPONENT_OP (cosh , OIIO::fast_cosh , fast_cosh)
MAKE_UNARY_PERCOMPONENT_OP (tanh , OIIO::fast_tanh , fast_tanh)
#else
MAKE_UNARY_PERCOMPONENT_OP (sin  , sinf      , sin  )
MAKE_UNARY_PERCOMPONENT_OP (cos  , cosf      , cos  )
MAKE_UNARY_PERCOMPONENT_OP (tan  , tanf      , tan  )
MAKE_UNARY_PERCOMPONENT_OP (asin , safe_asin , safe_asin )
MAKE_UNARY_PERCOMPONENT_OP (acos , safe_acos , safe_acos )
MAKE_UNARY_PERCOMPONENT_OP (atan , atanf     , atan )
MAKE_BINARY_PERCOMPONENT_OP(atan2, atan2f    , atan2)
MAKE_UNARY_PERCOMPONENT_OP (sinh , sinhf     , sinh )
MAKE_UNARY_PERCOMPONENT_OP (cosh , coshf     , cosh )
MAKE_UNARY_PERCOMPONENT_OP (tanh , tanhf     , tanh )
#endif

OSL_SHADEOP void osl_sincos_fff(float x, void *s_, void *c_)
{
#if OSL_FAST_MATH
    OIIO::fast_sincos(x, (float *)s_, (float *)c_);
#else
    OIIO::sincos(x, (float *)s_, (float *)c_);
#endif
}

OSL_SHADEOP void osl_sincos_dfdff(void *x_, void *s_, void *c_)
{
    Dual2<float> &x      = DFLOAT(x_);
    Dual2<float> &sine   = DFLOAT(s_);
    float        &cosine = *(float *)c_;

    float s_f, c_f;
#if OSL_FAST_MATH
    OIIO::fast_sincos(x.val(), &s_f, &c_f);
#else
    OIIO::sincos(x.val(), &s_f, &c_f);
#endif

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
#if OSL_FAST_MATH
    OIIO::fast_sincos(x.val(), &s_f, &c_f);
#else
    OIIO::sincos(x.val(), &s_f, &c_f);
#endif
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
#if OSL_FAST_MATH
    OIIO::fast_sincos(x.val(), &s_f, &c_f);
#else
    OIIO::sincos(x.val(), &s_f, &c_f);
#endif
    float xdx = x.dx(), xdy = x.dy(); // x might be aliased
    sine   = Dual2<float>(s_f,  c_f * xdx,  c_f * xdy);
    cosine = Dual2<float>(c_f, -s_f * xdx, -s_f * xdy);
}

OSL_SHADEOP void osl_sincos_vvv(void *x_, void *s_, void *c_)
{
#if OSL_FAST_MATH
        OIIO::fast_sincos(VEC(x_).x, &VEC(s_).x, &VEC(c_).x);
        OIIO::fast_sincos(VEC(x_).y, &VEC(s_).y, &VEC(c_).y);
        OIIO::fast_sincos(VEC(x_).z, &VEC(s_).z, &VEC(c_).z);
#else
        OIIO::sincos(VEC(x_).x, &VEC(s_).x, &VEC(c_).x);
        OIIO::sincos(VEC(x_).y, &VEC(s_).y, &VEC(c_).y);
        OIIO::sincos(VEC(x_).z, &VEC(s_).z, &VEC(c_).z);
#endif
}

OSL_SHADEOP void osl_sincos_dvdvv(void *x_, void *s_, void *c_)
{
    Dual2<Vec3> &x      = DVEC(x_);
    Dual2<Vec3> &sine   = DVEC(s_);
    Vec3        &cosine = VEC(c_);

#if 0 // older version using [i] deprecated to avoid potential aliasing issues
    for (int i = 0; i < 3; i++) {
        float s_f, c_f;
#if OSL_FAST_MATH
        OIIO::fast_sincos(x.val()[i], &s_f, &c_f);
#else
        OIIO::sincos(x.val()[i], &s_f, &c_f);
#endif
        float xdx = x.dx()[i], xdy = x.dy()[i]; // x might be aliased
        sine.val()[i] = s_f; sine.dx()[i] =  c_f * xdx; sine.dy()[i] =  c_f * xdy;
        cosine[i] = c_f;
    }
#else
    auto sincos_comp = [](const Dual2<float> & x, float &sine_val, float &sine_dx, float &sine_dy, float &cosine_val) {
        float s_f, c_f;
#if OSL_FAST_MATH
        OIIO::fast_sincos(x.val(), &s_f, &c_f);
#else
        OIIO::sincos(x.val(), &s_f, &c_f);
#endif
        sine_val = s_f; sine_dx =  c_f * x.dx(); sine_dy =  c_f * x.dy();
        cosine_val = c_f;
    };

    sincos_comp(comp_x(x), sine.val().x, sine.dx().x, sine.dy().x, cosine.x);
    sincos_comp(comp_y(x), sine.val().y, sine.dx().y, sine.dy().y, cosine.y);
    sincos_comp(comp_z(x), sine.val().z, sine.dx().z, sine.dy().z, cosine.z);
#endif
}

OSL_SHADEOP void osl_sincos_dvvdv(void *x_, void *s_, void *c_)
{
    Dual2<Vec3> &x      = DVEC(x_);
    Vec3        &sine   = VEC(s_);
    Dual2<Vec3> &cosine = DVEC(c_);

#if 0  // older version using [i] deprecated to avoid potential aliasing issues
    for (int i = 0; i < 3; i++) {
        float s_f, c_f;
#if OSL_FAST_MATH
        OIIO::fast_sincos(x.val()[i], &s_f, &c_f);
#else
        OIIO::sincos(x.val()[i], &s_f, &c_f);
#endif
        float xdx = x.dx()[i], xdy = x.dy()[i]; // x might be aliased
        sine[i] = s_f;
        cosine.val()[i] = c_f; cosine.dx()[i] = -s_f * xdx; cosine.dy()[i] = -s_f * xdy;
    }
#else
    auto sincos_comp = [](const Dual2<float> & x, float &sine_val, float &cosine_val, float &cosine_dx, float &cosine_dy) {
        float s_f, c_f;
#if OSL_FAST_MATH
        OIIO::fast_sincos(x.val(), &s_f, &c_f);
#else
        OIIO::sincos(x.val(), &s_f, &c_f);
#endif
        sine_val = s_f;
        cosine_val = c_f; cosine_dx = -s_f * x.dx(); cosine_dy = -s_f * x.dy();
    };

    sincos_comp(comp_x(x), sine.x, cosine.val().x, cosine.dx().x, cosine.dy().x);
    sincos_comp(comp_y(x), sine.y, cosine.val().y, cosine.dx().y, cosine.dy().y);
    sincos_comp(comp_z(x), sine.z, cosine.val().z, cosine.dx().z, cosine.dy().z);

#endif
}

OSL_SHADEOP void osl_sincos_dvdvdv(void *x_, void *s_, void *c_)
{
    Dual2<Vec3> &x      = DVEC(x_);
    Dual2<Vec3> &sine   = DVEC(s_);
    Dual2<Vec3> &cosine = DVEC(c_);

#if 0  // older version using [i] deprecated to avoid potential aliasing issues
    for (int i = 0; i < 3; i++) {
        float s_f, c_f;
#if OSL_FAST_MATH
        OIIO::fast_sincos(x.val()[i], &s_f, &c_f);
#else
        OIIO::sincos(x.val()[i], &s_f, &c_f);
#endif
        float xdx = x.dx()[i], xdy = x.dy()[i]; // x might be aliased
          sine.val()[i] = s_f;   sine.dx()[i] =  c_f * xdx;   sine.dy()[i] =  c_f * xdy;
        cosine.val()[i] = c_f; cosine.dx()[i] = -s_f * xdx; cosine.dy()[i] = -s_f * xdy;
    }
#else
    auto sincos_comp = [](const Dual2<float> & x, float &sine_val, float &sine_dx, float &sine_dy, float &cosine_val, float &cosine_dx, float &cosine_dy) {
        float s_f, c_f;
#if OSL_FAST_MATH
        OIIO::fast_sincos(x.val(), &s_f, &c_f);
#else
        OIIO::sincos(x.val(), &s_f, &c_f);
#endif
        sine_val = s_f;     sine_dx =  c_f * x.dx();   sine_dy =  c_f * x.dy();
        cosine_val = c_f; cosine_dx = -s_f * x.dx(); cosine_dy = -s_f * x.dy();
    };

    sincos_comp(comp_x(x), sine.val().x, sine.dx().x, sine.dy().x, cosine.val().x, cosine.dx().x, cosine.dy().x);
    sincos_comp(comp_y(x), sine.val().y, sine.dx().y, sine.dy().y, cosine.val().y, cosine.dx().y, cosine.dy().y);
    sincos_comp(comp_z(x), sine.val().z, sine.dx().z, sine.dy().z, cosine.val().z, cosine.dx().z, cosine.dy().z);


#endif
}

#if OSL_FAST_MATH
MAKE_UNARY_PERCOMPONENT_OP     (log        , OIIO::fast_log       , fast_log)
MAKE_UNARY_PERCOMPONENT_OP     (log2       , OIIO::fast_log2      , fast_log2)
MAKE_UNARY_PERCOMPONENT_OP     (log10      , OIIO::fast_log10     , fast_log10)
MAKE_UNARY_PERCOMPONENT_OP     (exp        , OIIO::fast_exp       , fast_exp)
MAKE_UNARY_PERCOMPONENT_OP     (exp2       , OIIO::fast_exp2      , fast_exp2)
MAKE_UNARY_PERCOMPONENT_OP     (expm1      , OIIO::fast_expm1     , fast_expm1)
MAKE_BINARY_PERCOMPONENT_OP    (pow        , OIIO::fast_safe_pow  , fast_safe_pow)
MAKE_BINARY_PERCOMPONENT_VF_OP (pow        , OIIO::fast_safe_pow  , fast_safe_pow)
MAKE_UNARY_PERCOMPONENT_OP     (erf        , OIIO::fast_erf       , fast_erf)
MAKE_UNARY_PERCOMPONENT_OP     (erfc       , OIIO::fast_erfc      , fast_erfc)
#else
MAKE_UNARY_PERCOMPONENT_OP     (log        , OIIO::safe_log       , safe_log)
MAKE_UNARY_PERCOMPONENT_OP     (log2       , OIIO::safe_log2      , safe_log2)
MAKE_UNARY_PERCOMPONENT_OP     (log10      , OIIO::safe_log10     , safe_log10)
MAKE_UNARY_PERCOMPONENT_OP     (exp        , expf                 , exp)
MAKE_UNARY_PERCOMPONENT_OP     (exp2       , exp2f                , exp2)
MAKE_UNARY_PERCOMPONENT_OP     (expm1      , expm1f               , expm1)
MAKE_BINARY_PERCOMPONENT_OP    (pow        , OIIO::safe_pow       , safe_pow)
MAKE_BINARY_PERCOMPONENT_VF_OP (pow        , OIIO::safe_pow       , safe_pow)
MAKE_UNARY_PERCOMPONENT_OP     (erf        , erff                 , erf)
MAKE_UNARY_PERCOMPONENT_OP     (erfc       , erfcf                , erfc)
#endif

MAKE_UNARY_PERCOMPONENT_OP     (sqrt       , OIIO::safe_sqrt      , sqrt)
MAKE_UNARY_PERCOMPONENT_OP     (inversesqrt, OIIO::safe_inversesqrt, inversesqrt)

OSL_SHADEOP float osl_logb_ff (float x) { return OIIO::fast_logb(x); }
OSL_SHADEOP void osl_logb_vv (void *r, void *x_) {
    const Vec3 &x (VEC(x_));
    VEC(r).setValue (OIIO::fast_logb(x.x), OIIO::fast_logb(x.y), OIIO::fast_logb(x.z));
}

OSL_SHADEOP float osl_floor_ff (float x) { return floorf(x); }
OSL_SHADEOP void osl_floor_vv (void *r, void *x_) {
    const Vec3 &x (VEC(x_));
    VEC(r).setValue (floorf(x.x), floorf(x.y), floorf(x.z));
}
OSL_SHADEOP float osl_ceil_ff (float x) { return ceilf(x); }
OSL_SHADEOP void osl_ceil_vv (void *r, void *x_) {
    const Vec3 &x (VEC(x_));
    VEC(r).setValue (ceilf(x.x), ceilf(x.y), ceilf(x.z));
}
OSL_SHADEOP float osl_round_ff (float x) { return roundf(x); }
OSL_SHADEOP void osl_round_vv (void *r, void *x_) {
    const Vec3 &x (VEC(x_));
    VEC(r).setValue (roundf(x.x), roundf(x.y), roundf(x.z));
}
OSL_SHADEOP float osl_trunc_ff (float x) { return truncf(x); }
OSL_SHADEOP void osl_trunc_vv (void *r, void *x_) {
    const Vec3 &x (VEC(x_));
    VEC(r).setValue (truncf(x.x), truncf(x.y), truncf(x.z));
}
OSL_SHADEOP float osl_sign_ff (float x) {
    return x < 0.0f ? -1.0f : (x==0.0f ? 0.0f : 1.0f);
}
OSL_SHADEOP void osl_sign_vv (void *r, void *x_) {
    const Vec3 &x (VEC(x_));
    VEC(r).setValue (osl_sign_ff(x.x), osl_sign_ff(x.y), osl_sign_ff(x.z));
}
OSL_SHADEOP float osl_step_fff (float edge, float x) {
    return x < edge ? 0.0f : 1.0f;
}
OSL_SHADEOP void osl_step_vvv (void *result, void *edge, void *x) {
    VEC(result).setValue (((float *)x)[0] < ((float *)edge)[0] ? 0.0f : 1.0f,
                          ((float *)x)[1] < ((float *)edge)[1] ? 0.0f : 1.0f,
                          ((float *)x)[2] < ((float *)edge)[2] ? 0.0f : 1.0f);

}

OSL_SHADEOP int osl_isnan_if (float f) { return OIIO::isnan (f); }
OSL_SHADEOP int osl_isinf_if (float f) { return OIIO::isinf (f); }
OSL_SHADEOP int osl_isfinite_if (float f) { return OIIO::isfinite (f); }


OSL_SHADEOP int osl_abs_ii (int x) { return abs(x); }
OSL_SHADEOP int osl_fabs_ii (int x) { return abs(x); }

OSL_HOSTDEVICE inline Dual2<float> fabsf (const Dual2<float> &x) {
    return x.val() >= 0 ? x : -x;
}

MAKE_UNARY_PERCOMPONENT_OP (abs, fabsf, fabsf);
MAKE_UNARY_PERCOMPONENT_OP (fabs, fabsf, fabsf);

OSL_SHADEOP int osl_safe_mod_iii (int a, int b) {
    return (b != 0) ? (a % b) : 0;
}

MAKE_BINARY_PERCOMPONENT_OP (fmod, safe_fmod, safe_fmod);
MAKE_BINARY_PERCOMPONENT_VF_OP (fmod, safe_fmod, safe_fmod)

OSL_SHADEOP float osl_safe_div_fff (float a, float b) {
    return (b != 0.0f) ? (a / b) : 0.0f;
}

OSL_SHADEOP int osl_safe_div_iii (int a, int b) {
    return (b != 0) ? (a / b) : 0;
}

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
    float x = a.x - b.x;
    float y = a.y - b.y;
    float z = a.z - b.z;
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
    using std::sqrt;
    // NOTE: must match with the Dual version of normalize used below
    Vec3 v = VEC(a);
    float len = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 0) {
        float invlen = 1 / len;
        v.x *= invlen;
        v.y *= invlen;
        v.z *= invlen;
    } else
        v.x = v.y = v.z = 0;
    VEC(result) = v;
}

OSL_SHADEOP void
osl_normalize_dvdv (void *result, void *a)
{
    DVEC(result) = normalize(DVEC(a));
}



OSL_HOSTDEVICE inline Vec3 calculatenormal(void *P_, bool flipHandedness)
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



OSL_HOSTDEVICE inline float filter_width(float dx, float dy)
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



// Asked if the raytype includes a bit pattern.
OSL_SHADEOP int osl_raytype_bit (void *sg_, int bit)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    return (sg->raytype & bit) != 0;
}



// extern declaration
OSL_SHADEOP_NOINLINE int osl_range_check_err (int indexvalue, int length,
                         const char *symname, void *sg,
                         const void *sourcefile, int sourceline,
                         const char *groupname, int layer,
                         const char *layername, const char *shadername);



OSL_SHADEOP int
osl_range_check (int indexvalue, int length, const char *symname,
                 void *sg, const void *sourcefile, int sourceline,
                 const char *groupname, int layer, const char *layername,
                 const char *shadername)
{
    if (indexvalue < 0 || indexvalue >= length) {
        indexvalue = osl_range_check_err (indexvalue, length, symname, sg,
                                          sourcefile, sourceline, groupname,
                                          layer, layername, shadername);
    }
    return indexvalue;
}



