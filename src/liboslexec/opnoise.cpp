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

#include <limits>

#include "oslexec_pvt.h"
#include "oslops.h"
#include "noiseimpl.h"
#include "OpenImageIO/varyingref.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {


template <typename FUNCTION>
DECLOP (generic_noise_function_noderivs)
{
    ASSERT (nargs == 2 || nargs == 3);
    OpImpl impl = NULL;
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));

    // type check first args
    ASSERT (Result.typespec().is_float() || Result.typespec().is_triple());
    ASSERT (A.typespec().is_float() || A.typespec().is_triple());

    if (nargs == 2) {
        // either ff or fp
        if (Result.typespec().is_float() && A.typespec().is_float())
            impl = unary_op_noderivs<float, float, FUNCTION>;
        else if (Result.typespec().is_float() && A.typespec().is_triple())
            impl = unary_op_noderivs<float, Vec3, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_float())
            impl = unary_op_noderivs<Vec3, float, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_triple())
            impl = unary_op_noderivs<Vec3, Vec3, FUNCTION>;
    } else if (nargs == 3) {
        // either fff or fpf
        Symbol &B (exec->sym (args[2]));
        ASSERT (B.typespec().is_float());
        if (Result.typespec().is_float() && A.typespec().is_float())
            impl = binary_op_noderivs<float, float, float, FUNCTION>;
        else if (Result.typespec().is_float() && A.typespec().is_triple())
            impl = binary_op_noderivs<float, Vec3, float, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_float())
            impl = binary_op_noderivs<Vec3, float, float, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_triple())
            impl = binary_op_noderivs<Vec3, Vec3, float, FUNCTION>;
    }

    if (impl) {
        impl (exec, nargs, args);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        exec->error_arg_types ();
        ASSERT (0 && "Function arg type can't be handled");
    }
}

template <typename FUNCTION>
DECLOP (generic_noise_function)
{
    ASSERT (nargs == 2 || nargs == 3);
    OpImpl impl = NULL;
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));

    // type check first args
    ASSERT (Result.typespec().is_float() || Result.typespec().is_triple());
    ASSERT (A.typespec().is_float() || A.typespec().is_triple());

    if (nargs == 2) {
        // either ff or fp
        if (Result.typespec().is_float() && A.typespec().is_float())
            impl = unary_op<float, float, FUNCTION>;
        else if (Result.typespec().is_float() && A.typespec().is_triple())
            impl = unary_op<float, Vec3, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_float())
            impl = unary_op<Vec3, float, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_triple())
            impl = unary_op<Vec3, Vec3, FUNCTION>;
    } else if (nargs == 3) {
        // either fff or fpf
        Symbol &B (exec->sym (args[2]));
        ASSERT (B.typespec().is_float());
        if (Result.typespec().is_float() && A.typespec().is_float())
            impl = binary_op<float, float, float, FUNCTION>;
        else if (Result.typespec().is_float() && A.typespec().is_triple())
            impl = binary_op<float, Vec3, float, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_float())
            impl = binary_op<Vec3, float, float, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_triple())
            impl = binary_op<Vec3, Vec3, float, FUNCTION>;
    }

    if (impl) {
        impl (exec, nargs, args);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        exec->error_arg_types ();
        ASSERT (0 && "Function arg type can't be handled");
    }
}



template <typename FUNCTION>
DECLOP (generic_pnoise_function)
{
    ASSERT (nargs == 3 || nargs == 5);
    OpImpl impl = NULL;
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));

    // type check first args
    ASSERT (Result.typespec().is_float() || Result.typespec().is_triple());
    ASSERT (A.typespec().is_float() || A.typespec().is_triple());

    if (nargs == 3) {
        // either fff or fpp
        ASSERT (B.typespec().is_float() || B.typespec().is_triple());
        ASSERT (A.typespec().is_float() == B.typespec().is_float());
        ASSERT (A.typespec().is_triple() == B.typespec().is_triple());

        // this isn't a regular binary op because we don't care about the
        // derivatives of the period (since that value gets floored)
        // so we manually instantiate binary_op_guts only considering the first
        // arguments need for derivatives
        if (Result.typespec().is_float() && A.typespec().is_float())
            impl = binary_op_unary_derivs<float, float, float, FUNCTION>;
        else if (Result.typespec().is_float() && A.typespec().is_triple())
            impl = binary_op_unary_derivs<float, Vec3, Vec3, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_float())
            impl = binary_op_unary_derivs<Vec3, float, float, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_triple())
            impl = binary_op_unary_derivs<Vec3, Vec3, Vec3, FUNCTION>;
    } else if (nargs == 5) {
        // either fffff or fpfpf
        Symbol &C (exec->sym (args[3]));
        Symbol &D (exec->sym (args[4]));

        ASSERT (B.typespec().is_float());
        ASSERT (C.typespec().is_float() || C.typespec().is_triple());
        ASSERT (D.typespec().is_float());
        ASSERT (C.typespec() == A.typespec());

        if (Result.typespec().is_float() && A.typespec().is_float())
            impl = quaternary_op_binary_derivs<float, float, float, float, float, FUNCTION>;
        else if (Result.typespec().is_float() && A.typespec().is_triple())
            impl = quaternary_op_binary_derivs<float, Vec3, float, Vec3, float, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_float())
            impl = quaternary_op_binary_derivs<Vec3, float, float, float, float, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_triple())
            impl = quaternary_op_binary_derivs<Vec3, Vec3, float, Vec3, float, FUNCTION>;
    }

    if (impl) {
        impl (exec, nargs, args);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        exec->error_arg_types ();
        ASSERT (0 && "Function arg type can't be handled");
    }
}



DECLOP (OP_cellnoise)
{
    // NOTE: cellnoise is a step function which is locally flat
    //       therefore its derivatives are always 0
    generic_noise_function_noderivs<CellNoise> (exec, nargs, args);
}



DECLOP (OP_noise)
{
    generic_noise_function<Noise> (exec, nargs, args);
}



DECLOP (OP_snoise)
{
    generic_noise_function<SNoise> (exec, nargs, args);
}



DECLOP (OP_pnoise)
{
    generic_pnoise_function<PeriodicNoise> (exec, nargs, args);
}



DECLOP (OP_psnoise)
{
    generic_pnoise_function<PeriodicSNoise> (exec, nargs, args);
}


}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif

#if 0 // only when testing the statistics of perlin noise to normalize the range

#include <boost/random.hpp>

void test_perlin(int d) {
    HashScalar h;
    float noise_min = +std::numeric_limits<float>::max();
    float noise_max = -std::numeric_limits<float>::max();
    float noise_avg = 0;
    float noise_avg2 = 0;
    float noise_stddev;
    boost::mt19937 rndgen;
    boost::uniform_01<boost::mt19937, float> rnd(rndgen);
    printf("Running perlin-%d noise test ...\n", d);
    const int n = 100000000;
    const float r = 1024;
    for (int i = 0; i < n; i++) {
        float noise;
        float nx = rnd(); nx = (2 * nx - 1) * r;
        float ny = rnd(); ny = (2 * ny - 1) * r;
        float nz = rnd(); nz = (2 * nz - 1) * r;
        float nw = rnd(); nw = (2 * nw - 1) * r;
        switch (d) {
            case 1: perlin(noise, h, nx); break;
            case 2: perlin(noise, h, nx, ny); break;
            case 3: perlin(noise, h, nx, ny, nz); break;
            case 4: perlin(noise, h, nx, ny, nz, nw); break;
        }
        if (noise_min > noise) noise_min = noise;
        if (noise_max < noise) noise_max = noise;
        noise_avg += noise;
        noise_avg2 += noise * noise;
    }
    noise_avg /= n;
    noise_stddev = std::sqrt((noise_avg2 - noise_avg * noise_avg * n) / n);
    printf("Result: perlin-%d noise stats:\n\tmin: %.17g\n\tmax: %.17g\n\tavg: %.17g\n\tdev: %.17g\n",
            d, noise_min, noise_max, noise_avg, noise_stddev);
    printf("Normalization: %.17g\n", 1.0f / std::max(fabsf(noise_min), fabsf(noise_max)));
}

#endif



/***********************************************************************
 * noise routines callable by the LLVM-generated code.
 */

#if 1
// Handy re-casting macros
#define VEC(v) (*(Vec3 *)v)
#define DFLOAT(x) (*(Dual2<Float> *)x)
#define DVEC(x) (*(Dual2<Vec3> *)x)


#define NOISE_IMPL(opname,implname)                                     \
extern "C" float osl_ ##opname## _ff (float x) {                        \
    implname impl(NULL);                                                \
    float r;                                                            \
    impl (r, x);                                                        \
    return r;                                                           \
}                                                                       \
                                                                        \
extern "C" float osl_ ##opname## _fff (float x, float y) {              \
    implname impl(NULL);                                                \
    float r;                                                            \
    impl (r, x, y);                                                     \
    return r;                                                           \
}                                                                       \
                                                                        \
extern "C" float osl_ ##opname## _fv (void *x) {                        \
    implname impl(NULL);                                                \
    float r;                                                            \
    impl (r, VEC(x));                                                   \
    return r;                                                           \
}                                                                       \
                                                                        \
extern "C" float osl_ ##opname## _fvf (void *x, float y) {              \
    implname impl(NULL);                                                \
    float r;                                                            \
    impl (r, VEC(x), y);                                                \
    return r;                                                           \
}                                                                       \
                                                                        \
                                                                        \
extern "C" void osl_ ##opname## _vf (void *r, float x) {                \
    implname impl(NULL);                                                \
    impl (VEC(r), x);                                                   \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _vff (void *r, float x, float y) {      \
    implname impl(NULL);                                                \
    impl (VEC(r), x, y);                                                \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _vv (void *r, void *x) {                \
    implname impl(NULL);                                                \
    impl (VEC(r), VEC(x));                                              \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _vvf (void *r, void *x, float y) {      \
    implname impl(NULL);                                                \
    impl (VEC(r), VEC(x), y);                                           \
}





#define NOISE_IMPL_DERIV(opname,implname)                               \
extern "C" void osl_ ##opname## _dfdf (void *r, void *x) {              \
    implname impl(NULL);                                                \
    impl (DFLOAT(r), DFLOAT(x));                                        \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dfdfdf (void *r, void *x, void *y) {   \
    implname impl(NULL);                                                \
    impl (DFLOAT(r), DFLOAT(x), DFLOAT(y));                             \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dfdff (void *r, void *x, float y) {    \
    implname impl(NULL);                                                \
    impl (DFLOAT(r), DFLOAT(x), Dual2<float>(y));                       \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dffdf (void *r, float x, void *y) {    \
    implname impl(NULL);                                                \
    impl (DFLOAT(r), Dual2<float>(x), DFLOAT(y));                       \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dfdv (void *r, void *x) {              \
    implname impl(NULL);                                                \
    impl (DFLOAT(r), DVEC(x));                                          \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dfdvdf (void *r, void *x, void *y) {   \
    implname impl(NULL);                                                \
    impl (DFLOAT(r), DVEC(x), DFLOAT(y));                               \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dfdvf (void *r, void *x, float y) {    \
    implname impl(NULL);                                                \
    impl (DFLOAT(r), DVEC(x), Dual2<float>(y));                         \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dfvdf (void *r, void *x, void *y) {    \
    implname impl(NULL);                                                \
    impl (DFLOAT(r), Dual2<Vec3>(VEC(x)), DFLOAT(y));                   \
}                                                                       \
                                                                        \
                                                                        \
extern "C" void osl_ ##opname## _dvdf (void *r, void *x) {              \
    implname impl(NULL);                                                \
    impl (DVEC(r), DFLOAT(x));                                          \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dvdfdf (void *r, void *x, void *y) {   \
    implname impl(NULL);                                                \
    impl (DVEC(r), DFLOAT(x), DFLOAT(y));                               \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dvdff (void *r, void *x, float y) {    \
    implname impl(NULL);                                                \
    impl (DVEC(r), DFLOAT(x), Dual2<float>(y));                         \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dvfdf (void *r, float x, void *y) {    \
    implname impl(NULL);                                                \
    impl (DVEC(r), Dual2<float>(x), DFLOAT(y));                         \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dvdv (void *r, void *x) {              \
    implname impl(NULL);                                                \
    impl (DVEC(r), DVEC(x));                                            \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dvdvdf (void *r, void *x, void *y) {   \
    implname impl(NULL);                                                \
    impl (DVEC(r), DVEC(x), DFLOAT(y));                                 \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dvdvf (void *r, void *x, float y) {    \
    implname impl(NULL);                                                \
    impl (DVEC(r), DVEC(x), Dual2<float>(y));                           \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dvvdf (void *r, void *x, void *y) {    \
    implname impl(NULL);                                                \
    impl (DVEC(r), Dual2<Vec3>(VEC(x)), DFLOAT(y));                     \
}




NOISE_IMPL (cellnoise, CellNoise)

NOISE_IMPL (noise, Noise)
NOISE_IMPL_DERIV (noise, Noise)
NOISE_IMPL (snoise, SNoise)
NOISE_IMPL_DERIV (snoise, SNoise)



#define PNOISE_IMPL(opname,implname)                                    \
    extern "C" float osl_ ##opname## _fff (float x, float px) {         \
    implname impl(NULL);                                                \
    float r;                                                            \
    impl (r, x, px);                                                    \
    return r;                                                           \
}                                                                       \
                                                                        \
extern "C" float osl_ ##opname## _fffff (float x, float y, float px, float py) { \
    implname impl(NULL);                                                \
    float r;                                                            \
    impl (r, x, y, px, py);                                             \
    return r;                                                           \
}                                                                       \
                                                                        \
extern "C" float osl_ ##opname## _fvv (void *x, void *px) {             \
    implname impl(NULL);                                                \
    float r;                                                            \
    impl (r, VEC(x), VEC(px));                                          \
    return r;                                                           \
}                                                                       \
                                                                        \
extern "C" float osl_ ##opname## _fvfvf (void *x, float y, void *px, float py) { \
    implname impl(NULL);                                                \
    float r;                                                            \
    impl (r, VEC(x), y, VEC(px), py);                                   \
    return r;                                                           \
}                                                                       \
                                                                        \
                                                                        \
extern "C" void osl_ ##opname## _vff (void *r, float x, float px) {    \
    implname impl(NULL);                                                \
    impl (VEC(r), x, px);                                               \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _vffff (void *r, float x, float y, float px, float py) { \
    implname impl(NULL);                                                \
    impl (VEC(r), x, y, px, py);                                        \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _vvv (void *r, void *x, void *px) {    \
    implname impl(NULL);                                                \
    impl (VEC(r), VEC(x), VEC(px));                                     \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _vvfvf (void *r, void *x, float y, void *px, float py) { \
    implname impl(NULL);                                                \
    impl (VEC(r), VEC(x), y, VEC(px), py);                              \
}





#define PNOISE_IMPL_DERIV(opname,implname)                              \
extern "C" void osl_ ##opname## _dfdff (void *r, void *x, float px) {  \
    implname impl(NULL);                                                \
    impl (DFLOAT(r), DFLOAT(x), px);                                    \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dfdfdfff (void *r, void *x, void *y, float px, float py) { \
    implname impl(NULL);                                                \
    impl (DFLOAT(r), DFLOAT(x), DFLOAT(y), px, py);                     \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dfdffff (void *r, void *x, float y, float px, float py) { \
    implname impl(NULL);                                                \
    impl (DFLOAT(r), DFLOAT(x), Dual2<float>(y), px, py);               \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dffdfff (void *r, float x, void *y, float px, float py) { \
    implname impl(NULL);                                                \
    impl (DFLOAT(r), Dual2<float>(x), DFLOAT(y), px, py);               \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dfdvv (void *r, void *x, void *px) {  \
    implname impl(NULL);                                                \
    impl (DFLOAT(r), DVEC(x), VEC(px));                                 \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dfdvdfvf (void *r, void *x, void *y, void *px, float py) { \
    implname impl(NULL);                                                \
    impl (DFLOAT(r), DVEC(x), DFLOAT(y), VEC(px), py);                  \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dfdvfvf (void *r, void *x, float y, void *px, float py) { \
    implname impl(NULL);                                                \
    impl (DFLOAT(r), DVEC(x), Dual2<float>(y), VEC(px), py);            \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dfvdfvf (void *r, void *x, void *y, void *px, float py) { \
    implname impl(NULL);                                                \
    impl (DFLOAT(r), Dual2<Vec3>(VEC(x)), DFLOAT(y), VEC(px), py);      \
}                                                                       \
                                                                        \
                                                                        \
extern "C" void osl_ ##opname## _dvdff (void *r, void *x, float px) {  \
    implname impl(NULL);                                                \
    impl (DVEC(r), DFLOAT(x), px);                                      \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dvdfdfff (void *r, void *x, void *y, float px, float py) { \
    implname impl(NULL);                                                \
    impl (DVEC(r), DFLOAT(x), DFLOAT(y), px, py);                       \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dvdffff (void *r, void *x, float y, float px, float py) { \
    implname impl(NULL);                                                \
    impl (DVEC(r), DFLOAT(x), Dual2<float>(y), px, py);                 \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dvfdfff (void *r, float x, void *y, float px, float py) { \
    implname impl(NULL);                                                \
    impl (DVEC(r), Dual2<float>(x), DFLOAT(y), px, py);                 \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dvdvv (void *r, void *x, void *px) {  \
    implname impl(NULL);                                                \
    impl (DVEC(r), DVEC(x), VEC(px));                                   \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dvdvdfvf (void *r, void *x, void *y, void *px, float py) { \
    implname impl(NULL);                                                \
    impl (DVEC(r), DVEC(x), DFLOAT(y), VEC(px), py);                    \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dvdvfvf (void *r, void *x, float y, float *px, float py) { \
    implname impl(NULL);                                                \
    impl (DVEC(r), DVEC(x), Dual2<float>(y), VEC(px), py);              \
}                                                                       \
                                                                        \
extern "C" void osl_ ##opname## _dvvdfvf (void *r, void *x, void *px, void *y, float py) { \
    implname impl(NULL);                                                \
    impl (DVEC(r), Dual2<Vec3>(VEC(x)), DFLOAT(y), VEC(px), py);        \
}


PNOISE_IMPL (pnoise, PeriodicNoise)
PNOISE_IMPL_DERIV (pnoise, PeriodicNoise)
PNOISE_IMPL (psnoise, PeriodicSNoise)
PNOISE_IMPL_DERIV (psnoise, PeriodicSNoise)

#endif
