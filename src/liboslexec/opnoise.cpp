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
#include "noiseimpl.h"
#include "dual_vec.h"
#include "Imathx.h"

#include <OpenImageIO/fmath.h>

using namespace OSL;


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
#define USTR(cstr) (*((ustring *)&cstr))


#define NOISE_IMPL(opname,implname)                                     \
OSL_SHADEOP float osl_ ##opname## _ff (float x) {                       \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, x);                                                        \
    return r;                                                           \
}                                                                       \
                                                                        \
OSL_SHADEOP float osl_ ##opname## _fff (float x, float y) {             \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, x, y);                                                     \
    return r;                                                           \
}                                                                       \
                                                                        \
OSL_SHADEOP float osl_ ##opname## _fv (char *x) {                       \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, VEC(x));                                                   \
    return r;                                                           \
}                                                                       \
                                                                        \
OSL_SHADEOP float osl_ ##opname## _fvf (char *x, float y) {             \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, VEC(x), y);                                                \
    return r;                                                           \
}                                                                       \
                                                                        \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _vf (char *r, float x) {               \
    implname impl;                                                      \
    impl (VEC(r), x);                                                   \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _vff (char *r, float x, float y) {     \
    implname impl;                                                      \
    impl (VEC(r), x, y);                                                \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _vv (char *r, char *x) {               \
    implname impl;                                                      \
    impl (VEC(r), VEC(x));                                              \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _vvf (char *r, char *x, float y) {     \
    implname impl;                                                      \
    impl (VEC(r), VEC(x), y);                                           \
}





#define NOISE_IMPL_DERIV(opname,implname)                               \
OSL_SHADEOP void osl_ ##opname## _dfdf (char *r, char *x) {             \
    implname impl;                                                      \
    impl (DFLOAT(r), DFLOAT(x));                                        \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdfdf (char *r, char *x, char *y) {  \
    implname impl;                                                      \
    impl (DFLOAT(r), DFLOAT(x), DFLOAT(y));                             \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdff (char *r, char *x, float y) {   \
    implname impl;                                                      \
    impl (DFLOAT(r), DFLOAT(x), Dual2<float>(y));                       \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dffdf (char *r, float x, char *y) {   \
    implname impl;                                                      \
    impl (DFLOAT(r), Dual2<float>(x), DFLOAT(y));                       \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdv (char *r, char *x) {             \
    implname impl;                                                      \
    impl (DFLOAT(r), DVEC(x));                                          \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdvdf (char *r, char *x, char *y) {  \
    implname impl;                                                      \
    impl (DFLOAT(r), DVEC(x), DFLOAT(y));                               \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdvf (char *r, char *x, float y) {   \
    implname impl;                                                      \
    impl (DFLOAT(r), DVEC(x), Dual2<float>(y));                         \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfvdf (char *r, char *x, char *y) {   \
    implname impl;                                                      \
    impl (DFLOAT(r), Dual2<Vec3>(VEC(x)), DFLOAT(y));                   \
}                                                                       \
                                                                        \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdf (char *r, char *x) {             \
    implname impl;                                                      \
    impl (DVEC(r), DFLOAT(x));                                          \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdfdf (char *r, char *x, char *y) {  \
    implname impl;                                                      \
    impl (DVEC(r), DFLOAT(x), DFLOAT(y));                               \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdff (char *r, char *x, float y) {   \
    implname impl;                                                      \
    impl (DVEC(r), DFLOAT(x), Dual2<float>(y));                         \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvfdf (char *r, float x, char *y) {   \
    implname impl;                                                      \
    impl (DVEC(r), Dual2<float>(x), DFLOAT(y));                         \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdv (char *r, char *x) {             \
    implname impl;                                                      \
    impl (DVEC(r), DVEC(x));                                            \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdvdf (char *r, char *x, char *y) {  \
    implname impl;                                                      \
    impl (DVEC(r), DVEC(x), DFLOAT(y));                                 \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdvf (char *r, char *x, float y) {   \
    implname impl;                                                      \
    impl (DVEC(r), DVEC(x), Dual2<float>(y));                           \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvvdf (char *r, char *x, char *y) {   \
    implname impl;                                                      \
    impl (DVEC(r), Dual2<Vec3>(VEC(x)), DFLOAT(y));                     \
}




#define NOISE_IMPL_DERIV_OPT(opname,implname)                           \
OSL_SHADEOP void osl_ ##opname## _dfdf (char *name, char *r, char *x, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (USTR(name), DFLOAT(r), DFLOAT(x), (ShaderGlobals *)sg, (NoiseParams *)opt);                                   \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdfdf (char *name, char *r, char *x, char *y, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (USTR(name), DFLOAT(r), DFLOAT(x), DFLOAT(y), (ShaderGlobals *)sg, (NoiseParams *)opt);                        \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdv (char *name, char *r, char *x, char *sg, char *opt) {  \
    implname impl;                                                      \
    impl (USTR(name), DFLOAT(r), DVEC(x), (ShaderGlobals *)sg, (NoiseParams *)opt);                                     \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdvdf (char *name, char *r, char *x, char *y, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (USTR(name), DFLOAT(r), DVEC(x), DFLOAT(y), (ShaderGlobals *)sg, (NoiseParams *)opt);                          \
}                                                                       \
                                                                        \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdf (char *name, char *r, char *x, char *sg, char *opt) {  \
    implname impl;                                                      \
    impl (USTR(name), DVEC(r), DFLOAT(x), (ShaderGlobals *)sg, (NoiseParams *)opt);                                     \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdfdf (char *name, char *r, char *x, char *y, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (USTR(name), DVEC(r), DFLOAT(x), DFLOAT(y), (ShaderGlobals *)sg, (NoiseParams *)opt);                                     \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdv (char *name, char *r, char *x, char *sg, char *opt) {  \
    implname impl;                                                      \
    impl (USTR(name), DVEC(r), DVEC(x), (ShaderGlobals *)sg, (NoiseParams *)opt);                                       \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdvdf (char *name, char *r, char *x, char *y, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (USTR(name), DVEC(r), DVEC(x), DFLOAT(y), (ShaderGlobals *)sg, (NoiseParams *)opt);                            \
}




NOISE_IMPL (cellnoise, CellNoise)
NOISE_IMPL (noise, Noise)
NOISE_IMPL_DERIV (noise, Noise)
NOISE_IMPL (snoise, SNoise)
NOISE_IMPL_DERIV (snoise, SNoise)



#define PNOISE_IMPL(opname,implname)                                    \
    OSL_SHADEOP float osl_ ##opname## _fff (float x, float px) {        \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, x, px);                                                    \
    return r;                                                           \
}                                                                       \
                                                                        \
OSL_SHADEOP float osl_ ##opname## _fffff (float x, float y, float px, float py) { \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, x, y, px, py);                                             \
    return r;                                                           \
}                                                                       \
                                                                        \
OSL_SHADEOP float osl_ ##opname## _fvv (char *x, char *px) {            \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, VEC(x), VEC(px));                                          \
    return r;                                                           \
}                                                                       \
                                                                        \
OSL_SHADEOP float osl_ ##opname## _fvfvf (char *x, float y, char *px, float py) { \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, VEC(x), y, VEC(px), py);                                   \
    return r;                                                           \
}                                                                       \
                                                                        \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _vff (char *r, float x, float px) {    \
    implname impl;                                                      \
    impl (VEC(r), x, px);                                               \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _vffff (char *r, float x, float y, float px, float py) { \
    implname impl;                                                      \
    impl (VEC(r), x, y, px, py);                                        \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _vvv (char *r, char *x, char *px) {    \
    implname impl;                                                      \
    impl (VEC(r), VEC(x), VEC(px));                                     \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _vvfvf (char *r, char *x, float y, char *px, float py) { \
    implname impl;                                                      \
    impl (VEC(r), VEC(x), y, VEC(px), py);                              \
}





#define PNOISE_IMPL_DERIV(opname,implname)                              \
OSL_SHADEOP void osl_ ##opname## _dfdff (char *r, char *x, float px) {  \
    implname impl;                                                      \
    impl (DFLOAT(r), DFLOAT(x), px);                                    \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdfdfff (char *r, char *x, char *y, float px, float py) { \
    implname impl;                                                      \
    impl (DFLOAT(r), DFLOAT(x), DFLOAT(y), px, py);                     \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdffff (char *r, char *x, float y, float px, float py) { \
    implname impl;                                                      \
    impl (DFLOAT(r), DFLOAT(x), Dual2<float>(y), px, py);               \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dffdfff (char *r, float x, char *y, float px, float py) { \
    implname impl;                                                      \
    impl (DFLOAT(r), Dual2<float>(x), DFLOAT(y), px, py);               \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdvv (char *r, char *x, char *px) {  \
    implname impl;                                                      \
    impl (DFLOAT(r), DVEC(x), VEC(px));                                 \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdvdfvf (char *r, char *x, char *y, char *px, float py) { \
    implname impl;                                                      \
    impl (DFLOAT(r), DVEC(x), DFLOAT(y), VEC(px), py);                  \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdvfvf (char *r, char *x, float y, char *px, float py) { \
    implname impl;                                                      \
    impl (DFLOAT(r), DVEC(x), Dual2<float>(y), VEC(px), py);            \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfvdfvf (char *r, char *x, char *y, char *px, float py) { \
    implname impl;                                                      \
    impl (DFLOAT(r), Dual2<Vec3>(VEC(x)), DFLOAT(y), VEC(px), py);      \
}                                                                       \
                                                                        \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdff (char *r, char *x, float px) {  \
    implname impl;                                                      \
    impl (DVEC(r), DFLOAT(x), px);                                      \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdfdfff (char *r, char *x, char *y, float px, float py) { \
    implname impl;                                                      \
    impl (DVEC(r), DFLOAT(x), DFLOAT(y), px, py);                       \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdffff (char *r, char *x, float y, float px, float py) { \
    implname impl;                                                      \
    impl (DVEC(r), DFLOAT(x), Dual2<float>(y), px, py);                 \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvfdfff (char *r, float x, char *y, float px, float py) { \
    implname impl;                                                      \
    impl (DVEC(r), Dual2<float>(x), DFLOAT(y), px, py);                 \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdvv (char *r, char *x, char *px) {  \
    implname impl;                                                      \
    impl (DVEC(r), DVEC(x), VEC(px));                                   \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdvdfvf (char *r, char *x, char *y, char *px, float py) { \
    implname impl;                                                      \
    impl (DVEC(r), DVEC(x), DFLOAT(y), VEC(px), py);                    \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdvfvf (char *r, char *x, float y, float *px, float py) { \
    implname impl;                                                      \
    impl (DVEC(r), DVEC(x), Dual2<float>(y), VEC(px), py);              \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvvdfvf (char *r, char *x, char *px, char *y, float py) { \
    implname impl;                                                      \
    impl (DVEC(r), Dual2<Vec3>(VEC(x)), DFLOAT(y), VEC(px), py);        \
}




#define PNOISE_IMPL_DERIV_OPT(opname,implname)                          \
OSL_SHADEOP void osl_ ##opname## _dfdff (char *name, char *r, char *x, float px, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (USTR(name), DFLOAT(r), DFLOAT(x), px, (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdfdfff (char *name, char *r, char *x, char *y, float px, float py, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (USTR(name), DFLOAT(r), DFLOAT(x), DFLOAT(y), px, py, (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdvv (char *name, char *r, char *x, char *px, char *sg, char *opt) {  \
    implname impl;                                                      \
    impl (USTR(name), DFLOAT(r), DVEC(x), VEC(px), (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdvdfvf (char *name, char *r, char *x, char *y, char *px, float py, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (USTR(name), DFLOAT(r), DVEC(x), DFLOAT(y), VEC(px), py, (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdff (char *name, char *r, char *x, float px, char *sg, char *opt) {  \
    implname impl;                                                      \
    impl (USTR(name), DVEC(r), DFLOAT(x), px, (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdfdfff (char *name, char *r, char *x, char *y, float px, float py, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (USTR(name), DVEC(r), DFLOAT(x), DFLOAT(y), px, py, (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdvv (char *name, char *r, char *x, char *px, char *sg, char *opt) {  \
    implname impl;                                                      \
    impl (USTR(name), DVEC(r), DVEC(x), VEC(px), (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdvdfvf (char *name, char *r, char *x, char *y, char *px, float py, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (USTR(name), DVEC(r), DVEC(x), DFLOAT(y), VEC(px), py, (ShaderGlobals *)sg, (NoiseParams *)opt); \
}




PNOISE_IMPL (pcellnoise, PeriodicCellNoise)
PNOISE_IMPL (pnoise, PeriodicNoise)
PNOISE_IMPL_DERIV (pnoise, PeriodicNoise)
PNOISE_IMPL (psnoise, PeriodicSNoise)
PNOISE_IMPL_DERIV (psnoise, PeriodicSNoise)



struct GaborNoise {
    GaborNoise () { }

    // Gabor always uses derivatives, so dual versions only

    inline void operator() (ustring noisename, Dual2<float> &result,
                            const Dual2<float> &x,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = gabor (x, opt);
    }

    inline void operator() (ustring noisename, Dual2<float> &result,
                            const Dual2<float> &x, const Dual2<float> &y,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = gabor (x, y, opt);
    }

    inline void operator() (ustring noisename, Dual2<float> &result,
                            const Dual2<Vec3> &p,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = gabor (p, opt);
    }

    inline void operator() (ustring noisename, Dual2<float> &result,
                            const Dual2<Vec3> &p, const Dual2<float> &t,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        // FIXME -- This is very broken, we are ignoring 4D!
        result = gabor (p, opt);
    }

    inline void operator() (ustring noisename, Dual2<Vec3> &result,
                            const Dual2<float> &x,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = gabor3 (x, opt);
    }

    inline void operator() (ustring noisename, Dual2<Vec3> &result,
                            const Dual2<float> &x, const Dual2<float> &y,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = gabor3 (x, y, opt);
    }

    inline void operator() (ustring noisename, Dual2<Vec3> &result,
                            const Dual2<Vec3> &p,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = gabor3 (p, opt);
    }

    inline void operator() (ustring noisename, Dual2<Vec3> &result,
                            const Dual2<Vec3> &p, const Dual2<float> &t,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        // FIXME -- This is very broken, we are ignoring 4D!
        result = gabor3 (p, opt);
    }
};



struct GaborPNoise {
    GaborPNoise () { }

    // Gabor always uses derivatives, so dual versions only

    inline void operator() (ustring noisename, Dual2<float> &result,
                            const Dual2<float> &x, float px,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = pgabor (x, px, opt);
    }

    inline void operator() (ustring noisename, Dual2<float> &result,
                            const Dual2<float> &x, const Dual2<float> &y,
                            float px, float py,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = pgabor (x, y, px, py, opt);
    }

    inline void operator() (ustring noisename, Dual2<float> &result,
                            const Dual2<Vec3> &p, const Vec3 &pp,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = pgabor (p, pp, opt);
    }

    inline void operator() (ustring noisename, Dual2<float> &result,
                            const Dual2<Vec3> &p, const Dual2<float> &t,
                            const Vec3 &pp, float tp,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        // FIXME -- This is very broken, we are ignoring 4D!
        result = pgabor (p, pp, opt);
    }

    inline void operator() (ustring noisename, Dual2<Vec3> &result,
                            const Dual2<float> &x, float px,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = pgabor (x, px, opt);
    }

    inline void operator() (ustring noisename, Dual2<Vec3> &result,
                            const Dual2<float> &x, const Dual2<float> &y,
                            float px, float py,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = pgabor (x, y, px, py, opt);
    }

    inline void operator() (ustring noisename, Dual2<Vec3> &result,
                            const Dual2<Vec3> &p, const Vec3 &pp,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = pgabor (p, pp, opt);
    }

    inline void operator() (ustring noisename, Dual2<Vec3> &result,
                            const Dual2<Vec3> &p, const Dual2<float> &t,
                            const Vec3 &pp, float tp,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        // FIXME -- This is very broken, we are ignoring 4D!
        result = pgabor (p, pp, opt);
    }
};



NOISE_IMPL_DERIV_OPT (gabornoise, GaborNoise)
PNOISE_IMPL_DERIV_OPT (gaborpnoise, GaborPNoise)



struct GenericNoise {
    GenericNoise () { }

    // Template on R, S, and T to be either float or Vec3

    // dual versions -- this is always called with derivs

    template<class R, class S>
    inline void operator() (ustring name, Dual2<R> &result, const Dual2<S> &s,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        if (name == Strings::uperlin || name == Strings::noise) {
            Noise noise;
            noise(result, s);
        } else if (name == Strings::perlin || name == Strings::snoise) {
            SNoise snoise;
            snoise(result, s);
        } else if (name == Strings::cell) {
            CellNoise cellnoise;
            cellnoise(result.val(), s.val());
            result.clear_d();
        } else if (name == Strings::gabor) {
            GaborNoise gnoise;
            gnoise (name, result, s, sg, opt);
        } else {
            ((ShadingContext *)sg->context)->shadingsys().error ("Unknown noise type \"%s\"", name.c_str());
        }
    }

    template<class R, class S, class T>
    inline void operator() (ustring name, Dual2<R> &result,
                            const Dual2<S> &s, const Dual2<T> &t,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        if (name == Strings::uperlin || name == Strings::noise) {
            Noise noise;
            noise(result, s, t);
        } else if (name == Strings::perlin || name == Strings::snoise) {
            SNoise snoise;
            snoise(result, s, t);
        } else if (name == Strings::cell) {
            CellNoise cellnoise;
            cellnoise(result.val(), s.val(), t.val());
            result.clear_d();
        } else if (name == Strings::gabor) {
            GaborNoise gnoise;
            gnoise (name, result, s, t, sg, opt);
        } else {
            ((ShadingContext *)sg->context)->shadingsys().error ("Unknown noise type \"%s\"", name.c_str());
        }
    }
};


NOISE_IMPL_DERIV_OPT (genericnoise, GenericNoise)


struct GenericPNoise {
    GenericPNoise () { }

    // Template on R, S, and T to be either float or Vec3

    // dual versions -- this is always called with derivs

    template<class R, class S>
    inline void operator() (ustring name, Dual2<R> &result, const Dual2<S> &s,
                            const S &sp,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        if (name == Strings::uperlin || name == Strings::noise) {
            PeriodicNoise noise;
            noise(result, s, sp);
        } else if (name == Strings::perlin || name == Strings::snoise) {
            PeriodicSNoise snoise;
            snoise(result, s, sp);
        } else if (name == Strings::cell) {
            PeriodicCellNoise cellnoise;
            cellnoise(result.val(), s.val(), sp);
            result.clear_d();
        } else if (name == Strings::gabor) {
            GaborPNoise gnoise;
            gnoise (name, result, s, sp, sg, opt);
        } else {
            ((ShadingContext *)sg->context)->shadingsys().error ("Unknown noise type \"%s\"", name.c_str());
        }
    }

    template<class R, class S, class T>
    inline void operator() (ustring name, Dual2<R> &result,
                            const Dual2<S> &s, const Dual2<T> &t,
                            const S &sp, const T &tp,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        if (name == Strings::uperlin || name == Strings::noise) {
            PeriodicNoise noise;
            noise(result, s, t, sp, tp);
        } else if (name == Strings::perlin || name == Strings::snoise) {
            PeriodicSNoise snoise;
            snoise(result, s, t, sp, tp);
        } else if (name == Strings::cell) {
            PeriodicCellNoise cellnoise;
            cellnoise(result.val(), s.val(), t.val(), sp, tp);
            result.clear_d();
        } else if (name == Strings::gabor) {
            GaborPNoise gnoise;
            gnoise (name, result, s, t, sp, tp, sg, opt);
        } else {
            ((ShadingContext *)sg->context)->shadingsys().error ("Unknown noise type \"%s\"", name.c_str());
        }
    }
};


PNOISE_IMPL_DERIV_OPT (genericpnoise, GenericPNoise)


#endif
