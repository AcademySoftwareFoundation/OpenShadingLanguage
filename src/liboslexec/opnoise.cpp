/*
Copyright (c) 2009-2019 Sony Pictures Imageworks Inc., et al.
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
#include <OSL/oslnoise.h>
#include <OSL/dual_vec.h>
#include <OSL/Imathx/Imathx.h>
#include <OSL/device_string.h>

#include <OpenImageIO/fmath.h>

#include "null_noise.h"

OSL_NAMESPACE_ENTER
namespace pvt {


#if 0 // only when testing the statistics of perlin noise to normalize the range

#include <random>

void test_perlin(int d) {
    HashScalar h;
    float noise_min = +std::numeric_limits<float>::max();
    float noise_max = -std::numeric_limits<float>::max();
    float noise_avg = 0;
    float noise_avg2 = 0;
    float noise_stddev;
    std::mt19937 rndgen;
    std::uniform_real_distribution<float> rnd (0.0f, 1.0f);
    printf("Running perlin-%d noise test ...\n", d);
    const int n = 100000000;
    const float r = 1024;
    for (int i = 0; i < n; i++) {
        float noise;
        float nx = rnd(rndgen); nx = (2 * nx - 1) * r;
        float ny = rnd(rndgen); ny = (2 * ny - 1) * r;
        float nz = rnd(rndgen); nz = (2 * nz - 1) * r;
        float nw = rnd(rndgen); nw = (2 * nw - 1) * r;
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


#define NOISE_IMPL(opname,implname)                                     \
OSL_SHADEOP OSL_HOSTDEVICE float osl_ ##opname## _ff (float x) {        \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, x);                                                        \
    return r;                                                           \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE float osl_ ##opname## _fff (float x, float y) { \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, x, y);                                                     \
    return r;                                                           \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE float osl_ ##opname## _fv (char *x) {        \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, VEC(x));                                                   \
    return r;                                                           \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE float osl_ ##opname## _fvf (char *x, float y) { \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, VEC(x), y);                                                \
    return r;                                                           \
}                                                                       \
                                                                        \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _vf (char *r, float x) { \
    implname impl;                                                      \
    impl (VEC(r), x);                                                   \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _vff (char *r, float x, float y) { \
    implname impl;                                                      \
    impl (VEC(r), x, y);                                                \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _vv (char *r, char *x) { \
    implname impl;                                                      \
    impl (VEC(r), VEC(x));                                              \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _vvf (char *r, char *x, float y) { \
    implname impl;                                                      \
    impl (VEC(r), VEC(x), y);                                           \
}





#define NOISE_IMPL_DERIV(opname,implname)                               \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdf (char *r, char *x) { \
    implname impl;                                                      \
    impl (DFLOAT(r), DFLOAT(x));                                        \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdfdf (char *r, char *x, char *y) { \
    implname impl;                                                      \
    impl (DFLOAT(r), DFLOAT(x), DFLOAT(y));                             \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdff (char *r, char *x, float y) { \
    implname impl;                                                      \
    impl (DFLOAT(r), DFLOAT(x), Dual2<float>(y));                       \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dffdf (char *r, float x, char *y) { \
    implname impl;                                                      \
    impl (DFLOAT(r), Dual2<float>(x), DFLOAT(y));                       \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdv (char *r, char *x) { \
    implname impl;                                                      \
    impl (DFLOAT(r), DVEC(x));                                          \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdvdf (char *r, char *x, char *y) { \
    implname impl;                                                      \
    impl (DFLOAT(r), DVEC(x), DFLOAT(y));                               \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdvf (char *r, char *x, float y) { \
    implname impl;                                                      \
    impl (DFLOAT(r), DVEC(x), Dual2<float>(y));                         \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfvdf (char *r, char *x, char *y) { \
    implname impl;                                                      \
    impl (DFLOAT(r), Dual2<Vec3>(VEC(x)), DFLOAT(y));                   \
}                                                                       \
                                                                        \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdf (char *r, char *x) { \
    implname impl;                                                      \
    impl (DVEC(r), DFLOAT(x));                                          \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdfdf (char *r, char *x, char *y) { \
    implname impl;                                                      \
    impl (DVEC(r), DFLOAT(x), DFLOAT(y));                               \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdff (char *r, char *x, float y) { \
    implname impl;                                                      \
    impl (DVEC(r), DFLOAT(x), Dual2<float>(y));                         \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvfdf (char *r, float x, char *y) { \
    implname impl;                                                      \
    impl (DVEC(r), Dual2<float>(x), DFLOAT(y));                         \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdv (char *r, char *x) { \
    implname impl;                                                      \
    impl (DVEC(r), DVEC(x));                                            \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdvdf (char *r, char *x, char *y) { \
    implname impl;                                                      \
    impl (DVEC(r), DVEC(x), DFLOAT(y));                                 \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdvf (char *r, char *x, float y) { \
    implname impl;                                                      \
    impl (DVEC(r), DVEC(x), Dual2<float>(y));                           \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvvdf (char *r, char *x, char *y) { \
    implname impl;                                                      \
    impl (DVEC(r), Dual2<Vec3>(VEC(x)), DFLOAT(y));                     \
}




#define NOISE_IMPL_DERIV_OPT(opname,implname)                           \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdf (char *name, char *r, char *x, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (HDSTR(name), DFLOAT(r), DFLOAT(x), (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdfdf (char *name, char *r, char *x, char *y, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (HDSTR(name), DFLOAT(r), DFLOAT(x), DFLOAT(y), (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdv (char *name, char *r, char *x, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (HDSTR(name), DFLOAT(r), DVEC(x), (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdvdf (char *name, char *r, char *x, char *y, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (HDSTR(name), DFLOAT(r), DVEC(x), DFLOAT(y), (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdf (char *name, char *r, char *x, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (HDSTR(name), DVEC(r), DFLOAT(x), (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdfdf (char *name, char *r, char *x, char *y, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (HDSTR(name), DVEC(r), DFLOAT(x), DFLOAT(y), (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdv (char *name, char *r, char *x, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (HDSTR(name), DVEC(r), DVEC(x), (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdvdf (char *name, char *r, char *x, char *y, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (HDSTR(name), DVEC(r), DVEC(x), DFLOAT(y), (ShaderGlobals *)sg, (NoiseParams *)opt); \
}




NOISE_IMPL (cellnoise, CellNoise)
NOISE_IMPL (hashnoise, HashNoise)

NOISE_IMPL (noise, Noise)
NOISE_IMPL_DERIV (noise, Noise)

NOISE_IMPL (snoise, SNoise)
NOISE_IMPL_DERIV (snoise, SNoise)

NOISE_IMPL (simplexnoise, SimplexNoise)
NOISE_IMPL_DERIV (simplexnoise, SimplexNoise)

NOISE_IMPL (usimplexnoise, USimplexNoise)
NOISE_IMPL_DERIV (usimplexnoise, USimplexNoise)



#define PNOISE_IMPL(opname,implname)                                    \
OSL_SHADEOP OSL_HOSTDEVICE float osl_ ##opname## _fff (float x, float px) { \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, x, px);                                                    \
    return r;                                                           \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE float osl_ ##opname## _fffff (float x, float y, float px, float py) { \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, x, y, px, py);                                             \
    return r;                                                           \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE float osl_ ##opname## _fvv (char *x, char *px) { \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, VEC(x), VEC(px));                                          \
    return r;                                                           \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE float osl_ ##opname## _fvfvf (char *x, float y, char *px, float py) { \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, VEC(x), y, VEC(px), py);                                   \
    return r;                                                           \
}                                                                       \
                                                                        \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _vff (char *r, float x, float px) { \
    implname impl;                                                      \
    impl (VEC(r), x, px);                                               \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _vffff (char *r, float x, float y, float px, float py) { \
    implname impl;                                                      \
    impl (VEC(r), x, y, px, py);                                        \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _vvv (char *r, char *x, char *px) { \
    implname impl;                                                      \
    impl (VEC(r), VEC(x), VEC(px));                                     \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _vvfvf (char *r, char *x, float y, char *px, float py) { \
    implname impl;                                                      \
    impl (VEC(r), VEC(x), y, VEC(px), py);                              \
}





#define PNOISE_IMPL_DERIV(opname,implname)                              \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdff (char *r, char *x, float px) {  \
    implname impl;                                                      \
    impl (DFLOAT(r), DFLOAT(x), px);                                    \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdfdfff (char *r, char *x, char *y, float px, float py) { \
    implname impl;                                                      \
    impl (DFLOAT(r), DFLOAT(x), DFLOAT(y), px, py);                     \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdffff (char *r, char *x, float y, float px, float py) { \
    implname impl;                                                      \
    impl (DFLOAT(r), DFLOAT(x), Dual2<float>(y), px, py);               \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dffdfff (char *r, float x, char *y, float px, float py) { \
    implname impl;                                                      \
    impl (DFLOAT(r), Dual2<float>(x), DFLOAT(y), px, py);               \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdvv (char *r, char *x, char *px) {  \
    implname impl;                                                      \
    impl (DFLOAT(r), DVEC(x), VEC(px));                                 \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdvdfvf (char *r, char *x, char *y, char *px, float py) { \
    implname impl;                                                      \
    impl (DFLOAT(r), DVEC(x), DFLOAT(y), VEC(px), py);                  \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdvfvf (char *r, char *x, float y, char *px, float py) { \
    implname impl;                                                      \
    impl (DFLOAT(r), DVEC(x), Dual2<float>(y), VEC(px), py);            \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfvdfvf (char *r, char *x, char *y, char *px, float py) { \
    implname impl;                                                      \
    impl (DFLOAT(r), Dual2<Vec3>(VEC(x)), DFLOAT(y), VEC(px), py);      \
}                                                                       \
                                                                        \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdff (char *r, char *x, float px) {  \
    implname impl;                                                      \
    impl (DVEC(r), DFLOAT(x), px);                                      \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdfdfff (char *r, char *x, char *y, float px, float py) { \
    implname impl;                                                      \
    impl (DVEC(r), DFLOAT(x), DFLOAT(y), px, py);                       \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdffff (char *r, char *x, float y, float px, float py) { \
    implname impl;                                                      \
    impl (DVEC(r), DFLOAT(x), Dual2<float>(y), px, py);                 \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvfdfff (char *r, float x, char *y, float px, float py) { \
    implname impl;                                                      \
    impl (DVEC(r), Dual2<float>(x), DFLOAT(y), px, py);                 \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdvv (char *r, char *x, char *px) {  \
    implname impl;                                                      \
    impl (DVEC(r), DVEC(x), VEC(px));                                   \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdvdfvf (char *r, char *x, char *y, char *px, float py) { \
    implname impl;                                                      \
    impl (DVEC(r), DVEC(x), DFLOAT(y), VEC(px), py);                    \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdvfvf (char *r, char *x, float y, void *px, float py) { \
    implname impl;                                                      \
    impl (DVEC(r), DVEC(x), Dual2<float>(y), VEC(px), py);              \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvvdfvf (char *r, char *x, char *px, char *y, float py) { \
    implname impl;                                                      \
    impl (DVEC(r), Dual2<Vec3>(VEC(x)), DFLOAT(y), VEC(px), py);        \
}




#define PNOISE_IMPL_DERIV_OPT(opname,implname)                          \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdff (char *name, char *r, char *x, float px, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (HDSTR(name), DFLOAT(r), DFLOAT(x), px, (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdfdfff (char *name, char *r, char *x, char *y, float px, float py, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (HDSTR(name), DFLOAT(r), DFLOAT(x), DFLOAT(y), px, py, (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdvv (char *name, char *r, char *x, char *px, char *sg, char *opt) {  \
    implname impl;                                                      \
    impl (HDSTR(name), DFLOAT(r), DVEC(x), VEC(px), (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dfdvdfvf (char *name, char *r, char *x, char *y, char *px, float py, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (HDSTR(name), DFLOAT(r), DVEC(x), DFLOAT(y), VEC(px), py, (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdff (char *name, char *r, char *x, float px, char *sg, char *opt) {  \
    implname impl;                                                      \
    impl (HDSTR(name), DVEC(r), DFLOAT(x), px, (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdfdfff (char *name, char *r, char *x, char *y, float px, float py, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (HDSTR(name), DVEC(r), DFLOAT(x), DFLOAT(y), px, py, (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdvv (char *name, char *r, char *x, char *px, char *sg, char *opt) {  \
    implname impl;                                                      \
    impl (HDSTR(name), DVEC(r), DVEC(x), VEC(px), (ShaderGlobals *)sg, (NoiseParams *)opt); \
}                                                                       \
                                                                        \
OSL_SHADEOP OSL_HOSTDEVICE void osl_ ##opname## _dvdvdfvf (char *name, char *r, char *x, char *y, char *px, float py, char *sg, char *opt) { \
    implname impl;                                                      \
    impl (HDSTR(name), DVEC(r), DVEC(x), DFLOAT(y), VEC(px), py, (ShaderGlobals *)sg, (NoiseParams *)opt); \
}




PNOISE_IMPL (pcellnoise, PeriodicCellNoise)
PNOISE_IMPL (phashnoise, PeriodicHashNoise)
PNOISE_IMPL (pnoise, PeriodicNoise)
PNOISE_IMPL_DERIV (pnoise, PeriodicNoise)
PNOISE_IMPL (psnoise, PeriodicSNoise)
PNOISE_IMPL_DERIV (psnoise, PeriodicSNoise)



// NB: We are excluding noise functions that require (u)string arguments
//     in the CUDA case, since strings are not currently well-supported
//     by the PTX backend. We will update this once string support has
//     been improved.

struct GaborNoise {
    OSL_HOSTDEVICE GaborNoise () { }

    // Gabor always uses derivatives, so dual versions only

    OSL_HOSTDEVICE
    inline void operator() (StringParam /*noisename*/, Dual2<float> &result,
                            const Dual2<float> &x,
                            ShaderGlobals* /*sg*/, const NoiseParams *opt) const {
        result = gabor (x, opt);
    }

    OSL_HOSTDEVICE
    inline void operator() (StringParam /*noisename*/, Dual2<float> &result,
                            const Dual2<float> &x, const Dual2<float> &y,
                            ShaderGlobals* /*sg*/, const NoiseParams *opt) const {
        result = gabor (x, y, opt);
    }

    OSL_HOSTDEVICE
    inline void operator() (StringParam /*noisename*/, Dual2<float> &result,
                            const Dual2<Vec3> &p,
                            ShaderGlobals* /*sg*/, const NoiseParams *opt) const {
        result = gabor (p, opt);
    }
    
    OSL_HOSTDEVICE
    inline void operator() (StringParam /*noisename*/, Dual2<float> &result,
                            const Dual2<Vec3> &p, const Dual2<float>& /*t*/,
                            ShaderGlobals* /*sg*/, const NoiseParams *opt) const {
        // FIXME -- This is very broken, we are ignoring 4D!
        result = gabor (p, opt);
    }

    OSL_HOSTDEVICE
    inline void operator() (StringParam /*noisename*/, Dual2<Vec3> &result,
                            const Dual2<float> &x,
                            ShaderGlobals* /*sg*/, const NoiseParams *opt) const {
        result = gabor3 (x, opt);
    }

    OSL_HOSTDEVICE
    inline void operator() (StringParam /*noisename*/, Dual2<Vec3> &result,
                            const Dual2<float> &x, const Dual2<float> &y,
                            ShaderGlobals* /*sg*/, const NoiseParams *opt) const {
        result = gabor3 (x, y, opt);
    }

    OSL_HOSTDEVICE
    inline void operator() (StringParam /*noisename*/, Dual2<Vec3> &result,
                            const Dual2<Vec3> &p,
                            ShaderGlobals* /*sg*/, const NoiseParams *opt) const {
        result = gabor3 (p, opt);
    }

    OSL_HOSTDEVICE
    inline void operator() (StringParam /*noisename*/, Dual2<Vec3> &result,
                            const Dual2<Vec3> &p, const Dual2<float>& /*t*/,
                            ShaderGlobals* /*sg*/, const NoiseParams *opt) const {
        // FIXME -- This is very broken, we are ignoring 4D!
        result = gabor3 (p, opt);
    }
};



struct GaborPNoise {
    OSL_HOSTDEVICE GaborPNoise () { }

    // Gabor always uses derivatives, so dual versions only

    OSL_HOSTDEVICE
    inline void operator() (StringParam /*noisename*/, Dual2<float> &result,
                            const Dual2<float> &x, float px,
                            ShaderGlobals* /*sg*/, const NoiseParams *opt) const {
        result = pgabor (x, px, opt);
    }

    OSL_HOSTDEVICE
    inline void operator() (StringParam /*noisename*/, Dual2<float> &result,
                            const Dual2<float> &x, const Dual2<float> &y,
                            float px, float py,
                            ShaderGlobals* /*sg*/, const NoiseParams *opt) const {
        result = pgabor (x, y, px, py, opt);
    }

    OSL_HOSTDEVICE
    inline void operator() (StringParam /*noisename*/, Dual2<float> &result,
                            const Dual2<Vec3> &p, const Vec3 &pp,
                            ShaderGlobals* /*sg*/, const NoiseParams *opt) const {
        result = pgabor (p, pp, opt);
    }

    OSL_HOSTDEVICE
    inline void operator() (StringParam /*noisename*/, Dual2<float> &result,
                            const Dual2<Vec3> &p, const Dual2<float>& /*t*/,
                            const Vec3 &pp, float /*tp*/,
                            ShaderGlobals* /*sg*/, const NoiseParams *opt) const {
        // FIXME -- This is very broken, we are ignoring 4D!
        result = pgabor (p, pp, opt);
    }

    OSL_HOSTDEVICE
    inline void operator() (StringParam /*noisename*/, Dual2<Vec3> &result,
                            const Dual2<float> &x, float px,
                            ShaderGlobals* /*sg*/, const NoiseParams *opt) const {
        result = pgabor3 (x, px, opt);
    }

    OSL_HOSTDEVICE
    inline void operator() (StringParam /*noisename*/, Dual2<Vec3> &result,
                            const Dual2<float> &x, const Dual2<float> &y,
                            float px, float py,
                            ShaderGlobals* /*sg*/, const NoiseParams *opt) const {
        result = pgabor3 (x, y, px, py, opt);
    }

    OSL_HOSTDEVICE
    inline void operator() (StringParam /*noisename*/, Dual2<Vec3> &result,
                            const Dual2<Vec3> &p, const Vec3 &pp,
                            ShaderGlobals* /*sg*/, const NoiseParams *opt) const {
        result = pgabor3 (p, pp, opt);
    }

    OSL_HOSTDEVICE
    inline void operator() (StringParam /*noisename*/, Dual2<Vec3> &result,
                            const Dual2<Vec3> &p, const Dual2<float>& /*t*/,
                            const Vec3 &pp, float /*tp*/,
                            ShaderGlobals* /*sg*/, const NoiseParams *opt) const {
        // FIXME -- This is very broken, we are ignoring 4D!
        result = pgabor3 (p, pp, opt);
    }
};



NOISE_IMPL_DERIV_OPT (gabornoise, GaborNoise)
PNOISE_IMPL_DERIV_OPT (gaborpnoise, GaborPNoise)

// moved struct NullNoise and UNullNoise to null_noise.h
NOISE_IMPL (nullnoise, NullNoise)
NOISE_IMPL_DERIV (nullnoise, NullNoise)
NOISE_IMPL (unullnoise, UNullNoise)
NOISE_IMPL_DERIV (unullnoise, UNullNoise)



struct GenericNoise {
    OSL_HOSTDEVICE GenericNoise () { }

    // Template on R, S, and T to be either float or Vec3

    // dual versions -- this is always called with derivs

    template<class R, class S> OSL_HOSTDEVICE
    inline void operator() (StringParam name, Dual2<R> &result, const Dual2<S> &s,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        if (name == StringParams::uperlin || name == StringParams::noise) {
            Noise noise;
            noise(result, s);
        } else if (name == StringParams::perlin || name == StringParams::snoise) {
            SNoise snoise;
            snoise(result, s);
        } else if (name == StringParams::simplexnoise || name == StringParams::simplex) {
            SimplexNoise simplexnoise;
            simplexnoise(result, s);
        } else if (name == StringParams::usimplexnoise || name == StringParams::usimplex) {
            USimplexNoise usimplexnoise;
            usimplexnoise(result, s);
        } else if (name == StringParams::cell) {
            CellNoise cellnoise;
            cellnoise(result.val(), s.val());
            result.clear_d();
        } else if (name == StringParams::gabor) {
            GaborNoise gnoise;
            gnoise (name, result, s, sg, opt);
        } else if (name == StringParams::null) {
            NullNoise noise; noise(result, s);
        } else if (name == StringParams::unull) {
            UNullNoise noise; noise(result, s);
        } else if (name == StringParams::hash) {
            HashNoise hashnoise;
            hashnoise(result.val(), s.val());
            result.clear_d();
        } else {
#ifndef __CUDA_ARCH__
            ((ShadingContext *)sg->context)->errorf("Unknown noise type \"%s\"", name);
#else
            // TODO: find a way to signal this error on the GPU
            result.clear_d();
#endif
        }
    }

    template<class R, class S, class T> OSL_HOSTDEVICE
    inline void operator() (StringParam name, Dual2<R> &result,
                            const Dual2<S> &s, const Dual2<T> &t,
                            ShaderGlobals *sg, const NoiseParams *opt) const {

        if (name == StringParams::uperlin || name == StringParams::noise) {
            Noise noise;
            noise(result, s, t);
        } else if (name == StringParams::perlin || name == StringParams::snoise) {
            SNoise snoise;
            snoise(result, s, t);
        } else if (name == StringParams::simplexnoise || name == StringParams::simplex) {
            SimplexNoise simplexnoise;
            simplexnoise(result, s, t);
        } else if (name == StringParams::usimplexnoise || name == StringParams::usimplex) {
            USimplexNoise usimplexnoise;
            usimplexnoise(result, s, t);
        } else if (name == StringParams::cell) {
            CellNoise cellnoise;
            cellnoise(result.val(), s.val(), t.val());
            result.clear_d();
        } else if (name == StringParams::gabor) {
            GaborNoise gnoise;
            gnoise (name, result, s, t, sg, opt);
        } else if (name == StringParams::null) {
            NullNoise noise; noise(result, s, t);
        } else if (name == StringParams::unull) {
            UNullNoise noise; noise(result, s, t);
        } else if (name == StringParams::hash) {
            HashNoise hashnoise;
            hashnoise(result.val(), s.val(), t.val());
            result.clear_d();
        } else {
#ifndef __CUDA_ARCH__
            ((ShadingContext *)sg->context)->errorf("Unknown noise type \"%s\"", name);
#else
            // TODO: find a way to signal this error on the GPU
            result.clear_d();
#endif
        }
    }
};


NOISE_IMPL_DERIV_OPT (genericnoise, GenericNoise)


struct GenericPNoise {
    OSL_HOSTDEVICE GenericPNoise () { }

    // Template on R, S, and T to be either float or Vec3

    // dual versions -- this is always called with derivs

    template<class R, class S> OSL_HOSTDEVICE
    inline void operator() (StringParam name, Dual2<R> &result, const Dual2<S> &s,
                            const S &sp,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        if (name == StringParams::uperlin || name == StringParams::noise) {
            PeriodicNoise noise;
            noise(result, s, sp);
        } else if (name == StringParams::perlin || name == StringParams::snoise) {
            PeriodicSNoise snoise;
            snoise(result, s, sp);
        } else if (name == StringParams::cell) {
            PeriodicCellNoise cellnoise;
            cellnoise(result.val(), s.val(), sp);
            result.clear_d();
        } else if (name == StringParams::gabor) {
            GaborPNoise gnoise;
            gnoise (name, result, s, sp, sg, opt);
        } else if (name == StringParams::hash) {
            PeriodicHashNoise hashnoise;
            hashnoise(result.val(), s.val(), sp);
            result.clear_d();
        } else {
#ifndef __CUDA_ARCH__
            ((ShadingContext *)sg->context)->errorf("Unknown noise type \"%s\"", name);
#else
            // TODO: find a way to signal this error on the GPU
            result.clear_d();
#endif
        }
    }

    template<class R, class S, class T> OSL_HOSTDEVICE
    inline void operator() (StringParam name, Dual2<R> &result,
                            const Dual2<S> &s, const Dual2<T> &t,
                            const S &sp, const T &tp,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        if (name == StringParams::uperlin || name == StringParams::noise) {
            PeriodicNoise noise;
            noise(result, s, t, sp, tp);
        } else if (name == StringParams::perlin || name == StringParams::snoise) {
            PeriodicSNoise snoise;
            snoise(result, s, t, sp, tp);
        } else if (name == StringParams::cell) {
            PeriodicCellNoise cellnoise;
            cellnoise(result.val(), s.val(), t.val(), sp, tp);
            result.clear_d();
        } else if (name == StringParams::gabor) {
            GaborPNoise gnoise;
            gnoise (name, result, s, t, sp, tp, sg, opt);
        } else if (name == StringParams::hash) {
            PeriodicHashNoise hashnoise;
            hashnoise(result.val(), s.val(), t.val(), sp, tp);
            result.clear_d();
        } else {
#ifndef __CUDA_ARCH__
            ((ShadingContext *)sg->context)->errorf("Unknown noise type \"%s\"", name);
#else
            // TODO: find a way to signal this error on the GPU
            result.clear_d();
#endif
        }
    }
};


PNOISE_IMPL_DERIV_OPT (genericpnoise, GenericPNoise)


// Utility: retrieve a pointer to the ShadingContext's noise params
// struct, also re-initialize its contents.
OSL_SHADEOP void *
osl_get_noise_options (void *sg_)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    RendererServices::NoiseOpt *opt = sg->context->noise_options_ptr ();
    new (opt) RendererServices::NoiseOpt;
    return opt;
}



OSL_SHADEOP OSL_HOSTDEVICE void
osl_noiseparams_set_anisotropic (void *opt, int a)
{
    ((RendererServices::NoiseOpt *)opt)->anisotropic = a;
}



OSL_SHADEOP OSL_HOSTDEVICE void
osl_noiseparams_set_do_filter (void *opt, int a)
{
    ((RendererServices::NoiseOpt *)opt)->do_filter = a;
}



OSL_SHADEOP OSL_HOSTDEVICE void
osl_noiseparams_set_direction (void *opt, void *dir)
{
    ((RendererServices::NoiseOpt *)opt)->direction = VEC(dir);
}



OSL_SHADEOP OSL_HOSTDEVICE void
osl_noiseparams_set_bandwidth (void *opt, float b)
{
    ((RendererServices::NoiseOpt *)opt)->bandwidth = b;
}



OSL_SHADEOP OSL_HOSTDEVICE void
osl_noiseparams_set_impulses (void *opt, float i)
{
    ((RendererServices::NoiseOpt *)opt)->impulses = i;
}



OSL_SHADEOP void
osl_count_noise (void *sg_)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    sg->context->shadingsys().count_noise ();
}



OSL_SHADEOP OSL_HOSTDEVICE int
osl_hash_ii (int x)
{
    return inthashi (x);
}

OSL_SHADEOP OSL_HOSTDEVICE int
osl_hash_if (float x)
{
    return inthashf (x);
}

OSL_SHADEOP OSL_HOSTDEVICE int
osl_hash_iff (float x, float y)
{
    return inthashf (x, y);
}


OSL_SHADEOP OSL_HOSTDEVICE int
osl_hash_iv (void *x)
{
    return inthashf (static_cast<float*>(x));
}


OSL_SHADEOP OSL_HOSTDEVICE int
osl_hash_ivf (void *x, float y)
{
    return inthashf (static_cast<float*>(x), y);
}


} // namespace pvt
OSL_NAMESPACE_EXIT

#endif
