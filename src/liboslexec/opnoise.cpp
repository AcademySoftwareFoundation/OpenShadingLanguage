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
OSL_SHADEOP float osl_ ##opname## _fv (void *x) {                       \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, VEC(x));                                                   \
    return r;                                                           \
}                                                                       \
                                                                        \
OSL_SHADEOP float osl_ ##opname## _fvf (void *x, float y) {             \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, VEC(x), y);                                                \
    return r;                                                           \
}                                                                       \
                                                                        \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _vf (void *r, float x) {               \
    implname impl;                                                      \
    impl (VEC(r), x);                                                   \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _vff (void *r, float x, float y) {     \
    implname impl;                                                      \
    impl (VEC(r), x, y);                                                \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _vv (void *r, void *x) {               \
    implname impl;                                                      \
    impl (VEC(r), VEC(x));                                              \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _vvf (void *r, void *x, float y) {     \
    implname impl;                                                      \
    impl (VEC(r), VEC(x), y);                                           \
}





#define NOISE_IMPL_DERIV(opname,implname)                               \
OSL_SHADEOP void osl_ ##opname## _dfdf (void *r, void *x) {             \
    implname impl;                                                      \
    impl (DFLOAT(r), DFLOAT(x));                                        \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdfdf (void *r, void *x, void *y) {  \
    implname impl;                                                      \
    impl (DFLOAT(r), DFLOAT(x), DFLOAT(y));                             \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdff (void *r, void *x, float y) {   \
    implname impl;                                                      \
    impl (DFLOAT(r), DFLOAT(x), Dual2<float>(y));                       \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dffdf (void *r, float x, void *y) {   \
    implname impl;                                                      \
    impl (DFLOAT(r), Dual2<float>(x), DFLOAT(y));                       \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdv (void *r, void *x) {             \
    implname impl;                                                      \
    impl (DFLOAT(r), DVEC(x));                                          \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdvdf (void *r, void *x, void *y) {  \
    implname impl;                                                      \
    impl (DFLOAT(r), DVEC(x), DFLOAT(y));                               \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdvf (void *r, void *x, float y) {   \
    implname impl;                                                      \
    impl (DFLOAT(r), DVEC(x), Dual2<float>(y));                         \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfvdf (void *r, void *x, void *y) {   \
    implname impl;                                                      \
    impl (DFLOAT(r), Dual2<Vec3>(VEC(x)), DFLOAT(y));                   \
}                                                                       \
                                                                        \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdf (void *r, void *x) {             \
    implname impl;                                                      \
    impl (DVEC(r), DFLOAT(x));                                          \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdfdf (void *r, void *x, void *y) {  \
    implname impl;                                                      \
    impl (DVEC(r), DFLOAT(x), DFLOAT(y));                               \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdff (void *r, void *x, float y) {   \
    implname impl;                                                      \
    impl (DVEC(r), DFLOAT(x), Dual2<float>(y));                         \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvfdf (void *r, float x, void *y) {   \
    implname impl;                                                      \
    impl (DVEC(r), Dual2<float>(x), DFLOAT(y));                         \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdv (void *r, void *x) {             \
    implname impl;                                                      \
    impl (DVEC(r), DVEC(x));                                            \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdvdf (void *r, void *x, void *y) {  \
    implname impl;                                                      \
    impl (DVEC(r), DVEC(x), DFLOAT(y));                                 \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdvf (void *r, void *x, float y) {   \
    implname impl;                                                      \
    impl (DVEC(r), DVEC(x), Dual2<float>(y));                           \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvvdf (void *r, void *x, void *y) {   \
    implname impl;                                                      \
    impl (DVEC(r), Dual2<Vec3>(VEC(x)), DFLOAT(y));                     \
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
OSL_SHADEOP float osl_ ##opname## _fvv (void *x, void *px) {            \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, VEC(x), VEC(px));                                          \
    return r;                                                           \
}                                                                       \
                                                                        \
OSL_SHADEOP float osl_ ##opname## _fvfvf (void *x, float y, void *px, float py) { \
    implname impl;                                                      \
    float r;                                                            \
    impl (r, VEC(x), y, VEC(px), py);                                   \
    return r;                                                           \
}                                                                       \
                                                                        \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _vff (void *r, float x, float px) {    \
    implname impl;                                                      \
    impl (VEC(r), x, px);                                               \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _vffff (void *r, float x, float y, float px, float py) { \
    implname impl;                                                      \
    impl (VEC(r), x, y, px, py);                                        \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _vvv (void *r, void *x, void *px) {    \
    implname impl;                                                      \
    impl (VEC(r), VEC(x), VEC(px));                                     \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _vvfvf (void *r, void *x, float y, void *px, float py) { \
    implname impl;                                                      \
    impl (VEC(r), VEC(x), y, VEC(px), py);                              \
}





#define PNOISE_IMPL_DERIV(opname,implname)                              \
OSL_SHADEOP void osl_ ##opname## _dfdff (void *r, void *x, float px) {  \
    implname impl;                                                      \
    impl (DFLOAT(r), DFLOAT(x), px);                                    \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdfdfff (void *r, void *x, void *y, float px, float py) { \
    implname impl;                                                      \
    impl (DFLOAT(r), DFLOAT(x), DFLOAT(y), px, py);                     \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdffff (void *r, void *x, float y, float px, float py) { \
    implname impl;                                                      \
    impl (DFLOAT(r), DFLOAT(x), Dual2<float>(y), px, py);               \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dffdfff (void *r, float x, void *y, float px, float py) { \
    implname impl;                                                      \
    impl (DFLOAT(r), Dual2<float>(x), DFLOAT(y), px, py);               \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdvv (void *r, void *x, void *px) {  \
    implname impl;                                                      \
    impl (DFLOAT(r), DVEC(x), VEC(px));                                 \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdvdfvf (void *r, void *x, void *y, void *px, float py) { \
    implname impl;                                                      \
    impl (DFLOAT(r), DVEC(x), DFLOAT(y), VEC(px), py);                  \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfdvfvf (void *r, void *x, float y, void *px, float py) { \
    implname impl;                                                      \
    impl (DFLOAT(r), DVEC(x), Dual2<float>(y), VEC(px), py);            \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dfvdfvf (void *r, void *x, void *y, void *px, float py) { \
    implname impl;                                                      \
    impl (DFLOAT(r), Dual2<Vec3>(VEC(x)), DFLOAT(y), VEC(px), py);      \
}                                                                       \
                                                                        \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdff (void *r, void *x, float px) {  \
    implname impl;                                                      \
    impl (DVEC(r), DFLOAT(x), px);                                      \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdfdfff (void *r, void *x, void *y, float px, float py) { \
    implname impl;                                                      \
    impl (DVEC(r), DFLOAT(x), DFLOAT(y), px, py);                       \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdffff (void *r, void *x, float y, float px, float py) { \
    implname impl;                                                      \
    impl (DVEC(r), DFLOAT(x), Dual2<float>(y), px, py);                 \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvfdfff (void *r, float x, void *y, float px, float py) { \
    implname impl;                                                      \
    impl (DVEC(r), Dual2<float>(x), DFLOAT(y), px, py);                 \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdvv (void *r, void *x, void *px) {  \
    implname impl;                                                      \
    impl (DVEC(r), DVEC(x), VEC(px));                                   \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdvdfvf (void *r, void *x, void *y, void *px, float py) { \
    implname impl;                                                      \
    impl (DVEC(r), DVEC(x), DFLOAT(y), VEC(px), py);                    \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvdvfvf (void *r, void *x, float y, float *px, float py) { \
    implname impl;                                                      \
    impl (DVEC(r), DVEC(x), Dual2<float>(y), VEC(px), py);              \
}                                                                       \
                                                                        \
OSL_SHADEOP void osl_ ##opname## _dvvdfvf (void *r, void *x, void *px, void *y, float py) { \
    implname impl;                                                      \
    impl (DVEC(r), Dual2<Vec3>(VEC(x)), DFLOAT(y), VEC(px), py);        \
}


PNOISE_IMPL (pnoise, PeriodicNoise)
PNOISE_IMPL_DERIV (pnoise, PeriodicNoise)
PNOISE_IMPL (psnoise, PeriodicSNoise)
PNOISE_IMPL_DERIV (psnoise, PeriodicSNoise)

#endif
