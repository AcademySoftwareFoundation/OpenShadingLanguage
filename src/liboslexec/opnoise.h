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



