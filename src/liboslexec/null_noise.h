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

#include "oslexec_pvt.h"


OSL_NAMESPACE_ENTER
namespace pvt {


struct NullNoise {
    OSL_HOSTDEVICE NullNoise () { }
    OSL_HOSTDEVICE inline void operator() (float &result, float x) const { result = 0.0f; }
    OSL_HOSTDEVICE inline void operator() (float &result, float x, float y) const { result = 0.0f; }
    OSL_HOSTDEVICE inline void operator() (float &result, const Vec3 &p) const { result = 0.0f; }
    OSL_HOSTDEVICE inline void operator() (float &result, const Vec3 &p, float t) const { result = 0.0f; }
    OSL_HOSTDEVICE inline void operator() (Vec3 &result, float x) const { result = v(); }
    OSL_HOSTDEVICE inline void operator() (Vec3 &result, float x, float y) const { result = v(); }
    OSL_HOSTDEVICE inline void operator() (Vec3 &result, const Vec3 &p) const { result = v(); }
    OSL_HOSTDEVICE inline void operator() (Vec3 &result, const Vec3 &p, float t) const { result = v(); }
    OSL_HOSTDEVICE inline void operator() (Dual2<float> &result, const Dual2<float> &x,
                                           int seed=0) const { result.set (0.0f, 0.0f, 0.0f); }
    OSL_HOSTDEVICE inline void operator() (Dual2<float> &result, const Dual2<float> &x,
                                           const Dual2<float> &y, int seed=0) const { result.set (0.0f, 0.0f, 0.0f); }
    OSL_HOSTDEVICE inline void operator() (Dual2<float> &result, const Dual2<Vec3> &p,
                                           int seed=0) const { result.set (0.0f, 0.0f, 0.0f); }
    OSL_HOSTDEVICE inline void operator() (Dual2<float> &result, const Dual2<Vec3> &p,
                                           const Dual2<float> &t, int seed=0) const { result.set (0.0f, 0.0f, 0.0f); }
    OSL_HOSTDEVICE inline void operator() (Dual2<Vec3> &result, const Dual2<float> &x) const { result.set (v(), v(), v()); }
    OSL_HOSTDEVICE inline void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y) const {  result.set (v(), v(), v()); }
    OSL_HOSTDEVICE inline void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p) const {  result.set (v(), v(), v()); }
    OSL_HOSTDEVICE inline void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const { result.set (v(), v(), v()); }
    OSL_HOSTDEVICE inline Vec3 v () const { return Vec3(0.0f, 0.0f, 0.0f); };
};

struct UNullNoise {
    OSL_HOSTDEVICE UNullNoise () { }
    OSL_HOSTDEVICE inline void operator() (float &result, float x) const { result = 0.5f; }
    OSL_HOSTDEVICE inline void operator() (float &result, float x, float y) const { result = 0.5f; }
    OSL_HOSTDEVICE inline void operator() (float &result, const Vec3 &p) const { result = 0.5f; }
    OSL_HOSTDEVICE inline void operator() (float &result, const Vec3 &p, float t) const { result = 0.5f; }
    OSL_HOSTDEVICE inline void operator() (Vec3 &result, float x) const { result = v(); }
    OSL_HOSTDEVICE inline void operator() (Vec3 &result, float x, float y) const { result = v(); }
    OSL_HOSTDEVICE inline void operator() (Vec3 &result, const Vec3 &p) const { result = v(); }
    OSL_HOSTDEVICE inline void operator() (Vec3 &result, const Vec3 &p, float t) const { result = v(); }
    OSL_HOSTDEVICE inline void operator() (Dual2<float> &result, const Dual2<float> &x,
                                           int seed=0) const { result.set (0.5f, 0.5f, 0.5f); }
    OSL_HOSTDEVICE inline void operator() (Dual2<float> &result, const Dual2<float> &x,
                                           const Dual2<float> &y, int seed=0) const { result.set (0.5f, 0.5f, 0.5f); }
    OSL_HOSTDEVICE inline void operator() (Dual2<float> &result, const Dual2<Vec3> &p,
                                           int seed=0) const { result.set (0.5f, 0.5f, 0.5f); }
    OSL_HOSTDEVICE inline void operator() (Dual2<float> &result, const Dual2<Vec3> &p,
                                           const Dual2<float> &t, int seed=0) const { result.set (0.5f, 0.5f, 0.5f); }
    OSL_HOSTDEVICE inline void operator() (Dual2<Vec3> &result, const Dual2<float> &x) const { result.set (v(), v(), v()); }
    OSL_HOSTDEVICE inline void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y) const {  result.set (v(), v(), v()); }
    OSL_HOSTDEVICE inline void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p) const {  result.set (v(), v(), v()); }
    OSL_HOSTDEVICE inline void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const { result.set (v(), v(), v()); }
    OSL_HOSTDEVICE inline Vec3 v () const { return Vec3(0.5f, 0.5f, 0.5f); };
};

} // namespace pvt
OSL_NAMESPACE_EXIT

