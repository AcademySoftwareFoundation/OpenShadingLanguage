// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage
// Contributions Copyright (c) 2017 Intel Inc., et al.

// clang-format off

#pragma once

#include <cmath>
#include <limits>

#include "dual.h"
#include "dual_vec.h"
#include <OSL/Imathx/Imathx.h>

#include <OpenImageIO/fmath.h>


OSL_NAMESPACE_BEGIN

#ifdef __OSL_WIDE_PVT
    namespace __OSL_WIDE_PVT {
#else
    namespace pvt {
#endif



// SIMD FRIENDLY MATH
// Scalar code meant to be used from inside
// compiler vectorized SIMD loops.
// No intrinsics or assembly, just vanilla C++
namespace sfm
{

    // Math code derived from OpenEXR/ImathMatrix.h
    // including it's copyrights in the namespace
    /*
       Copyright (c) 2002-2012, Industrial Light & Magic, a division of Lucas
       Digital Ltd. LLC

       All rights reserved.

       Redistribution and use in source and binary forms, with or without
       modification, are permitted provided that the following conditions are
       met:
       *       Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
       *       Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following disclaimer
       in the documentation and/or other materials provided with the
       distribution.
       *       Neither the name of Industrial Light & Magic nor the names of
       its contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

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

#if OSL_INTEL_CLASSIC_COMPILER_VERSION
    // std::isinf wasn't vectorizing and was branchy. This slightly
    // perturbed version fairs better and is branch free when vectorized
    // with the Intel compiler.
    OSL_FORCEINLINE OSL_HOSTDEVICE int isinf (float x) {
        int r = 0;
        // NOTE: using bitwise | to avoid branches
        if (!(std::isfinite(x)|std::isnan(x))) {
            r = static_cast<int>(copysignf(1.0f,x));
        }
        return r;
    }
#else
    // Other compilers don't seem to vectorize well no matter what, so just
    // use the standard version.
    using std::isinf;
#endif

    template<typename T>
    OSL_FORCEINLINE OSL_HOSTDEVICE T
    negate(const T &x) {
        #if OSL_FAST_MATH
            // Compiler using a constant bit mask to perform negation,
            // and reading a constant involves accessing its memory location.
            // Alternatively the compiler can create a 0 value in register
            // in a constant time not involving the memory subsystem.
            // So we can subtract from 0 to effectively negate a value.
            // Handling of +0.0f and -0.0f might differ from IEE here.
            // But in graphics practice we don't see a problem with codes
            // using this approach and a measurable 10%(+|-5%) performance gain
            return T(0) - x;
        #else
            return -x;
        #endif
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE Dual2<float>
    absf (const Dual2<float> &x)
    {
        // Avoid ternary ops whose operands have side effects
        // in favor of code that executes both sides masked
        // return x.val() >= 0.0f ? x : -x;

        // NOTE: negation happens outside of conditional, then is blended based on the condition
        Dual2<float> neg_x = OIIO::fast_neg(x);

        bool cond = x.val() < 0.0f;
        // Blend per builtin component to allow
        // the compiler to track builtins and privatize the data layout
        // versus requiring a stack location.
        float val = x.val();
        if (cond) {
            val = neg_x.val();
        }

        float dx = x.dx();
        if (cond) {
            dx = neg_x.dx();
        }

        float dy = x.dy();
        if (cond) {
            dy = neg_x.dy();
        }

        return Dual2<float>(val, dx, dy);
    }


    /// Round to nearest integer, returning as an int.
    OSL_FORCEINLINE OSL_HOSTDEVICE int fast_rint (float x) {
        // used by sin/cos/tan range reduction
    #if 0
        // single roundps instruction on SSE4.1+ (for gcc/clang at least)
        //return static_cast<int>(rintf(x));
        return rintf(x);
    #else
        // emulate rounding by adding/subtracting 0.5
        return static_cast<int>(x + copysignf(0.5f, x));

        // Other possible factorings
        //return (x >= 0.0f) ? static_cast<int>(x + 0.5f) : static_cast<int>(x - 0.5f);
        //return static_cast<int>(x +  (x >= 0.0f) ? 0.5f : - 0.5f);
        //float pad = (x >= 0.0f) ? 0.5f : - 0.5f;
        //return static_cast<int>(x + pad);
        //return nearbyint(x);
#endif
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE
    float length(const Vec3 &N)
    {
        return N.length();
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE Vec3
    normalize(const Vec3 &N)
    {
        return N.normalized();
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE Dual2<Vec3>
    normalize (const Dual2<Vec3> &a)
    {
        // NOTE: using bitwise & to avoid branches
        if (OSL_UNLIKELY((a.val().x == 0.0f) & (a.val().y == 0.0f) & (a.val().z == 0.0f))) {
            return Dual2<Vec3> (Vec3(0.0f, 0.0f, 0.0f),
                                Vec3(0.0f, 0.0f, 0.0f),
                                Vec3(0.0f, 0.0f, 0.0f));
        } else {
            Dual2<float> ax (a.val().x, a.dx().x, a.dy().x);
            Dual2<float> ay (a.val().y, a.dx().y, a.dy().y);
            Dual2<float> az (a.val().z, a.dx().z, a.dy().z);
            Dual2<float> inv_length = 1.0f / sqrt(ax*ax + ay*ay + az*az);
            ax = ax*inv_length;
            ay = ay*inv_length;
            az = az*inv_length;
            return Dual2<Vec3> (Vec3(ax.val(), ay.val(), az.val()),
                                Vec3(ax.dx(),  ay.dx(),  az.dx() ),
                                Vec3(ax.dy(),  ay.dy(),  az.dy() ));
        }
    }

#if OSL_ANY_CLANG && !OSL_INTEL_CLASSIC_COMPILER_VERSION && !OSL_INTEL_LLVM_COMPILER_VERSION

    // To make clang's loop vectorizor happy
    // we need to make sure result of min and max
    // is truly by value, not address or reference
    // to the original.
    // This required creating temporary result
    // versus just returning by value (which could have been elided)
    template<typename T>
    OSL_FORCEINLINE OSL_HOSTDEVICE
    T min_val(const T &left, const T &right)
    {
        T result(right);
        if (right > left)
            result = left;
        return result;
    }

    template<typename T>
    OSL_FORCEINLINE OSL_HOSTDEVICE
    T max_val(const T &left, const T &right)
    {
        T result(left);
        if (right > left)
            result = right;
        return result;
    }

    template<typename T>
    OSL_FORCEINLINE OSL_HOSTDEVICE
    T select_val(bool cond, const T &left, const T &right)
    {
        T result(right);
        if (cond)
            result = left;
        return result;
    }
#else

    template<typename T>
    OSL_FORCEINLINE OSL_HOSTDEVICE
    T min_val(T left, T right)
    {
        return (right > left)? left : right;
    }

    template<typename T>
    OSL_FORCEINLINE OSL_HOSTDEVICE
    T max_val(T left, T right)
    {
        return (right > left)? right : left;
    }

    template<typename T>
    OSL_FORCEINLINE OSL_HOSTDEVICE
    T select_val(bool cond, const T left, const T right)
    {
        if (cond)
            return left;
        else
            return right;
    }

#endif

    using Matrix33 = OSL::Matrix33;

    OSL_FORCEINLINE OSL_HOSTDEVICE sfm::Matrix33
    make_matrix33_cols (const Vec3 &a, const Vec3 &b, const Vec3 &c)
    {
        return sfm::Matrix33 (a.x, b.x, c.x,
                         a.y, b.y, c.y,
                         a.z, b.z, c.z);
    }



    // Considering having functionally equivalent versions of Vec3, Color3, Matrix44
    // with slight modifications to inlining and implementation to avoid aliasing and
    // improve likelihood of proper privation of local variables within a SIMD loop

}  // namespace sfm

}  // namespace __OSL_WIDE_PVT or pvt



OSL_NAMESPACE_END
