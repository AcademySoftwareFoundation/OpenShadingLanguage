// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage


// Extensions to Imath classes for use in OSL's internals.
//
// The original Imath classes bear the "new BSD" license (same as
// ours above) and this copyright:
// Copyright (c) 2002, Industrial Light & Magic, a division of
// Lucas Digital Ltd. LLC.  All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause


#pragma once

#include <OSL/oslconfig.h>


OSL_NAMESPACE_ENTER

// Choose to treat helper functions as static
// so their symbols don't escape and possibly collide
// with other compilation units who might be compiled with
// different compiler flags (like target ISA)

/// 3x3 matrix transforming a 3-vector.  This is curiously not supplied
/// by Imath, so we define it ourselves.
static OSL_FORCEINLINE OSL_HOSTDEVICE void
multMatrix (const Matrix33 &M, const Vec3 &src,
            Vec3 &dst)
{
    // Changed all Vec3 subscripts to access data members versus array casts
    auto a = src.x * M.x[0][0] + src.y * M.x[1][0] + src.z * M.x[2][0];
    auto b = src.x * M.x[0][1] + src.y * M.x[1][1] + src.z * M.x[2][1];
    auto c = src.x * M.x[0][2] + src.y * M.x[1][2] + src.z * M.x[2][2];
    dst.x = a;
    dst.y = b;
    dst.z = c;
}

// The current Imath::Matrix44<T>::multDirMatrix uses Imath::Vec3<T>::operator[] which
// causes correctness issues due to aliasing, so we have our own version to avoid it
static OSL_FORCEINLINE OSL_HOSTDEVICE void
multDirMatrix(const Matrix44 &M, const Vec3 &src, Vec3 &dst)
{
	auto a = src.x * M.x[0][0] + src.y * M.x[1][0] + src.z * M.x[2][0];
	auto b = src.x * M.x[0][1] + src.y * M.x[1][1] + src.z * M.x[2][1];
	auto c = src.x * M.x[0][2] + src.y * M.x[1][2] + src.z * M.x[2][2];

    dst.x = a;
    dst.y = b;
    dst.z = c;
}

//
// Inlinable version to enable vectorization
// Better results with return by value (versus taking reference parameter)
static OSL_FORCEINLINE OSL_HOSTDEVICE Vec3
multiplyDirByMatrix(const Matrix44 &M, const Vec3 &src)
{
	auto a = src.x * M.x[0][0] + src.y * M.x[1][0] + src.z * M.x[2][0];
	auto b = src.x * M.x[0][1] + src.y * M.x[1][1] + src.z * M.x[2][1];
	auto c = src.x * M.x[0][2] + src.y * M.x[1][2] + src.z * M.x[2][2];

    return Vec3(a,b,c);
}


/// Express dot product as a function rather than a method.
static OSL_FORCEINLINE OSL_HOSTDEVICE typename Vec2::BaseType
dot (const Vec2 &a, const Vec2 &b)
{
    return a.dot (b);
}


/// Express dot product as a function rather than a method.
static OSL_FORCEINLINE OSL_HOSTDEVICE typename Vec2::BaseType
dot (const Vec3 &a, const Vec3 &b)
{
    return a.dot (b);
}



/// Return the determinant of a 2x2 matrix.
static OSL_FORCEINLINE OSL_HOSTDEVICE typename Matrix22::BaseType
determinant (const Matrix22 &M)
{
    return M.x[0][0]*M.x[1][1] - M.x[0][1]*M.x[1][0];
}


static OSL_FORCEINLINE OSL_HOSTDEVICE void
makeIdentity(Matrix44 &m)
{
	using ScalarT = typename Matrix44::BaseType;
    // better allow SROA optimizations
    // by avoiding memset
    m.x[0][0] = ScalarT(1);
    m.x[0][1] = ScalarT(0);
    m.x[0][2] = ScalarT(0);
    m.x[0][3] = ScalarT(0);

    m.x[1][0] = ScalarT(0);
    m.x[1][1] = ScalarT(1);
    m.x[1][2] = ScalarT(0);
    m.x[1][3] = ScalarT(0);

    m.x[2][0] = ScalarT(0);
    m.x[2][1] = ScalarT(0);
    m.x[2][2] = ScalarT(1);
    m.x[2][3] = ScalarT(0);

    m.x[3][0] = ScalarT(0);
    m.x[3][1] = ScalarT(0);
    m.x[3][2] = ScalarT(0);
    m.x[3][3] = ScalarT(1);
}


// Partition general purpose inverse of Matrix44 into:
// test_if_affine
// affineInverse - fast path that is SIMD friendly
// nonAffineInverse - slow path to be used outside SIMD loop to
//                    handle any non-affine matrices
static OSL_FORCEINLINE bool test_if_affine(const Matrix44 & m) {
	using ScalarT = typename Matrix44::BaseType;
    return (m.x[0][3] == ScalarT(0)) &
           (m.x[1][3] == ScalarT(0)) &
           (m.x[2][3] == ScalarT(0)) &
           (m.x[3][3] == ScalarT(1));
}

static OSL_FORCEINLINE OSL_HOSTDEVICE Matrix44
affineInverse(const Matrix44 &m)
{
	using ScalarT = typename Matrix44::BaseType;
    // As we may speculatively call on non-affine matrices and just not use the result
    // we shouldn't bother verifying m is affine, just assume it and let caller non
    // use the results
    Matrix44 s (m.x[1][1] * m.x[2][2] - m.x[2][1] * m.x[1][2],
                m.x[2][1] * m.x[0][2] - m.x[0][1] * m.x[2][2],
                m.x[0][1] * m.x[1][2] - m.x[1][1] * m.x[0][2],
				ScalarT(0),

                m.x[2][0] * m.x[1][2] - m.x[1][0] * m.x[2][2],
                m.x[0][0] * m.x[2][2] - m.x[2][0] * m.x[0][2],
                m.x[1][0] * m.x[0][2] - m.x[0][0] * m.x[1][2],
				ScalarT(0),

                m.x[1][0] * m.x[2][1] - m.x[2][0] * m.x[1][1],
                m.x[2][0] * m.x[0][1] - m.x[0][0] * m.x[2][1],
                m.x[0][0] * m.x[1][1] - m.x[1][0] * m.x[0][1],
				ScalarT(0),

                ScalarT(0),
                ScalarT(0),
                ScalarT(0),
				ScalarT(1));

    auto r = m.x[0][0] * s.x[0][0] + m.x[0][1] * s.x[1][0] + m.x[0][2] * s.x[2][0];
    auto abs_r = IMATH_INTERNAL_NAMESPACE::abs (r);

    int may_have_divided_by_zero = 0;
    if (OSL_UNLIKELY(abs_r < ScalarT(1)))
    {
    	auto mr = abs_r / Imath::limits<ScalarT>::smallest();
#if 0
        OSL_PRAGMA(unroll)
        for (int i = 0; i < 3; ++i)
        {
            OSL_PRAGMA(unroll)
            for (int j = 0; j < 3; ++j)
            {
                if (mr <= IMATH_INTERNAL_NAMESPACE::abs (s.x[i][j]))
                {
                    may_have_divided_by_zero = 1;
                }
            }
        }
#else
        // NOTE: using bitwise OR to avoid C++ semantics that cannot evaluate
        // the right hand side of logical OR unless left hand side is false
        if (
            (mr <= IMATH_INTERNAL_NAMESPACE::abs (s.x[0][0])) |
            (mr <= IMATH_INTERNAL_NAMESPACE::abs (s.x[0][1])) |
            (mr <= IMATH_INTERNAL_NAMESPACE::abs (s.x[0][2])) |
            (mr <= IMATH_INTERNAL_NAMESPACE::abs (s.x[1][0])) |
            (mr <= IMATH_INTERNAL_NAMESPACE::abs (s.x[1][1])) |
            (mr <= IMATH_INTERNAL_NAMESPACE::abs (s.x[1][2])) |
            (mr <= IMATH_INTERNAL_NAMESPACE::abs (s.x[2][0])) |
            (mr <= IMATH_INTERNAL_NAMESPACE::abs (s.x[2][1])) |
            (mr <= IMATH_INTERNAL_NAMESPACE::abs (s.x[2][2]))
            ) {
            may_have_divided_by_zero = 1;
        }
#endif
    }

#if 0
    OSL_PRAGMA(unroll)
    for (int i = 0; i < 3; ++i)
    {
        OSL_PRAGMA(unroll)
        for (int j = 0; j < 3; ++j)
        {
            s.x[i][j] /= r;
        }
    }
#else
    // Just unroll by hand, about the same size as loop code
    // NOTE: opportunity to use reciprocal multiply here,
    // but that would affect the results, although compilers
    // might do that anyway based on optimization settings
    s.x[0][0] /= r;
    s.x[0][1] /= r;
    s.x[0][2] /= r;
    s.x[1][0] /= r;
    s.x[1][1] /= r;
    s.x[1][2] /= r;
    s.x[2][0] /= r;
    s.x[2][1] /= r;
    s.x[2][2] /= r;
#endif

    s.x[3][0] = -m.x[3][0] * s.x[0][0] - m.x[3][1] * s.x[1][0] - m.x[3][2] * s.x[2][0];
    s.x[3][1] = -m.x[3][0] * s.x[0][1] - m.x[3][1] * s.x[1][1] - m.x[3][2] * s.x[2][1];
    s.x[3][2] = -m.x[3][0] * s.x[0][2] - m.x[3][1] * s.x[1][2] - m.x[3][2] * s.x[2][2];

    if (OSL_UNLIKELY(may_have_divided_by_zero == 1))
    {
        makeIdentity(s);
    }
    return s;
}


// Avoid potential aliasing issues with OIIO implementation,
// but main purpose is to turn off FMA which can affect
// rounding and order of operations which can affect results
// with near 0 divisor.  Really only an issue when trying to
// match results between LLVM IR based implementation and
// compiler optimized versions which may have optimized
// differently than the LLVM IR version.
// Clang doesn't currently have a way to control fast-math/fp:strict
// at a per function level, so we will resort to disabling optimization
// for this function.  Not ideal, but this is already a slow path
// for exceptional situations.
// GCC could use __attribute__((optimize("-fno-fast-math")))
// NOTE:  only using "inline" to get ODR (One Definition Rule) behavior
static inline OSL_HOSTDEVICE Matrix44
nonAffineInverse(const Matrix44 &source) OSL_CLANG_ATTRIBUTE(optnone)
{
    OSL_INTEL_PRAGMA(float_control(strict,on,push))

	using ScalarT = typename Matrix44::BaseType;
    Matrix44 t(source);
    Matrix44 s;

    // Forward elimination

    for (int i = 0; i < 3 ; i++)
    {
        int pivot = i;

        ScalarT pivotsize = t.x[i][i];

        if (pivotsize < 0)
            pivotsize = -pivotsize;

        for (int j = i + 1; j < 4; j++)
        {
            ScalarT tmp = t.x[j][i];

            if (tmp < 0)
                tmp = -tmp;

            if (tmp > pivotsize)
            {
                pivot = j;
                pivotsize = tmp;
            }
        }

        if (pivotsize == 0)
        {
            return Matrix44();
        }

        if (pivot != i)
        {
            for (int j = 0; j < 4; j++)
            {
                ScalarT tmp;

                tmp = t.x[i][j];
                t.x[i][j] = t.x[pivot][j];
                t.x[pivot][j] = tmp;

                tmp = s.x[i][j];
                s.x[i][j] = s.x[pivot][j];
                s.x[pivot][j] = tmp;
            }
        }

        for (int j = i + 1; j < 4; j++)
        {
            ScalarT f = t.x[j][i] / t.x[i][i];

            for (int k = 0; k < 4; k++)
            {
                t.x[j][k] -= f * t.x[i][k];
                s.x[j][k] -= f * s.x[i][k];
            }
        }
    }

    // Backward substitution

    for (int i = 3; i >= 0; --i)
    {
        ScalarT f;

        if ((f = t.x[i][i]) == 0)
        {
            return Matrix44();
        }

        for (int j = 0; j < 4; j++)
        {
            t.x[i][j] /= f;
            s.x[i][j] /= f;
        }

        for (int j = 0; j < i; j++)
        {
            f = t.x[j][i];

            for (int k = 0; k < 4; k++)
            {
                t.x[j][k] -= f * t.x[i][k];
                s.x[j][k] -= f * s.x[i][k];
            }
        }
    }

    return s;
}


// In order to have inlinable Matrix44*float
// Override with a more specific version than
// template <class T>
// inline Matrix44<T>
// operator * (T a, const Matrix44<T> &v);

static OSL_FORCEINLINE OSL_HOSTDEVICE Matrix44
operator * (typename Matrix44::BaseType a, const Matrix44 &v)
{
    return Matrix44 (v.x[0][0] * a,
                     v.x[0][1] * a,
                     v.x[0][2] * a,
                     v.x[0][3] * a,
                     v.x[1][0] * a,
                     v.x[1][1] * a,
                     v.x[1][2] * a,
                     v.x[1][3] * a,
                     v.x[2][0] * a,
                     v.x[2][1] * a,
                     v.x[2][2] * a,
                     v.x[2][3] * a,
                     v.x[3][0] * a,
                     v.x[3][1] * a,
                     v.x[3][2] * a,
                     v.x[3][3] * a);
}

static OSL_FORCEINLINE OSL_HOSTDEVICE Matrix44
inlinedTransposed (const Matrix44 &m)
{
    return Matrix44 (m.x[0][0],
                     m.x[1][0],
                     m.x[2][0],
                     m.x[3][0],
                     m.x[0][1],
                     m.x[1][1],
                     m.x[2][1],
                     m.x[3][1],
                     m.x[0][2],
                     m.x[1][2],
                     m.x[2][2],
                     m.x[3][2],
                     m.x[0][3],
                     m.x[1][3],
                     m.x[2][3],
                     m.x[3][3]);
}

// Inlinable version to enable vectorization
// Better results with return by value (versus taking reference parameter)
static OSL_FORCEINLINE OSL_HOSTDEVICE Matrix44
multiplyMatrixByMatrix (const Matrix44 &a,
                       const Matrix44 &b)
{
    const auto a00 = a.x[0][0];
    const auto a01 = a.x[0][1];
    const auto a02 = a.x[0][2];
    const auto a03 = a.x[0][3];

    const auto c00  = a00 * b.x[0][0]  + a01 * b.x[1][0]  + a02 * b.x[2][0]  + a03 * b.x[3][0];
    const auto c01  = a00 * b.x[0][1]  + a01 * b.x[1][1]  + a02 * b.x[2][1]  + a03 * b.x[3][1];
    const auto c02  = a00 * b.x[0][2]  + a01 * b.x[1][2]  + a02 * b.x[2][2] + a03 * b.x[3][2];
    const auto c03  = a00 * b.x[0][3]  + a01 * b.x[1][3]  + a02 * b.x[2][3] + a03 * b.x[3][3];

    const auto a10 = a.x[1][0];
    const auto a11 = a.x[1][1];
    const auto a12 = a.x[1][2];
    const auto a13 = a.x[1][3];

    const auto c10  = a10 * b.x[0][0]  + a11 * b.x[1][0]  + a12 * b.x[2][0]  + a13 * b.x[3][0];
    const auto c11  = a10 * b.x[0][1]  + a11 * b.x[1][1]  + a12 * b.x[2][1]  + a13 * b.x[3][1];
    const auto c12  = a10 * b.x[0][2]  + a11 * b.x[1][2]  + a12 * b.x[2][2] + a13 * b.x[3][2];
    const auto c13  = a10 * b.x[0][3]  + a11 * b.x[1][3]  + a12 * b.x[2][3] + a13 * b.x[3][3];

    const auto a20 = a.x[2][0];
    const auto a21 = a.x[2][1];
    const auto a22 = a.x[2][2];
    const auto a23 = a.x[2][3];

    const auto c20  = a20 * b.x[0][0]  + a21 * b.x[1][0]  + a22 * b.x[2][0]  + a23 * b.x[3][0];
    const auto c21  = a20 * b.x[0][1]  + a21 * b.x[1][1]  + a22 * b.x[2][1]  + a23 * b.x[3][1];
    const auto c22 = a20 * b.x[0][2]  + a21 * b.x[1][2]  + a22 * b.x[2][2] + a23 * b.x[3][2];
    const auto c23 = a20 * b.x[0][3]  + a21 * b.x[1][3]  + a22 * b.x[2][3] + a23 * b.x[3][3];

    const auto a30 = a.x[3][0];
    const auto a31 = a.x[3][1];
    const auto a32 = a.x[3][2];
    const auto a33 = a.x[3][3];

    const auto c30 = a30 * b.x[0][0]  + a31 * b.x[1][0]  + a32 * b.x[2][0]  + a33 * b.x[3][0];
    const auto c31 = a30 * b.x[0][1]  + a31 * b.x[1][1]  + a32 * b.x[2][1]  + a33 * b.x[3][1];
    const auto c32 = a30 * b.x[0][2]  + a31 * b.x[1][2]  + a32 * b.x[2][2] + a33 * b.x[3][2];
    const auto c33 = a30 * b.x[0][3]  + a31 * b.x[1][3]  + a32 * b.x[2][3] + a33 * b.x[3][3];
    return Matrix44(
            c00, c01, c02, c03,
            c10, c11, c12, c13,
            c20, c21, c22, c23,
            c30, c31, c32, c33
        );

}

OSL_NAMESPACE_EXIT
