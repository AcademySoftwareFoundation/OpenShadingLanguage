// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// clang-format off

#pragma once

#include <limits>
#include <type_traits>

#include <OpenImageIO/fmath.h>
#include <OpenImageIO/hash.h>
#include <OpenImageIO/simd.h>

#include <OSL/dual.h>
#include <OSL/dual_vec.h>
#include <OSL/sfm_simplex.h>
#include <OSL/sfmath.h>


OSL_NAMESPACE_ENTER


namespace oslnoise {

/////////////////////////////////////////////////////////////////////////
//
// Simple public API for computing the same noise functions that you get
// from OSL shaders.
//
// These match the properties of OSL noises that have the same names,
// please see the OSL specification for more detailed descriptions, we
// won't recapitulate it here.
//
// For the sake of compactness, we express the noise functions as
// templates for a either one or two domain parameters, which may be:
//     (float)                 // 1-D domain noise, 1-argument variety
//     (float, float)          // 2-D domain noise, 2-argument variety
//     (const Vec3 &)          // 3-D domain noise, 1-argument variety
//     (const Vec3 &, float)   // 4-D domain noise, 2-argument variety
// And the range type may be
//     float noisename ()      // float-valued noise
//     Vec3  vnoisename()      // vector-valued noise
// Note that in OSL we can overload function calls by return type, but we
// can't in C++, so we prepend a "v" in front of the names of functions
// that return vector-valued noise.
//
/////////////////////////////////////////////////////////////////////////

// Signed Perlin-like noise on 1-4 dimensional domain, range [-1,1].
template <typename S >             OSL_HOSTDEVICE float snoise (S x);
template <typename S, typename T>  OSL_HOSTDEVICE float snoise (S x, T y);
template <typename S >             OSL_HOSTDEVICE Vec3  vsnoise (S x);
template <typename S, typename T>  OSL_HOSTDEVICE Vec3  vsnoise (S x, T y);

// Unsigned Perlin-like noise on 1-4 dimensional domain, range [0,1].
template <typename S >             OSL_HOSTDEVICE float noise (S x);
template <typename S, typename T>  OSL_HOSTDEVICE float noise (S x, T y);
template <typename S >             OSL_HOSTDEVICE Vec3  vnoise (S x);
template <typename S, typename T>  OSL_HOSTDEVICE Vec3  vnoise (S x, T y);

// Cell noise on 1-4 dimensional domain, range [0,1].
// cellnoise is constant within each unit cube (cell) on the domain, but
// discontinuous at integer boundaries (and uncorrelated from cell to
// cell).
template <typename S >             OSL_HOSTDEVICE float cellnoise (S x);
template <typename S, typename T>  OSL_HOSTDEVICE float cellnoise (S x, T y);
template <typename S >             OSL_HOSTDEVICE Vec3  vcellnoise (S x);
template <typename S, typename T>  OSL_HOSTDEVICE Vec3  vcellnoise (S x, T y);

// Hash noise on 1-4 dimensional domain, range [0,1].
// hashnoise is like cellnoise, but without the 'floor' -- in other words,
// it's an uncorrelated hash that is different for every floating point
// value.
template <typename S >             OSL_HOSTDEVICE float hashnoise (S x);
template <typename S, typename T>  OSL_HOSTDEVICE float hashnoise (S x, T y);
template <typename S >             OSL_HOSTDEVICE Vec3  vhashnoise (S x);
template <typename S, typename T>  OSL_HOSTDEVICE Vec3  vhashnoise (S x, T y);

// FIXME -- eventually consider adding to the public API:
//  * periodic varieties
//  * varieties with derivatives
//  * varieties that take/return simd::float3 rather than Imath::Vec3f.
//  * exposing the simplex & gabor varieties


}   // namespace oslnoise



///////////////////////////////////////////////////////////////////////
// Implementation follows...
//
// Users don't need to worry about this part
///////////////////////////////////////////////////////////////////////

struct NoiseParams;

namespace pvt {
using namespace OIIO::simd;
using namespace OIIO::bjhash;
#ifdef __OSL_WIDE_PVT
    namespace sfm = OSL::__OSL_WIDE_PVT::sfm;
#endif
typedef void (*NoiseGenericFunc)(int outdim, float *out, bool derivs,
                                 int indim, const float *in,
                                 const float *period, NoiseParams *params);
typedef void (*NoiseImplFunc)(float *out, const float *in,
                              const float *period, NoiseParams *params);

OSLNOISEPUBLIC OSL_HOSTDEVICE
float simplexnoise1 (float x, int seed=0, float *dnoise_dx=NULL);

OSLNOISEPUBLIC OSL_HOSTDEVICE
float simplexnoise2 (float x, float y, int seed=0,
                     float *dnoise_dx=NULL, float *dnoise_dy=NULL);

OSLNOISEPUBLIC OSL_HOSTDEVICE
float simplexnoise3 (float x, float y, float z, int seed=0,
                     float *dnoise_dx=NULL, float *dnoise_dy=NULL,
                     float *dnoise_dz=NULL);

OSLNOISEPUBLIC OSL_HOSTDEVICE
float simplexnoise4 (float x, float y, float z, float w, int seed=0,
                     float *dnoise_dx=NULL, float *dnoise_dy=NULL,
                     float *dnoise_dz=NULL, float *dnoise_dw=NULL);


namespace {

// convert a 32 bit integer into a floating point number in [0,1]
inline OSL_HOSTDEVICE float bits_to_01 (unsigned int bits) {
    // divide by 2^32-1
	// Calculate inverse constant with double precision to avoid
	//     warning: implicit conversion from 'unsigned int' to 'float' changes value from 4294967295 to 4294967296
    constexpr float convertFactor = static_cast<float>(static_cast<double>(1.0) / static_cast<double>(std::numeric_limits<unsigned int>::max()));
    return bits * convertFactor;
}


#ifndef __CUDA_ARCH__
// Perform a bjmix (see OpenImageIO/hash.h) on 4 sets of values at once.
OSL_FORCEINLINE void
bjmix (int4 &a, int4 &b, int4 &c)
{
    using OIIO::simd::rotl32;
    a -= c;  a ^= rotl32(c, 4);  c += b;
    b -= a;  b ^= rotl32(a, 6);  a += c;
    c -= b;  c ^= rotl32(b, 8);  b += a;
    a -= c;  a ^= rotl32(c,16);  c += b;
    b -= a;  b ^= rotl32(a,19);  a += c;
    c -= b;  c ^= rotl32(b, 4);  b += a;
}

// Perform a bjfinal (see OpenImageIO/hash.h) on 4 sets of values at once.
OSL_FORCEINLINE int4
bjfinal (const int4& a_, const int4& b_, const int4& c_)
{
    using OIIO::simd::rotl32;
    int4 a(a_), b(b_), c(c_);
    c ^= b; c -= rotl32(b,14);
    a ^= c; a -= rotl32(c,11);
    b ^= a; b -= rotl32(a,25);
    c ^= b; c -= rotl32(b,16);
    a ^= c; a -= rotl32(c,4);
    b ^= a; b -= rotl32(a,14);
    c ^= b; c -= rotl32(b,24);
    return c;
}
#endif

#ifndef __OSL_USE_REFERENCE_INT_HASH
	// Warning the reference hash may cause incorrect results when
	// used inside a SIMD loop due to its complexity
	#define __OSL_USE_REFERENCE_INT_HASH 0
#endif

/// hash an array of N 32 bit values into a pseudo-random value
/// based on my favorite hash: http://burtleburtle.net/bob/c/lookup3.c
/// templated so that the compiler can unroll the loops for us
template <int N>
OSL_FORCEINLINE OSL_HOSTDEVICE unsigned int
reference_inthash (const unsigned int k[N]) {
    // now hash the data!
    unsigned int a, b, c, len = N;
    a = b = c = 0xdeadbeef + (len << 2) + 13;
    while (len > 3) {
        a += k[0];
        b += k[1];
        c += k[2];
        OIIO::bjhash::bjmix(a, b, c);
        len -= 3;
        k += 3;
    }
    switch (len) {
        case 3 : c += k[2];
        case 2 : b += k[1];
        case 1 : a += k[0];
        c = OIIO::bjhash::bjfinal(a, b, c);
        case 0:
            break;
    }
    return c;
}

#if (__OSL_USE_REFERENCE_INT_HASH == 0)
	// Do not rely on compilers to fully optimizing the
	// reference_inthash<int N> template above.
	// The fact it takes an array parameter
	// could cause confusion and extra work for a compiler to convert
	// from x,y,z,w -> k[4] then index them.  So to simplify things
	// for compilers and avoid requiring actual stack space for the k[N]
	// array versus tracking builtin integer types in registers,
	// here are hand unrolled versions of inthash for 1-5 parameters.
	// The upshot is no need to create a k[N] on the stack and hopefully
	// simpler time for optimizer as it has no need to deal with while
	// loop and switch statement.
	OSL_FORCEINLINE OSL_HOSTDEVICE unsigned int
	inthash (const unsigned int k0) {
		// now hash the data!
		unsigned int start_val = 0xdeadbeef + (1 << 2) + 13;

		unsigned int a = start_val + k0;
		unsigned int c = OIIO::bjhash::bjfinal(a, start_val, start_val);
		return c;
	}

	OSL_FORCEINLINE OSL_HOSTDEVICE unsigned int
	inthash (const unsigned int k0, const unsigned int k1) {
		// now hash the data!
		unsigned int start_val = 0xdeadbeef + (2 << 2) + 13;

		unsigned int a = start_val + k0;
		unsigned int b = start_val + k1;
		unsigned int c = OIIO::bjhash::bjfinal(a, b, start_val);
		return c;
	}

	OSL_FORCEINLINE OSL_HOSTDEVICE unsigned int
	inthash (const unsigned int k0, const unsigned int k1, const unsigned int k2) {
		// now hash the data!
		unsigned int start_val = 0xdeadbeef + (3 << 2) + 13;

		unsigned int a = start_val + k0;
		unsigned int b = start_val + k1;
		unsigned int c = start_val + k2;
		c = OIIO::bjhash::bjfinal(a, b, c);
		return c;
	}

	OSL_FORCEINLINE OSL_HOSTDEVICE unsigned int
	inthash (const unsigned int k0, const unsigned int k1, const unsigned int k2, const unsigned int k3) {
		// now hash the data!
		unsigned int start_val = 0xdeadbeef + (4 << 2) + 13;

		unsigned int a = start_val + k0;
		unsigned int b = start_val + k1;
		unsigned int c = start_val + k2;
		OIIO::bjhash::bjmix(a, b, c);
		a += k3;
		c = OIIO::bjhash::bjfinal(a, b, c);
		return c;
	}

	OSL_FORCEINLINE OSL_HOSTDEVICE unsigned int
	inthash (const unsigned int k0, const unsigned int k1, const unsigned int k2, const unsigned int k3, const unsigned int k4) {
		// now hash the data!
		unsigned int start_val = 0xdeadbeef + (5 << 2) + 13;

		unsigned int a = start_val + k0;
		unsigned int b = start_val + k1;
		unsigned int c = start_val + k2;
		OIIO::bjhash::bjmix(a, b, c);
		b += k4;
		a += k3;
		c = OIIO::bjhash::bjfinal(a, b, c);
		return c;
	}
#endif

#ifndef __CUDA_ARCH__
// Do four 2D hashes simultaneously.
inline int4
inthash_simd (const int4& key_x, const int4& key_y)
{
    const int len = 2;
    const int seed_ = (0xdeadbeef + (len << 2) + 13);
    static const OIIO_SIMD4_ALIGN int seed[4] = { seed_,seed_,seed_,seed_};
    int4 a = (*(int4*)&seed)+key_x, b = (*(int4*)&seed)+key_y, c = (*(int4*)&seed);
    return bjfinal (a, b, c);
}


// Do four 3D hashes simultaneously.
inline int4
inthash_simd (const int4& key_x, const int4& key_y, const int4& key_z)
{
    const int len = 3;
    const int seed_ = (0xdeadbeef + (len << 2) + 13);
    static const OIIO_SIMD4_ALIGN int seed[4] = { seed_,seed_,seed_,seed_};
    int4 a = (*(int4*)&seed)+key_x, b = (*(int4*)&seed)+key_y, c = (*(int4*)&seed)+key_z;
    return bjfinal (a, b, c);
}



// Do four 3D hashes simultaneously.
inline int4
inthash_simd (const int4& key_x, const int4& key_y, const int4& key_z, const int4& key_w)
{
    const int len = 4;
    const int seed_ = (0xdeadbeef + (len << 2) + 13);
    static const OIIO_SIMD4_ALIGN int seed[4] = { seed_,seed_,seed_,seed_};
    int4 a = (*(int4*)&seed)+key_x, b = (*(int4*)&seed)+key_y, c = (*(int4*)&seed)+key_z;
    bjmix (a, b, c);
    a += key_w;
    return bjfinal(a, b, c);
}
#endif


// Cell and Hash noise only differ in how they transform their inputs from
// float to unsigned int for use in the inthash function.
// IntHashNoiseBase serves as base class with DerivedT::transformToUint
// controlling how the float inputs are transformed
template<typename DerivedT>
struct IntHashNoiseBase  {
	OSL_FORCEINLINE OSL_HOSTDEVICE IntHashNoiseBase () { }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, float x) const {
    	result = hashFloat(x);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, float x, float y) const {
    	result = hashFloat(x, y);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p) const {
    	result = hashFloat(p.x, p.y, p.z);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, float t) const {
    	result = hashFloat(p.x, p.y, p.z, t);
    }


    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, float x) const {
    	result = hashVec(x);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float y) const {
    	result = hashVec(x, y);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p) const {
    	result = hashVec(p.x, p.y, p.z);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, float t) const {
    	result = hashVec(p.x, p.y, p.z, t);
    }


private:
    template<typename ...ListT>
    OSL_FORCEINLINE OSL_HOSTDEVICE float
    hashFloat (ListT... floatList) const {
        return bits_to_01(inthash(DerivedT::transformToUint(floatList)...));
    }

    template<typename ...ListT>
    OSL_FORCEINLINE OSL_HOSTDEVICE Vec3
    hashVec (ListT... floatList) const {
    	return inthashVec(DerivedT::transformToUint(floatList)...);
    }

#if __OSL_USE_REFERENCE_INT_HASH
    // Allow any number of arguments to be adapted to the array based reference_inthash
    // and leave door open for overloads of hash3 with specific parameters

    // Produce Vec3 result from any number of arguments with array based reference_inthash
    // and extra seed values, but leave door open for overloads of inthashVec
    // for specific parameter combinations
    template<typename ...ListT>
    OSL_FORCEINLINE OSL_HOSTDEVICE Vec3
    inthashVec (ListT... uintList) const {
        constexpr int N = sizeof...(uintList) + 1;
    	unsigned int k[N] = {uintList...};

    	Vec3 result;
        k[N-1] = 0u; result.x = bits_to_01 (reference_inthash<N> (k));
        k[N-1] = 1u; result.y = bits_to_01 (reference_inthash<N> (k));
        k[N-1] = 2u; result.z = bits_to_01 (reference_inthash<N> (k));
        return result;
    }
#else
    // Produce Vec3 result from any number of arguments with inthash and
    // extra seed values, but leave door open for overloads of inthashVec
    // for specific parameter combinations
    template<typename ...ListT>
    OSL_FORCEINLINE OSL_HOSTDEVICE Vec3
	inthashVec (ListT... uintList) const  {
        return Vec3(bits_to_01(inthash(uintList..., 0u)),
        			bits_to_01(inthash(uintList..., 1u)),
					bits_to_01(inthash(uintList..., 2u)));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE Vec3
	inthashVec (const unsigned int k0, const unsigned int k1, const unsigned int k2) const {
        // combine implementations of reference_inthash<3> and reference_inthash<4>
    	// so that we can only perform the bjmix once because k0,k1,k2 are the
    	// same values passed to reference_inthash<4>, only k3 differs
        // and if we unroll the work it can be separated

        // now hash the data!
        unsigned int start_val = 0xdeadbeef + (4 << 2) + 13;

        unsigned int a = start_val + k0;
        unsigned int b = start_val + k1;
        unsigned int c = start_val + k2;
        OIIO::bjhash::bjmix(a, b, c);

        return Vec3(bits_to_01( OIIO::bjhash::bjfinal(a+0, b, c) ),
                    bits_to_01( OIIO::bjhash::bjfinal(a+1, b, c) ),
                    bits_to_01( OIIO::bjhash::bjfinal(a+2, b, c) ));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE Vec3
	inthashVec (const unsigned int k0, const unsigned int k1, const unsigned int k2, const unsigned int k3) const {
        // combine implementations of reference_inthash<3> and reference_inthash<5>
    	// so that we can only perform the bjmix once because k0,k1,k2,k3 are the
    	// same values passed to reference_inthash<5>, only k4 differs
        // and if we unroll the work it can be separated

        // now hash the data!
        unsigned int start_val = 0xdeadbeef + (5 << 2) + 13;

        unsigned int a = start_val + k0;
        unsigned int b = start_val + k1;
        unsigned int c = start_val + k2;
        OIIO::bjhash::bjmix(a, b, c);
        unsigned int a2 = a + k3;

        return Vec3(bits_to_01( OIIO::bjhash::bjfinal(a2, b+0, c) ),
                    bits_to_01( OIIO::bjhash::bjfinal(a2, b+1, c) ),
                    bits_to_01( OIIO::bjhash::bjfinal(a2, b+2, c) ));
    }
#endif
};

struct CellNoise: public IntHashNoiseBase<CellNoise>  {
    OSL_FORCEINLINE OSL_HOSTDEVICE CellNoise () { }

    static OSL_FORCEINLINE OSL_HOSTDEVICE unsigned int
    transformToUint(float val)
    {
        return OIIO::ifloor(val);
    }
};

struct HashNoise: public IntHashNoiseBase<HashNoise>  {
    OSL_FORCEINLINE OSL_HOSTDEVICE HashNoise () { }

    static OSL_FORCEINLINE OSL_HOSTDEVICE unsigned int
    transformToUint(float val)
    {
        return OIIO::bit_cast<float,unsigned int>(val);
    }
};

// Periodic Cell and Hash Noise simply wraps its inputs before
// performing the same conversions and hashing as the non-periodic version.
// We define a wrapper on top of Cell or Hash Noise to reuse
// its underlying implementation.
template <typename BaseNoiseT>
struct PeriodicAdaptionOf {
	OSL_FORCEINLINE OSL_HOSTDEVICE PeriodicAdaptionOf () { }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, float x, float px) const {
    	m_impl(result, wrap (x, px));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, float x, float y,
                            float px, float py) const {
    	m_impl(result, wrap (x, px),
                       wrap (y, py));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p,
                            const Vec3 &pp) const {
    	m_impl(result, wrap (p, pp));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, float t,
                            const Vec3 &pp, float tt) const {
    	m_impl(result, wrap (p, pp),
					   wrap (t, tt));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float px) const {
    	m_impl(result, wrap (x, px));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float y,
                            float px, float py) const {
    	m_impl(result, wrap (x, px),
    				   wrap (y, py));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, const Vec3 &pp) const {
    	m_impl(result, wrap (p, pp));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, float t,
                            const Vec3 &pp, float tt) const {
    	m_impl(result, wrap (p, pp),
					   wrap (t, tt));
    }

private:
    OSL_FORCEINLINE OSL_HOSTDEVICE float wrap (float s, float period) const {
        // TODO: investigate if quick_floor is legal here
        // and or beneficial
        period = floorf (period);
        if (period < 1.0f)
            period = 1.0f;
        return s - period * floorf (s / period);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE Vec3 wrap (const Vec3 &s, const Vec3 &period) const {
    	return Vec3(wrap(s.x, period.x),
    			    wrap(s.y, period.y),
					wrap(s.z, period.z));
    }

    BaseNoiseT m_impl;
};

using PeriodicCellNoise = PeriodicAdaptionOf<CellNoise>;
using PeriodicHashNoise = PeriodicAdaptionOf<HashNoise>;




inline OSL_HOSTDEVICE int
inthashi (int x)
{
    return static_cast<int>(inthash(
		static_cast<unsigned int>(x)
	));
}

inline OSL_HOSTDEVICE int
inthashf (float x)
{
    return static_cast<int>(
		inthash(OIIO::bit_cast<float,unsigned int>(x))
	);
}

inline OSL_HOSTDEVICE int
inthashf (float x, float y)
{
    return static_cast<int>(
		inthash(
			OIIO::bit_cast<float,unsigned int>(x),
			OIIO::bit_cast<float,unsigned int>(y)
		)
	);
}


inline OSL_HOSTDEVICE int
inthashf (const float *x)
{
    return static_cast<int>(
		inthash(
			OIIO::bit_cast<float,unsigned int>(x[0]),
			OIIO::bit_cast<float,unsigned int>(x[1]),
			OIIO::bit_cast<float,unsigned int>(x[2])
		)
	);
}


inline OSL_HOSTDEVICE int
inthashf (const float *x, float y)
{
    return static_cast<int>(
		inthash(
			OIIO::bit_cast<float,unsigned int>(x[0]),
			OIIO::bit_cast<float,unsigned int>(x[1]),
			OIIO::bit_cast<float,unsigned int>(x[2]),
			OIIO::bit_cast<float,unsigned int>(y)
		)
	);
}


// Define select(bool,truevalue,falsevalue) template that works for a
// variety of types that we can use for both scalars and vectors. Because ?:
// won't work properly in template code with vector ops.
// NOTE: Removing template and require explicit functions for different
// combinations be specified using polymorphism.  Main reason is so that
// different versions can pass parameters by value vs. reference.
//template <typename B, typename F>
//OSL_FORCEINLINE OSL_HOSTDEVICE F select (B b, const F& t, const F& f) { return b ? t : f; }

OSL_FORCEINLINE OSL_HOSTDEVICE float select(const bool b, float t, float f) {
    // NOTE:  parameters must NOT be references, to avoid inlining caller's creation of
    // these values down inside the conditional assignments which can create to complex
    // of a code flow for Clang's vectorizer to handle
    static_assert(!std::is_reference<decltype(b)>::value, "parameters to select cannot be references");
    static_assert(!std::is_reference<decltype(t)>::value, "parameters to select cannot be references");
    static_assert(!std::is_reference<decltype(f)>::value, "parameters to select cannot be references");
    return b ? t : f;
}

OSL_FORCEINLINE OSL_HOSTDEVICE Dual2<float> select(const bool b, const Dual2<float> &t, const Dual2<float> &f) {
    // Because t & f are a references, we don't want inlining to expand t.val(), t.dx(), or t.dy()
    // inside the conditional branch creating a more complex flow by forcing the inlined
    // abstract syntax tree to only be evaluated when the condition is true.
    // To avoid this we assign t.val(), t.dx(), t.dy() to local values outside the
    // conditional to insulate it.
    // NOTE:  This technique was also necessary to enable vectorization of nested select calls
    float val(f.val());
    float tval(t.val());

    float dx(f.dx());
    float tdx(t.dx());

    float dy(f.dy());
    float tdy(t.dy());

    // Blend per builtin component to allow
    // the compiler to track builtins and privatize the data layout
    // versus requiring a stack location.
    // Without this work per component, gathers & scatters were being emitted
    // when used inside SIMD loops.
#if OSL_NON_INTEL_CLANG
    // Clang's vectorizor was really insistent that a select operation could not be replaced
    // with control flow, so had to re-introduce the ? operator to make it happy
    return Dual2<float> (
        b ? tval : val,
        b ? tdx : dx,
        b ? tdy : dy);
#else
    if (b)
        val = tval;

    if (b)
        dx = tdx;

    if (b)
        dy = tdy;

    return Dual2<float> (
            val,
            dx,
            dy);
#endif
}

#ifndef __CUDA_ARCH__
// Already provided by oiio/simd.h
//OSL_FORCEINLINE int4 select (const bool4& b, const int4& t, const int4& f) {
//    return blend (f, t, b);
//}

// Already provided by oiio/simd.h
//OSL_FORCEINLINE float4 select (const bool4& b, const float4& t, const float4& f) {
//    return blend (f, t, b);
//}

OSL_FORCEINLINE float4 select (const int4& b, const float4& t, const float4& f) {
    return blend (f, t, bool4(b));
}

OSL_FORCEINLINE Dual2<float4>
select (const bool4& b, const Dual2<float4>& t, const Dual2<float4>& f) {
    return Dual2<float4> (blend (f.val(), t.val(), b),
                          blend (f.dx(),  t.dx(),  b),
                          blend (f.dy(),  t.dy(),  b));
}

OSL_FORCEINLINE Dual2<float4> select (const int4& b, const Dual2<float4>& t, const Dual2<float4>& f) {
    return select (bool4(b), t, f);
}
#endif



// Define negate_if(value,bool) that will work for both scalars and vectors,
// as well as Dual2's of both.
// NOTE: Removing template and require explicit functions for different
// combinations be specified using polymorphism.  Main reason is so that
// different versions can pass parameters by value vs. reference.
//template <typename B, typename F>
//template<typename FLOAT, typename BOOL>
//OSL_FORCEINLINE OSL_HOSTDEVICE FLOAT negate_if (const FLOAT& val, const BOOL& b) {
//    return b ? -val : val;
//}

OSL_FORCEINLINE OSL_HOSTDEVICE float negate_if (const float val, const bool cond) {
    // NOTE:  parameters must NOT be references, to avoid inlining caller's creation of
    // these values down inside the conditional assignments which can create to complex
    // of a code flow for vectorizer to handle
    static_assert(!std::is_reference<decltype(val)>::value, "parameters to negate_if cannot be references");
    return cond ? -val : val;
}


OSL_FORCEINLINE OSL_HOSTDEVICE Dual2<float> negate_if(const Dual2<float> &val, const bool cond) {
    // NOTE: negation happens outside of conditional, then is blended based on the condition
    Dual2<float> neg_val(-val);

    // Blend per builtin component to allow
    // the compiler to track builtins and privatize the data layout
    // versus requiring a stack location.
    float v = val.val();
    if (cond) {
        v = neg_val.val();
    }

    float dx = val.dx();
    if (cond) {
        dx = neg_val.dx();
    }

    float dy = val.dy();
    if (cond) {
        dy = neg_val.dy();
    }

    return Dual2<float>(v, dx, dy);
}

#if 0 // Unneeded currently
template <> OSL_FORCEINLINE Dual2<Vec3> negate_if(const Dual2<Vec3> &val, const bool &b) {
    Dual2<Vec3> r;
    // Use a ternary operation per builtin component to allow
    // the compiler to track builtins and privatize the data layout
    // versus requiring a stack location.
    Dual2<Vec3> neg_val(-val);
    r.val() = b ? neg_val.val() : val.val();
    r.dx() = b ? neg_val.dx() : val.dx();
    r.dy() = b ? neg_val.dy() : val.dy();
    return r;
}
#endif

#ifndef __CUDA_ARCH__
OSL_FORCEINLINE float4 negate_if (const float4& val, const int4& b) {
    // Special case negate_if for SIMD -- can do it with bit tricks, no branches
    int4 highbit (0x80000000);
    return bitcast_to_float4 (bitcast_to_int4(val) ^ (blend0 (highbit, bool4(b))));
}

// Special case negate_if for SIMD -- can do it with bit tricks, no branches
OSL_FORCEINLINE Dual2<float4> negate_if (const Dual2<float4>& val, const int4& b)
{
    return Dual2<float4> (negate_if (val.val(), b),
                          negate_if (val.dx(),  b),
                          negate_if (val.dy(),  b));
}
#endif


#ifndef __CUDA_ARCH__
// Define shuffle<> template that works with Dual2<float4> analogously to
// how it works for float4.
template<int i0, int i1, int i2, int i3>
OSL_FORCEINLINE Dual2<float4> shuffle (const Dual2<float4>& a)
{
    return Dual2<float4> (OIIO::simd::shuffle<i0,i1,i2,i3>(a.val()),
                          OIIO::simd::shuffle<i0,i1,i2,i3>(a.dx()),
                          OIIO::simd::shuffle<i0,i1,i2,i3>(a.dy()));
}

template<int i>
OSL_FORCEINLINE Dual2<float4> shuffle (const Dual2<float4>& a)
{
    return Dual2<float4> (OIIO::simd::shuffle<i>(a.val()),
                          OIIO::simd::shuffle<i>(a.dx()),
                          OIIO::simd::shuffle<i>(a.dy()));
}

// Define extract<> that works with Dual2<float4> analogously to how it
// works for float4.
template<int i>
OSL_FORCEINLINE Dual2<float> extract (const Dual2<float4>& a)
{
    return Dual2<float> (OIIO::simd::extract<i>(a.val()),
                         OIIO::simd::extract<i>(a.dx()),
                         OIIO::simd::extract<i>(a.dy()));
}
#endif



// Equivalent to OIIO::bilerp (a, b, c, d, u, v), but if abcd are already
// packed into a float4. We assume T is float and VECTYPE is float4,
// but it also works if T is Dual2<float> and VECTYPE is Dual2<float4>.
template<typename T, typename VECTYPE>
OSL_FORCEINLINE OSL_HOSTDEVICE T bilerp (VECTYPE abcd, T u, T v) {
    VECTYPE xx = OIIO::lerp (abcd, OIIO::simd::shuffle<1,1,3,3>(abcd), u);
    return OIIO::simd::extract<0>(OIIO::lerp (xx,OIIO::simd::shuffle<2>(xx), v));
}

#ifndef __CUDA_ARCH__
// Equivalent to OIIO::bilerp (a, b, c, d, u, v), but if abcd are already
// packed into a float4 and uv are already packed into the first two
// elements of a float4. We assume VECTYPE is float4, but it also works if
// VECTYPE is Dual2<float4>.
OSL_FORCEINLINE Dual2<float> bilerp (const Dual2<float4>& abcd, const Dual2<float4>& uv) {
    Dual2<float4> xx = OIIO::lerp (abcd, shuffle<1,1,3,3>(abcd), shuffle<0>(uv));
    return extract<0>(OIIO::lerp (xx,shuffle<2>(xx), shuffle<1>(uv)));
}


// Equivalent to OIIO::trilerp (a, b, c, d, e, f, g, h, u, v, w), but if
// abcd and efgh are already packed into float4's and uvw are packed into
// the first 3 elements of a float4.
OSL_FORCEINLINE float trilerp (const float4& abcd, const float4& efgh, const float4& uvw) {
    // Interpolate along z axis by w
    float4 xy = OIIO::lerp (abcd, efgh, OIIO::simd::shuffle<2>(uvw));
    // Interpolate along x axis by u
    float4 xx = OIIO::lerp (xy, OIIO::simd::shuffle<1,1,3,3>(xy), OIIO::simd::shuffle<0>(uvw));
    // interpolate along y axis by v
    return OIIO::simd::extract<0>(OIIO::lerp (xx, OIIO::simd::shuffle<2>(xx), OIIO::simd::shuffle<1>(uvw)));
}
#endif



// always return a value inside [0,b) - even for negative numbers
OSL_FORCEINLINE OSL_HOSTDEVICE int imod(int a, int b) {
#if 0
    a %= b;
    return a < 0 ? a + b : a;
#else
    // Avoid confusing vectorizor by returning a single value
    int remainder = a % b;
    if (remainder < 0) {
        remainder += b;
    }
    return remainder;
#endif
}

#ifndef __CUDA_ARCH__
// imod four values at once
inline int4 imod(const int4& a, int b) {
    int4 c = a % b;
    return c + select(c < 0, int4(b), int4::Zero());
}
#endif

// floorfrac return ifloor as well as the fractional remainder
// FIXME: already implemented inside OIIO but can't easily override it for duals
//        inside a different namespace
OSL_FORCEINLINE OSL_HOSTDEVICE float floorfrac(float x, int* i) {
    *i = OIIO::ifloor(x);
    return x - *i;
}

// floorfrac with derivs
OSL_FORCEINLINE OSL_HOSTDEVICE Dual2<float> floorfrac(const Dual2<float> &x, int* i) {
    float frac = floorfrac(x.val(), i);
    // slope of x is not affected by this operation
    return Dual2<float>(frac, x.dx(), x.dy());
}

#ifndef __CUDA_ARCH__
// floatfrac for four sets of values at once.
inline float4 floorfrac(const float4& x, int4 * i) {
#if 0
    float4 thefloor = floor(x);
    *i = int4(thefloor);
    return x-thefloor;
#else
    int4 thefloor = OIIO::simd::ifloor (x);
    *i = thefloor;
    return x - float4(thefloor);
#endif
}

// floorfrac with derivs, computed on 4 values at once.
inline Dual2<float4> floorfrac(const Dual2<float4> &x, int4* i) {
    float4 frac = floorfrac(x.val(), i);
    // slope of x is not affected by this operation
    return Dual2<float4>(frac, x.dx(), x.dy());
}
#endif


// Perlin 'fade' function. Can be overloaded for float, Dual2, as well
// as float4 / Dual2<float4>.
template <typename T>
OSL_HOSTDEVICE OSL_FORCEINLINE T fade (const T &t) {
   return t * t * t * (t * (t * T(6.0f) - T(15.0f)) + T(10.0f));
}



// 1,2,3 and 4 dimensional gradient functions - perform a dot product against a
// randomly chosen vector. Note that the gradient vector is not normalized, but
// this only affects the overall "scale" of the result, so we simply account for
// the scale by multiplying in the corresponding "perlin" function.
// These factors were experimentally calculated to be:
//    1D:   0.188
//    2D:   0.507
//    3D:   0.936
//    4D:   0.870

template <typename T>
OSL_FORCEINLINE OSL_HOSTDEVICE T grad (int hash, const T &x) {
    int h = hash & 15;
    float g = 1 + (h & 7);  // 1, 2, .., 8
    if (h&8) g = -g;        // random sign
    return g * x;           // dot-product
}

template <typename I, typename T>
OSL_FORCEINLINE OSL_HOSTDEVICE T grad (const I &hash, const T &x, const T &y) {
    // 8 possible directions (+-1,+-2) and (+-2,+-1)
    I h = hash & 7;
    T u = select (h<4, x, y);
    T v = 2.0f * select (h<4, y, x);
    // compute the dot product with (x,y).
    return negate_if(u, h&1) + negate_if(v, h&2);
}

template <typename I, typename T>
OSL_FORCEINLINE OSL_HOSTDEVICE T grad (const I &hash, const T &x, const T &y, const T &z) {
    // use vectors pointing to the edges of the cube
    I h = hash & 15;
    T u = select (h<8, x, y);
    T v = select (h<4, y, select ((h==I(12))|(h==I(14)), x, z));
    return negate_if(u,h&1) + negate_if(v,h&2);
}

template <typename T>
OSL_FORCEINLINE OSL_HOSTDEVICE T grad (const int hash, const T &x, const T &y, const T &z) {
    // use vectors pointing to the edges of the cube
    int h = hash & 15;
    T u = select (h<8, x, y);
    // Changed from bitwise | to logical || to avoid conversions
    // to integer from native boolean that was causing gather + scatters
    // to be generated when used with the select vs.
    // simple masking or blending.
    // TODO: couldn't change the grad version above because OpenImageIO::v1_7::simd::bool4
    // has no || operator defined.  Would be preferable to implement bool4::operator||
    // and not have this version of grad separate
    T v = select (h<4, y, select ((h==int(12))||(h==int(14)), x, z));
    return negate_if(u,h&1) + negate_if(v,h&2);
}

template <typename I, typename T>
OSL_FORCEINLINE OSL_HOSTDEVICE T grad (const I &hash, const T &x, const T &y, const T &z, const T &w) {
    // use vectors pointing to the edges of the hypercube
    I h = hash & 31;
    T u = select (h<24, x, y);
    T v = select (h<16, y, z);
    T s = select (h<8 , z, w);
    return negate_if(u,h&1) + negate_if(v,h&2) + negate_if(s,h&4);
}

typedef Imath::Vec3<int> Vec3i;

OSL_FORCEINLINE OSL_HOSTDEVICE Vec3 grad (const Vec3i &hash, float x) {
    return Vec3 (grad (hash.x, x),
                 grad (hash.y, x),
                 grad (hash.z, x));
}

OSL_FORCEINLINE OSL_HOSTDEVICE Dual2<Vec3> grad (const Vec3i &hash, Dual2<float> x) {
    Dual2<float> rx = grad (hash.x, x);
    Dual2<float> ry = grad (hash.y, x);
    Dual2<float> rz = grad (hash.z, x);
    return make_Vec3 (rx, ry, rz);
}


OSL_FORCEINLINE OSL_HOSTDEVICE Vec3 grad (const Vec3i &hash, float x, float y) {
    return Vec3 (grad (hash.x, x, y),
                 grad (hash.y, x, y),
                 grad (hash.z, x, y));
}

OSL_FORCEINLINE OSL_HOSTDEVICE Dual2<Vec3> grad (const Vec3i &hash, Dual2<float> x, Dual2<float> y) {
    Dual2<float> rx = grad (hash.x, x, y);
    Dual2<float> ry = grad (hash.y, x, y);
    Dual2<float> rz = grad (hash.z, x, y);
    return make_Vec3 (rx, ry, rz);
}

OSL_FORCEINLINE OSL_HOSTDEVICE Vec3 grad (const Vec3i &hash, float x, float y, float z) {
    return Vec3 (grad (hash.x, x, y, z),
                 grad (hash.y, x, y, z),
                 grad (hash.z, x, y, z));
}

OSL_FORCEINLINE OSL_HOSTDEVICE Dual2<Vec3> grad (const Vec3i &hash, Dual2<float> x, Dual2<float> y, Dual2<float> z) {
    Dual2<float> rx = grad (hash.x, x, y, z);
    Dual2<float> ry = grad (hash.y, x, y, z);
    Dual2<float> rz = grad (hash.z, x, y, z);
    return make_Vec3 (rx, ry, rz);
}

OSL_FORCEINLINE OSL_HOSTDEVICE Vec3 grad (const Vec3i &hash, float x, float y, float z, float w) {
    return Vec3 (grad (hash.x, x, y, z, w),
                 grad (hash.y, x, y, z, w),
                 grad (hash.z, x, y, z, w));
}

OSL_FORCEINLINE OSL_HOSTDEVICE Dual2<Vec3> grad (const Vec3i &hash, Dual2<float> x, Dual2<float> y, Dual2<float> z, Dual2<float> w) {
    Dual2<float> rx = grad (hash.x, x, y, z, w);
    Dual2<float> ry = grad (hash.y, x, y, z, w);
    Dual2<float> rz = grad (hash.z, x, y, z, w);
    return make_Vec3 (rx, ry, rz);
}

template <typename T>
OSL_FORCEINLINE OSL_HOSTDEVICE T scale1 (const T &result) { return 0.2500f * result; }
template <typename T>
OSL_FORCEINLINE OSL_HOSTDEVICE T scale2 (const T &result) { return 0.6616f * result; }
template <typename T>
OSL_FORCEINLINE OSL_HOSTDEVICE T scale3 (const T &result) { return 0.9820f * result; }
template <typename T>
OSL_FORCEINLINE OSL_HOSTDEVICE T scale4 (const T &result) { return 0.8344f * result; }


struct HashScalar {

    OSL_FORCEINLINE OSL_HOSTDEVICE int operator() (int x) const {
        return static_cast<int>(
			inthash(static_cast<unsigned int>(x))
		);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE int operator() (int x, int y) const {
    	return static_cast<int>(
			inthash(static_cast<unsigned int>(x),
        	        static_cast<unsigned int>(y))
		);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE int operator() (int x, int y, int z) const {
    	return static_cast<int>(
			inthash(static_cast<unsigned int>(x),
        	        static_cast<unsigned int>(y),
			 	    static_cast<unsigned int>(z))
		);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE int operator() (int x, int y, int z, int w) const {
    	return static_cast<int>(
			inthash(static_cast<unsigned int>(x),
        	        static_cast<unsigned int>(y),
			  	    static_cast<unsigned int>(z),
					static_cast<unsigned int>(w))
		);
    }

#ifndef __CUDA_ARCH__
    // 4 2D hashes at once!
    OSL_FORCEINLINE int4 operator() (const int4& x, const int4& y) const {
        return inthash_simd (x, y);
    }

    // 4 3D hashes at once!
    OSL_FORCEINLINE int4 operator() (const int4& x, const int4& y, const int4& z) const {
        return inthash_simd (x, y, z);
    }

    // 4 3D hashes at once!
    OSL_FORCEINLINE int4 operator() (const int4& x, const int4& y, const int4& z, const int4& w) const {
        return inthash_simd (x, y, z, w);
    }
#endif

};


OSL_FORCEINLINE OSL_HOSTDEVICE Vec3i sliceup (unsigned int h) {
    // we only need the low-order bits to be random, so split out
    // the 32 bit result into 3 parts for each channel
    return Vec3i(
		(h      ) & 0xFF,
		(h >> 8 ) & 0xFF,
		(h >> 16) & 0xFF);
}

struct HashVector {
    static OSL_FORCEINLINE OSL_HOSTDEVICE HashScalar convertToScalar() { return HashScalar(); }

    OSL_FORCEINLINE OSL_HOSTDEVICE Vec3i operator() (int x) const {
        return sliceup(inthash(static_cast<unsigned int>(x)));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE Vec3i operator() (int x, int y) const {
        return sliceup(inthash(static_cast<unsigned int>(x),
        		               static_cast<unsigned int>(y)));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE Vec3i operator() (int x, int y, int z) const {
        return sliceup(inthash(static_cast<unsigned int>(x),
        		               static_cast<unsigned int>(y),
							   static_cast<unsigned int>(z)));
    }
    OSL_FORCEINLINE OSL_HOSTDEVICE Vec3i operator() (int x, int y, int z, int w) const {
        return sliceup(inthash(static_cast<unsigned int>(x),
        		               static_cast<unsigned int>(y),
							   static_cast<unsigned int>(z),
							   static_cast<unsigned int>(w)));
    }

#ifndef __CUDA_ARCH__
    // Vector hash of 4 3D points at once
    OSL_FORCEINLINE void operator() (int4 *result, const int4& x, const int4& y) const {
        int4 h = inthash_simd (x, y);
        result[0] = (h        ) & 0xFF;
        result[1] = (srl(h,8 )) & 0xFF;
        result[2] = (srl(h,16)) & 0xFF;
    }

    // Vector hash of 4 3D points at once
    OSL_FORCEINLINE void operator() (int4 *result, const int4& x, const int4& y, const int4& z) const {
        int4 h = inthash_simd (x, y, z);
        result[0] = (h        ) & 0xFF;
        result[1] = (srl(h,8 )) & 0xFF;
        result[2] = (srl(h,16)) & 0xFF;
    }

    // Vector hash of 4 3D points at once
    OSL_FORCEINLINE void operator() (int4 *result, const int4& x, const int4& y, const int4& z, const int4& w) const {
        int4 h = inthash_simd (x, y, z, w);
        result[0] = (h        ) & 0xFF;
        result[1] = (srl(h,8 )) & 0xFF;
        result[2] = (srl(h,16)) & 0xFF;
    }
#endif

};


struct HashScalarPeriodic {
private:
    friend struct HashVectorPeriodic;
    OSL_FORCEINLINE OSL_HOSTDEVICE HashScalarPeriodic () {}
public:
    OSL_FORCEINLINE OSL_HOSTDEVICE HashScalarPeriodic (float px) {
        m_px = OIIO::ifloor(px); if (m_px < 1) m_px = 1;
    }
    OSL_FORCEINLINE OSL_HOSTDEVICE HashScalarPeriodic (float px, float py) {
        m_px = OIIO::ifloor(px); if (m_px < 1) m_px = 1;
        m_py = OIIO::ifloor(py); if (m_py < 1) m_py = 1;
    }
    OSL_FORCEINLINE OSL_HOSTDEVICE HashScalarPeriodic (float px, float py, float pz) {
        m_px = OIIO::ifloor(px); if (m_px < 1) m_px = 1;
        m_py = OIIO::ifloor(py); if (m_py < 1) m_py = 1;
        m_pz = OIIO::ifloor(pz); if (m_pz < 1) m_pz = 1;
    }
    OSL_FORCEINLINE OSL_HOSTDEVICE HashScalarPeriodic (float px, float py, float pz, float pw) {
        m_px = OIIO::ifloor(px); if (m_px < 1) m_px = 1;
        m_py = OIIO::ifloor(py); if (m_py < 1) m_py = 1;
        m_pz = OIIO::ifloor(pz); if (m_pz < 1) m_pz = 1;
        m_pw = OIIO::ifloor(pw); if (m_pw < 1) m_pw = 1;
    }

    int m_px, m_py, m_pz, m_pw;

    OSL_FORCEINLINE OSL_HOSTDEVICE int operator() (int x) const {
        return static_cast<int>(
			inthash (static_cast<unsigned int>(imod (x, m_px)))
		);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE int operator() (int x, int y) const {
        return static_cast<int>(
			inthash(static_cast<unsigned int>(imod (x, m_px)),
        	        static_cast<unsigned int>(imod (y, m_py)))
		);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE int operator() (int x, int y, int z) const {
        return static_cast<int>(
			inthash(static_cast<unsigned int>(imod (x, m_px)),
        	        static_cast<unsigned int>(imod (y, m_py)),
			 	    static_cast<unsigned int>(imod (z, m_pz)))
		);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE int operator() (int x, int y, int z, int w) const {
        return static_cast<int>(
			inthash(static_cast<unsigned int>(imod (x, m_px)),
        	        static_cast<unsigned int>(imod (y, m_py)),
			  	    static_cast<unsigned int>(imod (z, m_pz)),
					static_cast<unsigned int>(imod (w, m_pw)))
		);
    }

#ifndef __CUDA_ARCH__
    // 4 2D hashes at once!
    int4 operator() (const int4& x, const int4& y) const {
        return inthash_simd (imod(x,m_px), imod(y,m_py));
    }

    // 4 3D hashes at once!
    int4 operator() (const int4& x, const int4& y, const int4& z) const {
        return inthash_simd (imod(x,m_px), imod(y,m_py), imod(z,m_pz));
    }

    // 4 4D hashes at once
    int4 operator() (const int4& x, const int4& y, const int4& z, const int4& w) const {
        return inthash_simd (imod(x,m_px), imod(y,m_py), imod(z,m_pz), imod(w,m_pw));
    }
#endif

};

struct HashVectorPeriodic {
	OSL_FORCEINLINE OSL_HOSTDEVICE HashVectorPeriodic (float px) {
        m_px = OIIO::ifloor(px); if (m_px < 1) m_px = 1;
    }
	OSL_FORCEINLINE OSL_HOSTDEVICE HashVectorPeriodic (float px, float py) {
        m_px = OIIO::ifloor(px); if (m_px < 1) m_px = 1;
        m_py = OIIO::ifloor(py); if (m_py < 1) m_py = 1;
    }
	OSL_FORCEINLINE OSL_HOSTDEVICE HashVectorPeriodic (float px, float py, float pz) {
        m_px = OIIO::ifloor(px); if (m_px < 1) m_px = 1;
        m_py = OIIO::ifloor(py); if (m_py < 1) m_py = 1;
        m_pz = OIIO::ifloor(pz); if (m_pz < 1) m_pz = 1;
    }
	OSL_FORCEINLINE OSL_HOSTDEVICE HashVectorPeriodic (float px, float py, float pz, float pw) {
        m_px = OIIO::ifloor(px); if (m_px < 1) m_px = 1;
        m_py = OIIO::ifloor(py); if (m_py < 1) m_py = 1;
        m_pz = OIIO::ifloor(pz); if (m_pz < 1) m_pz = 1;
        m_pw = OIIO::ifloor(pw); if (m_pw < 1) m_pw = 1;
    }

    int m_px, m_py, m_pz, m_pw;

    OSL_FORCEINLINE OSL_HOSTDEVICE HashScalarPeriodic convertToScalar() const {
        HashScalarPeriodic r;
        r.m_px = m_px;
        r.m_py = m_py;
        r.m_pz = m_pz;
        r.m_pw = m_pw;
        return r;
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE Vec3i operator() (int x) const {
        return sliceup(inthash(static_cast<unsigned int>(imod (x, m_px))));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE Vec3i operator() (int x, int y) const {
        return sliceup(inthash(static_cast<unsigned int>(imod (x, m_px)),
        		               static_cast<unsigned int>(imod (y, m_py))));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE Vec3i operator() (int x, int y, int z) const {
        return sliceup(inthash(static_cast<unsigned int>(imod (x, m_px)),
        		               static_cast<unsigned int>(imod (y, m_py)),
							   static_cast<unsigned int>(imod (z, m_pz))));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE Vec3i operator() (int x, int y, int z, int w) const {
        return sliceup(inthash(static_cast<unsigned int>(imod (x, m_px)),
        		               static_cast<unsigned int>(imod (y, m_py)),
							   static_cast<unsigned int>(imod (z, m_pz)),
							   static_cast<unsigned int>(imod (w, m_pw))));
    }

#ifndef __CUDA_ARCH__
    // Vector hash of 4 3D points at once
    void operator() (int4 *result, const int4& x, const int4& y) const {
        int4 h = inthash_simd (imod(x,m_px), imod(y,m_py));
        result[0] = (h        ) & 0xFF;
        result[1] = (srl(h,8 )) & 0xFF;
        result[2] = (srl(h,16)) & 0xFF;
    }

    // Vector hash of 4 3D points at once
    void operator() (int4 *result, const int4& x, const int4& y, const int4& z) const {
        int4 h = inthash_simd (imod(x,m_px), imod(y,m_py), imod(z,m_pz));
        result[0] = (h        ) & 0xFF;
        result[1] = (srl(h,8 )) & 0xFF;
        result[2] = (srl(h,16)) & 0xFF;
    }

    // Vector hash of 4 4D points at once
    void operator() (int4 *result, const int4& x, const int4& y, const int4& z, const int4& w) const {
        int4 h = inthash_simd (imod(x,m_px), imod(y,m_py), imod(z,m_pz), imod(w,m_pw));
        result[0] = (h        ) & 0xFF;
        result[1] = (srl(h,8 )) & 0xFF;
        result[2] = (srl(h,16)) & 0xFF;
    }
#endif
};

// Code Generation Policies
// main intent is to allow top level classes to share implementation and carry
// the policy down into helper functions who might diverge in implementation
// or conditionally emit different code paths
struct CGDefault
{
    static constexpr bool allowSIMD = true;
};

struct CGScalar
{
    static constexpr bool allowSIMD = false;
};

template <typename CGPolicyT = CGDefault, typename V, typename H, typename T>
OSL_FORCEINLINE OSL_HOSTDEVICE void perlin (V& result, H& hash, const T &x) {
    int X; T fx = floorfrac(x, &X);
    T u = fade(fx);

    auto lerp_result = OIIO::lerp(grad (hash (X  ), fx     ),
                        grad (hash (X+1), fx-1.0f), u);
    result = scale1 (lerp_result);
}

template <typename CGPolicyT = CGDefault, typename H>
OSL_FORCEINLINE OSL_HOSTDEVICE void perlin (float &result, const H &hash, const float &x, const float &y)
{
#if OIIO_SIMD
    if (CGPolicyT::allowSIMD)
    {
    int4 XY;
    float4 fxy = floorfrac (float4(x,y,0.0f), &XY);
    float4 uv = fade(fxy);  // Note: will be (fade(fx), fade(fy), 0, 0)

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously.
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = OIIO::simd::shuffle<0>(XY) + (*(int4*)i0101);
    int4 cornery = OIIO::simd::shuffle<1>(XY) + (*(int4*)i0011);
    int4 corner_hash = hash (cornerx, cornery);
    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    float4 remainderx = OIIO::simd::shuffle<0>(fxy) - (*(float4*)f0101);
    float4 remaindery = OIIO::simd::shuffle<1>(fxy) - (*(float4*)f0011);
    float4 corner_grad = grad (corner_hash, remainderx, remaindery);
    result = scale2 (bilerp (corner_grad, uv[0], uv[1]));

    } else
#endif
    {
    // ORIGINAL, non-SIMD
    int X; float fx = floorfrac(x, &X);
    int Y; float fy = floorfrac(y, &Y);

    float u = fade(fx);
    float v = fade(fy);

    float bilerp_result = OIIO::bilerp (grad (hash (X  , Y  ), fx     , fy     ),
                           grad (hash (X+1, Y  ), fx-1.0f, fy     ),
                           grad (hash (X  , Y+1), fx     , fy-1.0f),
                           grad (hash (X+1, Y+1), fx-1.0f, fy-1.0f), u, v);
    result = scale2 (bilerp_result);
    }
}

template <typename CGPolicyT = CGDefault, typename H>
OSL_FORCEINLINE OSL_HOSTDEVICE void perlin (float &result, const H &hash,
                    const float &x, const float &y, const float &z)
{
#if OIIO_SIMD
    if (CGPolicyT::allowSIMD)
    {
#if 0
    // You'd think it would be faster to do the floorfrac in parallel, but
    // according to my timings, it is not. I don't understand exactly why.
    int4 XYZ;
    float4 fxyz = floorfrac (float4(x,y,z), &XYZ);
    float4 uvw = fade (fxyz);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = shuffle<0>(XYZ) + (*(int4*)i0101);
    int4 cornery = shuffle<1>(XYZ) + (*(int4*)i0011);
    int4 cornerz = shuffle<2>(XYZ);
#else
    int X; float fx = floorfrac(x, &X);
    int Y; float fy = floorfrac(y, &Y);
    int Z; float fz = floorfrac(z, &Z);
    float4 fxyz (fx, fy, fz); // = floorfrac (xyz, &XYZ);
    float4 uvw = fade (fxyz);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = X + (*(int4*)i0101);
    int4 cornery = Y + (*(int4*)i0011);
    int4 cornerz = Z;
#endif
    int4 corner_hash_z0 = hash (cornerx, cornery, cornerz);
    int4 corner_hash_z1 = hash (cornerx, cornery, cornerz+int4::One());

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    float4 remainderx = OIIO::simd::shuffle<0>(fxyz) - (*(float4*)f0101);
    float4 remaindery = OIIO::simd::shuffle<1>(fxyz) - (*(float4*)f0011);
    float4 remainderz = OIIO::simd::shuffle<2>(fxyz);
    float4 corner_grad_z0 = grad (corner_hash_z0, remainderx, remaindery, remainderz);
    float4 corner_grad_z1 = grad (corner_hash_z1, remainderx, remaindery, remainderz-float4::One());

    result = scale3 (trilerp (corner_grad_z0, corner_grad_z1, uvw));
    } else
#endif
    {
    // ORIGINAL -- non-SIMD
    int X; float fx = floorfrac(x, &X);
    int Y; float fy = floorfrac(y, &Y);
    int Z; float fz = floorfrac(z, &Z);
    float u = fade(fx);
    float v = fade(fy);
    float w = fade(fz);
    float trilerp_result = OIIO::trilerp (grad (hash (X  , Y  , Z  ), fx     , fy     , fz     ),
                            grad (hash (X+1, Y  , Z  ), fx-1.0f, fy     , fz     ),
                            grad (hash (X  , Y+1, Z  ), fx     , fy-1.0f, fz     ),
                            grad (hash (X+1, Y+1, Z  ), fx-1.0f, fy-1.0f, fz     ),
                            grad (hash (X  , Y  , Z+1), fx     , fy     , fz-1.0f),
                            grad (hash (X+1, Y  , Z+1), fx-1.0f, fy     , fz-1.0f),
                            grad (hash (X  , Y+1, Z+1), fx     , fy-1.0f, fz-1.0f),
                            grad (hash (X+1, Y+1, Z+1), fx-1.0f, fy-1.0f, fz-1.0f),
                            u, v, w);
    result = scale3 (trilerp_result);
    }
}



template <typename CGPolicyT = CGDefault, typename H>
OSL_FORCEINLINE OSL_HOSTDEVICE void perlin (float &result, const H &hash,
                    const float &x, const float &y, const float &z, const float &w)
{
#if OIIO_SIMD
    if (CGPolicyT::allowSIMD)
    {

    int4 XYZW;
    float4 fxyzw = floorfrac (float4(x,y,z,w), &XYZW);
    float4 uvts = fade (fxyzw);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = OIIO::simd::shuffle<0>(XYZW) + (*(int4*)i0101);
    int4 cornery = OIIO::simd::shuffle<1>(XYZW) + (*(int4*)i0011);
    int4 cornerz = OIIO::simd::shuffle<2>(XYZW);
    int4 cornerz1 = cornerz + int4::One();
    int4 cornerw = OIIO::simd::shuffle<3>(XYZW);

    int4 corner_hash_z0 = hash (cornerx, cornery, cornerz,  cornerw);
    int4 corner_hash_z1 = hash (cornerx, cornery, cornerz1, cornerw);
    int4 cornerw1 = cornerw + int4::One();
    int4 corner_hash_z2 = hash (cornerx, cornery, cornerz,  cornerw1);
    int4 corner_hash_z3 = hash (cornerx, cornery, cornerz1, cornerw1);

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    float4 remainderx = OIIO::simd::shuffle<0>(fxyzw) - (*(float4*)f0101);
    float4 remaindery = OIIO::simd::shuffle<1>(fxyzw) - (*(float4*)f0011);
    float4 remainderz = OIIO::simd::shuffle<2>(fxyzw);
    float4 remainderz1 = remainderz - float4::One();
    float4 remainderw = OIIO::simd::shuffle<3>(fxyzw);
    float4 corner_grad_z0 = grad (corner_hash_z0, remainderx, remaindery, remainderz,  remainderw);
    float4 corner_grad_z1 = grad (corner_hash_z1, remainderx, remaindery, remainderz1, remainderw);
    float4 remainderw1 = remainderw - float4::One();
    float4 corner_grad_z2 = grad (corner_hash_z2, remainderx, remaindery, remainderz,  remainderw1);
    float4 corner_grad_z3 = grad (corner_hash_z3, remainderx, remaindery, remainderz1, remainderw1);

    result = scale4 (OIIO::lerp (trilerp (corner_grad_z0, corner_grad_z1, uvts),
                                 trilerp (corner_grad_z2, corner_grad_z3, uvts),
                                 OIIO::simd::extract<3>(uvts)));
    } else
#endif
    {
    // ORIGINAL -- non-SIMD
    int X; float fx = floorfrac(x, &X);
    int Y; float fy = floorfrac(y, &Y);
    int Z; float fz = floorfrac(z, &Z);
    int W; float fw = floorfrac(w, &W);

    float u = fade(fx);
    float v = fade(fy);
    float t = fade(fz);
    float s = fade(fw);

    float lerp_result = OIIO::lerp (
               OIIO::trilerp (grad (hash (X  , Y  , Z  , W  ), fx     , fy     , fz     , fw     ),
                              grad (hash (X+1, Y  , Z  , W  ), fx-1.0f, fy     , fz     , fw     ),
                              grad (hash (X  , Y+1, Z  , W  ), fx     , fy-1.0f, fz     , fw     ),
                              grad (hash (X+1, Y+1, Z  , W  ), fx-1.0f, fy-1.0f, fz     , fw     ),
                              grad (hash (X  , Y  , Z+1, W  ), fx     , fy     , fz-1.0f, fw     ),
                              grad (hash (X+1, Y  , Z+1, W  ), fx-1.0f, fy     , fz-1.0f, fw     ),
                              grad (hash (X  , Y+1, Z+1, W  ), fx     , fy-1.0f, fz-1.0f, fw     ),
                              grad (hash (X+1, Y+1, Z+1, W  ), fx-1.0f, fy-1.0f, fz-1.0f, fw     ),
                              u, v, t),
               OIIO::trilerp (grad (hash (X  , Y  , Z  , W+1), fx     , fy     , fz     , fw-1.0f),
                              grad (hash (X+1, Y  , Z  , W+1), fx-1.0f, fy     , fz     , fw-1.0f),
                              grad (hash (X  , Y+1, Z  , W+1), fx     , fy-1.0f, fz     , fw-1.0f),
                              grad (hash (X+1, Y+1, Z  , W+1), fx-1.0f, fy-1.0f, fz     , fw-1.0f),
                              grad (hash (X  , Y  , Z+1, W+1), fx     , fy     , fz-1.0f, fw-1.0f),
                              grad (hash (X+1, Y  , Z+1, W+1), fx-1.0f, fy     , fz-1.0f, fw-1.0f),
                              grad (hash (X  , Y+1, Z+1, W+1), fx     , fy-1.0f, fz-1.0f, fw-1.0f),
                              grad (hash (X+1, Y+1, Z+1, W+1), fx-1.0f, fy-1.0f, fz-1.0f, fw-1.0f),
                              u, v, t),
               s);
    result = scale4 (lerp_result);
    }
}

template <typename CGPolicyT = CGDefault, typename H>
OSL_FORCEINLINE OSL_HOSTDEVICE void perlin (Dual2<float> &result, const H &hash,
                    const Dual2<float> &x, const Dual2<float> &y)
{
#if OIIO_SIMD
    if (CGPolicyT::allowSIMD)
    {
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    Dual2<float4> fxy (float4(fx.val(), fy.val(), 0.0f),
                       float4(fx.dx(), fy.dx(), 0.0f),
                       float4(fx.dy(), fy.dy(), 0.0f));
    Dual2<float4> uv = fade (fxy);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously.
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = X + (*(int4*)i0101);
    int4 cornery = Y + (*(int4*)i0011);
    int4 corner_hash = hash (cornerx, cornery);

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    Dual2<float4> remainderx = shuffle<0>(fxy) - (*(float4*)f0101);
    Dual2<float4> remaindery = shuffle<1>(fxy) - (*(float4*)f0011);
    Dual2<float4> corner_grad = grad (corner_hash, remainderx, remaindery);

    result = scale2 (bilerp (corner_grad, uv));
    } else
#endif
    {
    // Non-SIMD case
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    Dual2<float> u = fade(fx);
    Dual2<float> v = fade(fy);
    auto bilerp_result = OIIO::bilerp (grad (hash (X  , Y  ), fx     , fy     ),
                           grad (hash (X+1, Y  ), fx-1.0f, fy     ),
                           grad (hash (X  , Y+1), fx     , fy-1.0f),
                           grad (hash (X+1, Y+1), fx-1.0f, fy-1.0f), u, v);
    result = scale2 (bilerp_result);
    }
}

template <typename CGPolicyT = CGDefault, typename H>
OSL_FORCEINLINE OSL_HOSTDEVICE void perlin (Dual2<float> &result, const H &hash,
                    const Dual2<float> &x, const Dual2<float> &y, const Dual2<float> &z)
{
#if OIIO_SIMD
    if (CGPolicyT::allowSIMD)
    {
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    int Z; Dual2<float> fz = floorfrac(z, &Z);
    Dual2<float4> fxyz (float4(fx.val(), fy.val(), fz.val()),
                        float4(fx.dx(), fy.dx(), fz.dx()),
                        float4(fx.dy(), fy.dy(), fz.dy()));
    Dual2<float4> uvw = fade (fxyz);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = X + (*(int4*)i0101);
    int4 cornery = Y + (*(int4*)i0011);
    int4 cornerz = Z;
    int4 corner_hash_z0 = hash (cornerx, cornery, cornerz);
    int4 corner_hash_z1 = hash (cornerx, cornery, cornerz+int4::One());

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    Dual2<float4> remainderx = shuffle<0>(fxyz) - (*(float4*)f0101);
    Dual2<float4> remaindery = shuffle<1>(fxyz) - (*(float4*)f0011);
    Dual2<float4> remainderz = shuffle<2>(fxyz);

    Dual2<float4> corner_grad_z0 = grad (corner_hash_z0, remainderx, remaindery, remainderz);
    Dual2<float4> corner_grad_z1 = grad (corner_hash_z1, remainderx, remaindery, remainderz-float4::One());

    // Interpolate along the z axis first
    Dual2<float4> xy = OIIO::lerp (corner_grad_z0, corner_grad_z1, shuffle<2>(uvw));
    // Interpolate along x axis
    Dual2<float4> xx = OIIO::lerp (xy, shuffle<1,1,3,3>(xy), shuffle<0>(uvw));
    // interpolate along y axis
    result = scale3 (extract<0>(OIIO::lerp (xx,shuffle<2>(xx), shuffle<1>(uvw))));
    } else
#endif
    {
    // Non-SIMD case
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    int Z; Dual2<float> fz = floorfrac(z, &Z);
    Dual2<float> u = fade(fx);
    Dual2<float> v = fade(fy);
    Dual2<float> w = fade(fz);
    auto tril_result = OIIO::trilerp (grad (hash (X  , Y  , Z  ), fx     , fy     , fz     ),
                            grad (hash (X+1, Y  , Z  ), fx-1.0f, fy     , fz     ),
                            grad (hash (X  , Y+1, Z  ), fx     , fy-1.0f, fz     ),
                            grad (hash (X+1, Y+1, Z  ), fx-1.0f, fy-1.0f, fz     ),
                            grad (hash (X  , Y  , Z+1), fx     , fy     , fz-1.0f),
                            grad (hash (X+1, Y  , Z+1), fx-1.0f, fy     , fz-1.0f),
                            grad (hash (X  , Y+1, Z+1), fx     , fy-1.0f, fz-1.0f),
                            grad (hash (X+1, Y+1, Z+1), fx-1.0f, fy-1.0f, fz-1.0f),
                            u, v, w);
    result = scale3 (tril_result);
    }
}

template <typename CGPolicyT = CGDefault, typename H>
OSL_FORCEINLINE OSL_HOSTDEVICE void perlin (Dual2<float> &result, const H &hash,
                    const Dual2<float> &x, const Dual2<float> &y,
                    const Dual2<float> &z, const Dual2<float> &w)
{
#if OIIO_SIMD
    if (CGPolicyT::allowSIMD)
    {
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    int Z; Dual2<float> fz = floorfrac(z, &Z);
    int W; Dual2<float> fw = floorfrac(w, &W);
    Dual2<float4> fxyzw (float4(fx.val(), fy.val(), fz.val(), fw.val()),
                         float4(fx.dx (), fy.dx (), fz.dx (), fw.dx ()),
                         float4(fx.dy (), fy.dy (), fz.dy (), fw.dy ()));
    Dual2<float4> uvts = fade (fxyzw);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = X + (*(int4*)i0101);
    int4 cornery = Y + (*(int4*)i0011);
    int4 cornerz = Z;
    int4 cornerz1 = Z + int4::One();
    int4 cornerw = W;
    int4 cornerw1 = W + int4::One();
    int4 corner_hash_z0 = hash (cornerx, cornery, cornerz,  cornerw);
    int4 corner_hash_z1 = hash (cornerx, cornery, cornerz1, cornerw);
    int4 corner_hash_z2 = hash (cornerx, cornery, cornerz,  cornerw1);
    int4 corner_hash_z3 = hash (cornerx, cornery, cornerz1, cornerw1);

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    Dual2<float4> remainderx  = shuffle<0>(fxyzw) - (*(float4*)f0101);
    Dual2<float4> remaindery  = shuffle<1>(fxyzw) - (*(float4*)f0011);
    Dual2<float4> remainderz  = shuffle<2>(fxyzw);
    Dual2<float4> remainderz1 = remainderz - float4::One();
    Dual2<float4> remainderw  = shuffle<3>(fxyzw);
    Dual2<float4> remainderw1 = remainderw - float4::One();

    Dual2<float4> corner_grad_z0 = grad (corner_hash_z0, remainderx, remaindery, remainderz,  remainderw);
    Dual2<float4> corner_grad_z1 = grad (corner_hash_z1, remainderx, remaindery, remainderz1, remainderw);
    Dual2<float4> corner_grad_z2 = grad (corner_hash_z2, remainderx, remaindery, remainderz,  remainderw1);
    Dual2<float4> corner_grad_z3 = grad (corner_hash_z3, remainderx, remaindery, remainderz1, remainderw1);

    // Interpolate along the w axis first
    Dual2<float4> xyz0 = OIIO::lerp (corner_grad_z0, corner_grad_z2, shuffle<3>(uvts));
    Dual2<float4> xyz1 = OIIO::lerp (corner_grad_z1, corner_grad_z3, shuffle<3>(uvts));
    Dual2<float4> xy = OIIO::lerp (xyz0, xyz1, shuffle<2>(uvts));
    // Interpolate along x axis
    Dual2<float4> xx = OIIO::lerp (xy, shuffle<1,1,3,3>(xy), shuffle<0>(uvts));
    // interpolate along y axis
    result = scale4 (extract<0>(OIIO::lerp (xx,shuffle<2>(xx), shuffle<1>(uvts))));
    } else
#endif
    {
    // ORIGINAL -- non-SIMD
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    int Z; Dual2<float> fz = floorfrac(z, &Z);
    int W; Dual2<float> fw = floorfrac(w, &W);

    Dual2<float> u = fade(fx);
    Dual2<float> v = fade(fy);
    Dual2<float> t = fade(fz);
    Dual2<float> s = fade(fw);

    auto lerp_result = OIIO::lerp (
               OIIO::trilerp (grad (hash (X  , Y  , Z  , W  ), fx     , fy     , fz     , fw     ),
                              grad (hash (X+1, Y  , Z  , W  ), fx-1.0f, fy     , fz     , fw     ),
                              grad (hash (X  , Y+1, Z  , W  ), fx     , fy-1.0f, fz     , fw     ),
                              grad (hash (X+1, Y+1, Z  , W  ), fx-1.0f, fy-1.0f, fz     , fw     ),
                              grad (hash (X  , Y  , Z+1, W  ), fx     , fy     , fz-1.0f, fw     ),
                              grad (hash (X+1, Y  , Z+1, W  ), fx-1.0f, fy     , fz-1.0f, fw     ),
                              grad (hash (X  , Y+1, Z+1, W  ), fx     , fy-1.0f, fz-1.0f, fw     ),
                              grad (hash (X+1, Y+1, Z+1, W  ), fx-1.0f, fy-1.0f, fz-1.0f, fw     ),
                              u, v, t),
               OIIO::trilerp (grad (hash (X  , Y  , Z  , W+1), fx     , fy     , fz     , fw-1.0f),
                              grad (hash (X+1, Y  , Z  , W+1), fx-1.0f, fy     , fz     , fw-1.0f),
                              grad (hash (X  , Y+1, Z  , W+1), fx     , fy-1.0f, fz     , fw-1.0f),
                              grad (hash (X+1, Y+1, Z  , W+1), fx-1.0f, fy-1.0f, fz     , fw-1.0f),
                              grad (hash (X  , Y  , Z+1, W+1), fx     , fy     , fz-1.0f, fw-1.0f),
                              grad (hash (X+1, Y  , Z+1, W+1), fx-1.0f, fy     , fz-1.0f, fw-1.0f),
                              grad (hash (X  , Y+1, Z+1, W+1), fx     , fy-1.0f, fz-1.0f, fw-1.0f),
                              grad (hash (X+1, Y+1, Z+1, W+1), fx-1.0f, fy-1.0f, fz-1.0f, fw-1.0f),
                              u, v, t),
               s);
    result = scale4 (lerp_result);
    }
}


template <typename CGPolicyT = CGDefault, typename H>
OSL_FORCEINLINE OSL_HOSTDEVICE void perlin (Vec3 &result, const H &hash,
                    const float &x, const float &y)
{
#if OIIO_SIMD
    if (CGPolicyT::allowSIMD)
    {
    int4 XYZ;
    float4 fxyz = floorfrac (float4(x,y,0), &XYZ);
    float4 uv = fade (fxyz);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = OIIO::simd::shuffle<0>(XYZ) + (*(int4*)i0101);
    int4 cornery = OIIO::simd::shuffle<1>(XYZ) + (*(int4*)i0011);

    // We actually derive 3 hashes (one for each output dimension) for each
    // corner.
    int4 corner_hash[3];
    hash (corner_hash, cornerx, cornery);

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    float4 remainderx = OIIO::simd::shuffle<0>(fxyz) - (*(float4*)f0101);
    float4 remaindery = OIIO::simd::shuffle<1>(fxyz) - (*(float4*)f0011);
    float result_comp[3];
    for (int i = 0; i < 3; ++i) {
        float4 corner_grad = grad (corner_hash[i], remainderx, remaindery);
        // Do the bilinear interpolation with SIMD. Here's the fastest way
        // I've found to do it.
        float4 xx = OIIO::lerp (corner_grad, OIIO::simd::shuffle<1,1,3,3>(corner_grad), uv[0]);
        result_comp[i] = scale2 (OIIO::simd::extract<0>(OIIO::lerp (xx,OIIO::simd::shuffle<2>(xx), uv[1])));
    }
    result = Vec3(result_comp[0], result_comp[1], result_comp[2]);
    } else
#endif
    {
    // Non-SIMD case
    typedef float T;
    int X; T fx = floorfrac(x, &X);
    int Y; T fy = floorfrac(y, &Y);
    T u = fade(fx);
    T v = fade(fy);
    auto bil_result = OIIO::bilerp (grad (hash (X  , Y  ), fx     , fy     ),
                           grad (hash (X+1, Y  ), fx-1.0f, fy     ),
                           grad (hash (X  , Y+1), fx     , fy-1.0f),
                           grad (hash (X+1, Y+1), fx-1.0f, fy-1.0f), u, v);
    result = scale2 (bil_result);
    }
}



template <typename CGPolicyT = CGDefault, typename H>
OSL_FORCEINLINE OSL_HOSTDEVICE void perlin (Vec3 &result, const H &hash,
                    const float &x, const float &y, const float &z)
{
#if OIIO_SIMD
    if (CGPolicyT::allowSIMD)
    {
#if 0
    // You'd think it would be faster to do the floorfrac in parallel, but
    // according to my timings, it is not. Come back and understand why.
    int4 XYZ;
    float4 fxyz = floorfrac (float4(x,y,z), &XYZ);
    float4 uvw = fade (fxyz);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    int4 cornerx = shuffle<0>(XYZ) + int4(0,1,0,1);
    int4 cornery = shuffle<1>(XYZ) + int4(0,0,1,1);
    int4 cornerz = shuffle<2>(XYZ);
#else
    int X; float fx = floorfrac(x, &X);
    int Y; float fy = floorfrac(y, &Y);
    int Z; float fz = floorfrac(z, &Z);
    float4 fxyz (fx, fy, fz); // = floorfrac (xyz, &XYZ);
    float4 uvw = fade (fxyz);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = X + (*(int4*)i0101);
    int4 cornery = Y + (*(int4*)i0011);
    int4 cornerz = Z;
#endif

    // We actually derive 3 hashes (one for each output dimension) for each
    // corner.
    int4 corner_hash_z0[3], corner_hash_z1[3];
    hash (corner_hash_z0, cornerx, cornery, cornerz);
    hash (corner_hash_z1, cornerx, cornery, cornerz+int4::One());

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    float4 remainderx = OIIO::simd::shuffle<0>(fxyz) - (*(float4*)f0101);
    float4 remaindery = OIIO::simd::shuffle<1>(fxyz) - (*(float4*)f0011);
    float4 remainderz0 = OIIO::simd::shuffle<2>(fxyz);
    float4 remainderz1 = OIIO::simd::shuffle<2>(fxyz) - float4::One();
    float result_comp[3];
    for (int i = 0; i < 3; ++i) {
        float4 corner_grad_z0 = grad (corner_hash_z0[i], remainderx, remaindery, remainderz0);
        float4 corner_grad_z1 = grad (corner_hash_z1[i], remainderx, remaindery, remainderz1);

        // Interpolate along the z axis first
        float4 xy = OIIO::lerp (corner_grad_z0, corner_grad_z1, OIIO::simd::shuffle<2>(uvw));
        // Interpolate along x axis
        float4 xx = OIIO::lerp (xy, OIIO::simd::shuffle<1,1,3,3>(xy), OIIO::simd::shuffle<0>(uvw));
        // interpolate along y axis
        result_comp[i] = scale3 (OIIO::simd::extract<0>(OIIO::lerp (xx,OIIO::simd::shuffle<2>(xx), OIIO::simd::shuffle<1>(uvw))));
    }
    result = Vec3(result_comp[0],result_comp[1],result_comp[2]);
    } else
#endif
    {
    // ORIGINAL -- non-SIMD
    int X; float fx = floorfrac(x, &X);
    int Y; float fy = floorfrac(y, &Y);
    int Z; float fz = floorfrac(z, &Z);
    float u = fade(fx);
    float v = fade(fy);
    float w = fade(fz);

    // A.W. the OIIO_SIMD above differs from the original results
    // so we are re-implementing the non-SIMD version to match below
#if 0
    result = OIIO::trilerp (grad (hash (X  , Y  , Z  ), fx     , fy     , fz      ),
                            grad (hash (X+1, Y  , Z  ), fx-1.0f, fy     , fz      ),
                            grad (hash (X  , Y+1, Z  ), fx     , fy-1.0f, fz      ),
                            grad (hash (X+1, Y+1, Z  ), fx-1.0f, fy-1.0f, fz      ),
                            grad (hash (X  , Y  , Z+1), fx     , fy     , fz-1.0f ),
                            grad (hash (X+1, Y  , Z+1), fx-1.0f, fy     , fz-1.0f ),
                            grad (hash (X  , Y+1, Z+1), fx     , fy-1.0f, fz-1.0f ),
                            grad (hash (X+1, Y+1, Z+1), fx-1.0f, fy-1.0f, fz-1.0f ),
                            u, v, w);
    result = scale3 (result);
#else
    static_assert(std::is_same<Vec3i, decltype(hash(X, Y, Z))>::value, "This re-implementation was developed for Hashs returning a vector type only");
    // Want to avoid repeating the same hash work 3 times in a row
    // so rather than executing the vector version, we will use HashScalar
    // and directly mask its results on a per component basis
    auto hash_scalar = hash.convertToScalar();
    int h000 = hash_scalar (X  , Y  , Z  );
    int h100 = hash_scalar (X+1, Y  , Z  );
    int h010 = hash_scalar (X  , Y+1, Z  );
    int h110 = hash_scalar (X+1, Y+1, Z  );
    int h001 = hash_scalar (X  , Y  , Z+1);
    int h101 = hash_scalar (X+1, Y  , Z+1);
    int h011 = hash_scalar (X  , Y+1, Z+1);
    int h111 = hash_scalar (X+1, Y+1, Z+1);

    // We are mimicking the OIIO_SSE behavior
    //      result[0] = (h        ) & 0xFF;
    //      result[1] = (srl(h,8 )) & 0xFF;
    //      result[2] = (srl(h,16)) & 0xFF;
    // skipping masking the 0th version, perhaps that is a mistake

    float result0 = OIIO::trilerp (grad (h000, fx     , fy     , fz      ),
                            grad (h100, fx-1.0f, fy     , fz      ),
                            grad (h010, fx     , fy-1.0f, fz      ),
                            grad (h110, fx-1.0f, fy-1.0f, fz      ),
                            grad (h001, fx     , fy     , fz-1.0f ),
                            grad (h101, fx-1.0f, fy     , fz-1.0f ),
                            grad (h011, fx     , fy-1.0f, fz-1.0f ),
                            grad (h111, fx-1.0f, fy-1.0f, fz-1.0f ),
                            u, v, w);

    float result1 = OIIO::trilerp (
        grad ((h000>>8) & 0xFF, fx     , fy     , fz      ),
        grad ((h100>>8) & 0xFF, fx-1.0f, fy     , fz      ),
        grad ((h010>>8) & 0xFF, fx     , fy-1.0f, fz      ),
        grad ((h110>>8) & 0xFF, fx-1.0f, fy-1.0f, fz      ),
        grad ((h001>>8) & 0xFF, fx     , fy     , fz-1.0f ),
        grad ((h101>>8) & 0xFF, fx-1.0f, fy     , fz-1.0f ),
        grad ((h011>>8) & 0xFF, fx     , fy-1.0f, fz-1.0f ),
        grad ((h111>>8) & 0xFF, fx-1.0f, fy-1.0f, fz-1.0f ),
        u, v, w);

    float result2 = OIIO::trilerp (
        grad ((h000>>16) & 0xFF, fx     , fy     , fz      ),
        grad ((h100>>16) & 0xFF, fx-1.0f, fy     , fz      ),
        grad ((h010>>16) & 0xFF, fx     , fy-1.0f, fz      ),
        grad ((h110>>16) & 0xFF, fx-1.0f, fy-1.0f, fz      ),
        grad ((h001>>16) & 0xFF, fx     , fy     , fz-1.0f ),
        grad ((h101>>16) & 0xFF, fx-1.0f, fy     , fz-1.0f ),
        grad ((h011>>16) & 0xFF, fx     , fy-1.0f, fz-1.0f ),
        grad ((h111>>16) & 0xFF, fx-1.0f, fy-1.0f, fz-1.0f ),
        u, v, w);

    result = Vec3(scale3 (result0), scale3 (result1), scale3 (result2));
#endif
    }
}

template <typename CGPolicyT = CGDefault, typename H>
OSL_FORCEINLINE OSL_HOSTDEVICE void perlin (Vec3 &result, const H &hash,
                    const float &x, const float &y, const float &z, const float &w)
{
#if OIIO_SIMD
    if (CGPolicyT::allowSIMD)
    {
    int4 XYZW;
    float4 fxyzw = floorfrac (float4(x,y,z,w), &XYZW);
    float4 uvts = fade (fxyzw);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = OIIO::simd::shuffle<0>(XYZW) + (*(int4*)i0101);
    int4 cornery = OIIO::simd::shuffle<1>(XYZW) + (*(int4*)i0011);
    int4 cornerz = OIIO::simd::shuffle<2>(XYZW);
    int4 cornerw = OIIO::simd::shuffle<3>(XYZW);

    // We actually derive 3 hashes (one for each output dimension) for each
    // corner.
    int4 corner_hash_z0[3], corner_hash_z1[3];
    int4 corner_hash_z2[3], corner_hash_z3[3];
    hash (corner_hash_z0, cornerx, cornery, cornerz, cornerw);
    hash (corner_hash_z1, cornerx, cornery, cornerz+int4::One(), cornerw);
    hash (corner_hash_z2, cornerx, cornery, cornerz, cornerw+int4::One());
    hash (corner_hash_z3, cornerx, cornery, cornerz+int4::One(), cornerw+int4::One());

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    float4 remainderx = OIIO::simd::shuffle<0>(fxyzw) - (*(float4*)f0101);
    float4 remaindery = OIIO::simd::shuffle<1>(fxyzw) - (*(float4*)f0011);
    float4 remainderz = OIIO::simd::shuffle<2>(fxyzw);
    float4 remainderw = OIIO::simd::shuffle<3>(fxyzw);
//    float4 remainderz0 = shuffle<2>(fxyz);
//    float4 remainderz1 = shuffle<2>(fxyz) - 1.0f;
    float result_comp[3];
    for (int i = 0; i < 3; ++i) {
        float4 corner_grad_z0 = grad (corner_hash_z0[i], remainderx, remaindery, remainderz, remainderw);
        float4 corner_grad_z1 = grad (corner_hash_z1[i], remainderx, remaindery, remainderz-float4::One(), remainderw);
        float4 corner_grad_z2 = grad (corner_hash_z2[i], remainderx, remaindery, remainderz, remainderw-float4::One());
        float4 corner_grad_z3 = grad (corner_hash_z3[i], remainderx, remaindery, remainderz-float4::One(), remainderw-float4::One());
        result_comp[i] = scale4 (OIIO::lerp (trilerp (corner_grad_z0, corner_grad_z1, uvts),
                                        trilerp (corner_grad_z2, corner_grad_z3, uvts),
                                        OIIO::simd::extract<3>(uvts)));
    }
    result = Vec3(result_comp[0],result_comp[1],result_comp[2]);
    } else
#endif
    {
    // ORIGINAL -- non-SIMD
    int X; float fx = floorfrac(x, &X);
    int Y; float fy = floorfrac(y, &Y);
    int Z; float fz = floorfrac(z, &Z);
    int W; float fw = floorfrac(w, &W);

    float u = fade(fx);
    float v = fade(fy);
    float t = fade(fz);
    float s = fade(fw);

    auto l_result = OIIO::lerp (
               OIIO::trilerp (grad (hash (X  , Y  , Z  , W  ), fx     , fy     , fz     , fw     ),
                              grad (hash (X+1, Y  , Z  , W  ), fx-1.0f, fy     , fz     , fw     ),
                              grad (hash (X  , Y+1, Z  , W  ), fx     , fy-1.0f, fz     , fw     ),
                              grad (hash (X+1, Y+1, Z  , W  ), fx-1.0f, fy-1.0f, fz     , fw     ),
                              grad (hash (X  , Y  , Z+1, W  ), fx     , fy     , fz-1.0f, fw     ),
                              grad (hash (X+1, Y  , Z+1, W  ), fx-1.0f, fy     , fz-1.0f, fw     ),
                              grad (hash (X  , Y+1, Z+1, W  ), fx     , fy-1.0f, fz-1.0f, fw     ),
                              grad (hash (X+1, Y+1, Z+1, W  ), fx-1.0f, fy-1.0f, fz-1.0f, fw     ),
                              u, v, t),
               OIIO::trilerp (grad (hash (X  , Y  , Z  , W+1), fx     , fy     , fz     , fw-1.0f),
                              grad (hash (X+1, Y  , Z  , W+1), fx-1.0f, fy     , fz     , fw-1.0f),
                              grad (hash (X  , Y+1, Z  , W+1), fx     , fy-1.0f, fz     , fw-1.0f),
                              grad (hash (X+1, Y+1, Z  , W+1), fx-1.0f, fy-1.0f, fz     , fw-1.0f),
                              grad (hash (X  , Y  , Z+1, W+1), fx     , fy     , fz-1.0f, fw-1.0f),
                              grad (hash (X+1, Y  , Z+1, W+1), fx-1.0f, fy     , fz-1.0f, fw-1.0f),
                              grad (hash (X  , Y+1, Z+1, W+1), fx     , fy-1.0f, fz-1.0f, fw-1.0f),
                              grad (hash (X+1, Y+1, Z+1, W+1), fx-1.0f, fy-1.0f, fz-1.0f, fw-1.0f),
                              u, v, t),
               s);
    result = scale4 (l_result);
    }
}


template <typename CGPolicyT = CGDefault, typename H>
OSL_FORCEINLINE OSL_HOSTDEVICE void perlin (Dual2<Vec3> &result, const H &hash,
                    const Dual2<float> &x, const Dual2<float> &y)
{
#if OIIO_SIMD
    if (CGPolicyT::allowSIMD)
    {
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    Dual2<float4> fxyz (float4(fx.val(), fy.val(), 0.0f),
                        float4(fx.dx(),  fy.dx(),  0.0f),
                        float4(fx.dy(),  fy.dy(),  0.0f));
    Dual2<float4> uvw = fade (fxyz);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = X + (*(int4*)i0101);
    int4 cornery = Y + (*(int4*)i0011);

    // We actually derive 3 hashes (one for each output dimension) for each
    // corner.
    int4 corner_hash[3];
    hash (corner_hash, cornerx, cornery);

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    Dual2<float4> remainderx = shuffle<0>(fxyz) - (*(float4*)f0101);
    Dual2<float4> remaindery = shuffle<1>(fxyz) - (*(float4*)f0011);
    Dual2<float> r[3];
    for (int i = 0; i < 3; ++i) {
        Dual2<float4> corner_grad = grad (corner_hash[i], remainderx, remaindery);
        // Interpolate along x axis
        Dual2<float4> xx = OIIO::lerp (corner_grad, shuffle<1,1,3,3>(corner_grad), shuffle<0>(uvw));
        // interpolate along y axis
        r[i] = scale2 (extract<0>(OIIO::lerp (xx,shuffle<2>(xx), shuffle<1>(uvw))));
    }
    result.set (Vec3 (r[0].val(), r[1].val(), r[2].val()),
                Vec3 (r[0].dx(),  r[1].dx(),  r[2].dx()),
                Vec3 (r[0].dy(),  r[1].dy(),  r[2].dy()));
    } else
#endif
    {
    // ORIGINAL -- non-SIMD
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    Dual2<float> u = fade(fx);
    Dual2<float> v = fade(fy);
    auto bil_result = OIIO::bilerp (grad (hash (X  , Y  ), fx     , fy     ),
                           grad (hash (X+1, Y  ), fx-1.0f, fy     ),
                           grad (hash (X  , Y+1), fx     , fy-1.0f),
                           grad (hash (X+1, Y+1), fx-1.0f, fy-1.0f),
                           u, v);
    result = scale2 (bil_result);
    }
}


template <typename CGPolicyT = CGDefault, typename H>
OSL_FORCEINLINE OSL_HOSTDEVICE void perlin (Dual2<Vec3> &result, const H &hash,
                    const Dual2<float> &x, const Dual2<float> &y, const Dual2<float> &z)
{
#if OIIO_SIMD
    if (CGPolicyT::allowSIMD)
    {
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    int Z; Dual2<float> fz = floorfrac(z, &Z);
    Dual2<float4> fxyz (float4(fx.val(), fy.val(), fz.val()),
                        float4(fx.dx(),  fy.dx(),  fz.dx()),
                        float4(fx.dy(),  fy.dy(),  fz.dy()));
    Dual2<float4> uvw = fade (fxyz);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = X + (*(int4*)i0101);
    int4 cornery = Y + (*(int4*)i0011);
    int4 cornerz = Z;

    // We actually derive 3 hashes (one for each output dimension) for each
    // corner.
    int4 corner_hash_z0[3], corner_hash_z1[3];
    hash (corner_hash_z0, cornerx, cornery, cornerz);
    hash (corner_hash_z1, cornerx, cornery, cornerz+int4::One());

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    Dual2<float4> remainderx = shuffle<0>(fxyz) - (*(float4*)f0101);
    Dual2<float4> remaindery = shuffle<1>(fxyz) - (*(float4*)f0011);
    Dual2<float4> remainderz0 = shuffle<2>(fxyz);
    Dual2<float4> remainderz1 = shuffle<2>(fxyz) - float4::One();
    Dual2<float> r[3];
    for (int i = 0; i < 3; ++i) {
        Dual2<float4> corner_grad_z0 = grad (corner_hash_z0[i], remainderx, remaindery, remainderz0);
        Dual2<float4> corner_grad_z1 = grad (corner_hash_z1[i], remainderx, remaindery, remainderz1);

        // Interpolate along the z axis first
        Dual2<float4> xy = OIIO::lerp (corner_grad_z0, corner_grad_z1, shuffle<2>(uvw));
        // Interpolate along x axis
        Dual2<float4> xx = OIIO::lerp (xy, shuffle<1,1,3,3>(xy), shuffle<0>(uvw));
        // interpolate along y axis
        r[i] = scale3 (extract<0>(OIIO::lerp (xx,shuffle<2>(xx), shuffle<1>(uvw))));
    }
    result.set (Vec3 (r[0].val(), r[1].val(), r[2].val()),
                Vec3 (r[0].dx(),  r[1].dx(),  r[2].dx()),
                Vec3 (r[0].dy(),  r[1].dy(),  r[2].dy()));
    } else
#endif
    {
    // ORIGINAL -- non-SIMD
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    int Z; Dual2<float> fz = floorfrac(z, &Z);
    Dual2<float> u = fade(fx);
    Dual2<float> v = fade(fy);
    Dual2<float> w = fade(fz);
    auto til_result = OIIO::trilerp (grad (hash (X  , Y  , Z  ), fx     , fy     , fz      ),
                            grad (hash (X+1, Y  , Z  ), fx-1.0f, fy     , fz      ),
                            grad (hash (X  , Y+1, Z  ), fx     , fy-1.0f, fz      ),
                            grad (hash (X+1, Y+1, Z  ), fx-1.0f, fy-1.0f, fz      ),
                            grad (hash (X  , Y  , Z+1), fx     , fy     , fz-1.0f ),
                            grad (hash (X+1, Y  , Z+1), fx-1.0f, fy     , fz-1.0f ),
                            grad (hash (X  , Y+1, Z+1), fx     , fy-1.0f, fz-1.0f ),
                            grad (hash (X+1, Y+1, Z+1), fx-1.0f, fy-1.0f, fz-1.0f ),
                            u, v, w);
    result = scale3 (til_result);
    }
}


template <typename CGPolicyT = CGDefault, typename H>
OSL_FORCEINLINE OSL_HOSTDEVICE void perlin (Dual2<Vec3> &result, const H &hash,
                    const Dual2<float> &x, const Dual2<float> &y,
                    const Dual2<float> &z, const Dual2<float> &w)
{
#if OIIO_SIMD
    if (CGPolicyT::allowSIMD)
    {
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    int Z; Dual2<float> fz = floorfrac(z, &Z);
    int W; Dual2<float> fw = floorfrac(w, &W);
    Dual2<float4> fxyzw (float4(fx.val(), fy.val(), fz.val(), fw.val()),
                         float4(fx.dx (), fy.dx (), fz.dx (), fw.dx ()),
                         float4(fx.dy (), fy.dy (), fz.dy (), fw.dy ()));
    Dual2<float4> uvts = fade (fxyzw);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = X + (*(int4*)i0101);
    int4 cornery = Y + (*(int4*)i0011);
    int4 cornerz = Z;
    int4 cornerz1 = Z + int4::One();
    int4 cornerw = W;
    int4 cornerw1 = W + int4::One();

    // We actually derive 3 hashes (one for each output dimension) for each
    // corner.
    int4 corner_hash_z0[3], corner_hash_z1[3], corner_hash_z2[3], corner_hash_z3[3];
    hash (corner_hash_z0, cornerx, cornery, cornerz,  cornerw);
    hash (corner_hash_z1, cornerx, cornery, cornerz1, cornerw);
    hash (corner_hash_z2, cornerx, cornery, cornerz,  cornerw1);
    hash (corner_hash_z3, cornerx, cornery, cornerz1, cornerw1);

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    Dual2<float4> remainderx  = shuffle<0>(fxyzw) - (*(float4*)f0101);
    Dual2<float4> remaindery  = shuffle<1>(fxyzw) - (*(float4*)f0011);
    Dual2<float4> remainderz  = shuffle<2>(fxyzw);
    Dual2<float4> remainderz1 = remainderz - float4::One();
    Dual2<float4> remainderw  = shuffle<3>(fxyzw);
    Dual2<float4> remainderw1 = remainderw - float4::One();

    Dual2<float> r[3];
    for (int i = 0; i < 3; ++i) {
        Dual2<float4> corner_grad_z0 = grad (corner_hash_z0[i], remainderx, remaindery, remainderz,  remainderw);
        Dual2<float4> corner_grad_z1 = grad (corner_hash_z1[i], remainderx, remaindery, remainderz1, remainderw);
        Dual2<float4> corner_grad_z2 = grad (corner_hash_z2[i], remainderx, remaindery, remainderz,  remainderw1);
        Dual2<float4> corner_grad_z3 = grad (corner_hash_z3[i], remainderx, remaindery, remainderz1, remainderw1);

        // Interpolate along the w axis first
        Dual2<float4> xyz0 = OIIO::lerp (corner_grad_z0, corner_grad_z2, shuffle<3>(uvts));
        Dual2<float4> xyz1 = OIIO::lerp (corner_grad_z1, corner_grad_z3, shuffle<3>(uvts));
        Dual2<float4> xy = OIIO::lerp (xyz0, xyz1, shuffle<2>(uvts));
        // Interpolate along x axis
        Dual2<float4> xx = OIIO::lerp (xy, shuffle<1,1,3,3>(xy), shuffle<0>(uvts));
        // interpolate along y axis
        r[i] = scale4 (extract<0>(OIIO::lerp (xx,shuffle<2>(xx), shuffle<1>(uvts))));

    }
    result.set (Vec3 (r[0].val(), r[1].val(), r[2].val()),
                Vec3 (r[0].dx(),  r[1].dx(),  r[2].dx()),
                Vec3 (r[0].dy(),  r[1].dy(),  r[2].dy()));
    } else
#endif
    {
    // ORIGINAL -- non-SIMD
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    int Z; Dual2<float> fz = floorfrac(z, &Z);
    int W; Dual2<float> fw = floorfrac(w, &W);

    Dual2<float> u = fade(fx);
    Dual2<float> v = fade(fy);
    Dual2<float> t = fade(fz);
    Dual2<float> s = fade(fw);

    // With Dual2<Vec3> data types, a lot of code is generated below
    // which caused some runaway compiler memory consumption when vectorizing
#if !OSL_INTEL_COMPILER
    auto l_result = OIIO::lerp (
               OIIO::trilerp (grad (hash (X  , Y  , Z  , W  ), fx     , fy     , fz     , fw     ),
                              grad (hash (X+1, Y  , Z  , W  ), fx-1.0f, fy     , fz     , fw     ),
                              grad (hash (X  , Y+1, Z  , W  ), fx     , fy-1.0f, fz     , fw     ),
                              grad (hash (X+1, Y+1, Z  , W  ), fx-1.0f, fy-1.0f, fz     , fw     ),
                              grad (hash (X  , Y  , Z+1, W  ), fx     , fy     , fz-1.0f, fw     ),
                              grad (hash (X+1, Y  , Z+1, W  ), fx-1.0f, fy     , fz-1.0f, fw     ),
                              grad (hash (X  , Y+1, Z+1, W  ), fx     , fy-1.0f, fz-1.0f, fw     ),
                              grad (hash (X+1, Y+1, Z+1, W  ), fx-1.0f, fy-1.0f, fz-1.0f, fw     ),
                              u, v, t),
               OIIO::trilerp (grad (hash (X  , Y  , Z  , W+1), fx     , fy     , fz     , fw-1.0f),
                              grad (hash (X+1, Y  , Z  , W+1), fx-1.0f, fy     , fz     , fw-1.0f),
                              grad (hash (X  , Y+1, Z  , W+1), fx     , fy-1.0f, fz     , fw-1.0f),
                              grad (hash (X+1, Y+1, Z  , W+1), fx-1.0f, fy-1.0f, fz     , fw-1.0f),
                              grad (hash (X  , Y  , Z+1, W+1), fx     , fy     , fz-1.0f, fw-1.0f),
                              grad (hash (X+1, Y  , Z+1, W+1), fx-1.0f, fy     , fz-1.0f, fw-1.0f),
                              grad (hash (X  , Y+1, Z+1, W+1), fx     , fy-1.0f, fz-1.0f, fw-1.0f),
                              grad (hash (X+1, Y+1, Z+1, W+1), fx-1.0f, fy-1.0f, fz-1.0f, fw-1.0f),
                              u, v, t),
               s);
#else
    // Use a loop to avoid repeating code gen twice
    Dual2<Vec3> v0, v1;
    // GCC emits -Wmaybe-uninitialized errors for v0,v1.
    // To avoid, GCC uses reference version above

    // Clang doesn't want to vectorize with the vIndex loop
    // To enable vectorization, Clang uses reference version above
    OSL_INTEL_PRAGMA(nounroll_and_jam)
    for(int vIndex=0; vIndex < 2;++vIndex) {
        int vW = W + vIndex;
        Dual2<float> vfw = fw - float(vIndex);

        Dual2<Vec3> vResult = OIIO::trilerp (
            grad (hash (X  , Y  , Z  , vW  ), fx     , fy     , fz     , vfw     ),
            grad (hash (X+1, Y  , Z  , vW  ), fx-1.0f, fy     , fz     , vfw     ),
            grad (hash (X  , Y+1, Z  , vW  ), fx     , fy-1.0f, fz     , vfw     ),
            grad (hash (X+1, Y+1, Z  , vW  ), fx-1.0f, fy-1.0f, fz     , vfw     ),
            grad (hash (X  , Y  , Z+1, vW  ), fx     , fy     , fz-1.0f, vfw     ),
            grad (hash (X+1, Y  , Z+1, vW  ), fx-1.0f, fy     , fz-1.0f, vfw     ),
            grad (hash (X  , Y+1, Z+1, vW  ), fx     , fy-1.0f, fz-1.0f, vfw     ),
            grad (hash (X+1, Y+1, Z+1, vW  ), fx-1.0f, fy-1.0f, fz-1.0f, vfw     ),
            u, v, t);
        // Rather than dynamic indexing array,
        // use masking to store outputs,
        // to better enable SROA (Scalar Replacement of Aggregates) optimizations
        if (vIndex == 0) {
            v0 = vResult;
        } else {
            v1 = vResult;
        }
    }
    auto l_result = OIIO::lerp (v0, v1, s);
#endif

    result = scale4 (l_result);
    }
}



template<typename CGPolicyT = CGDefault>
struct NoiseImpl {
	OSL_FORCEINLINE OSL_HOSTDEVICE NoiseImpl () { }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, float x) const {
        HashScalar h;
        float perlin_result;
        perlin<CGPolicyT>(perlin_result, h, x);
        result = 0.5f * (perlin_result + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, float x, float y) const {
        HashScalar h;
        float perlin_result;
        perlin<CGPolicyT>(perlin_result, h, x, y);
        result = 0.5f * (perlin_result + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p) const {
        HashScalar h;
        float perlin_result;
        perlin<CGPolicyT>(perlin_result, h, p.x, p.y, p.z);
        result = 0.5f * (perlin_result + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, float t) const {
        HashScalar h;
        float perlin_result;
        perlin<CGPolicyT>(perlin_result, h, p.x, p.y, p.z, t);
        result = 0.5f * (perlin_result + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, float x) const {
        HashVector h;
        Vec3 perlin_result;
        perlin<CGPolicyT>(perlin_result, h, x);
        result = 0.5f * (perlin_result + Vec3(1.0f, 1.0f, 1.0f));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float y) const {
        HashVector h;
        Vec3 perlin_result;
        perlin<CGPolicyT>(perlin_result, h, x, y);
        result = 0.5f * (perlin_result + Vec3(1.0f, 1.0f, 1.0f));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p) const {
        HashVector h;
        Vec3 perlin_result;
        perlin<CGPolicyT>(perlin_result, h, p.x, p.y, p.z);
        result = 0.5f * (perlin_result + Vec3(1.0f, 1.0f, 1.0f));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, float t) const {
        HashVector h;
        Vec3 perlin_result;
        perlin<CGPolicyT>(perlin_result, h, p.x, p.y, p.z, t);
        result = 0.5f * (perlin_result + Vec3(1.0f, 1.0f, 1.0f));
    }

    
    // dual versions

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x) const {
        HashScalar h;
        Dual2<float> perlin_result;
        perlin<CGPolicyT>(perlin_result, h, x);
        result = 0.5f * (perlin_result + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x, const Dual2<float> &y) const {
        HashScalar h;
        Dual2<float> perlin_result;
        perlin<CGPolicyT>(perlin_result, h, x, y);
        result = 0.5f * (perlin_result + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p) const {
        HashScalar h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        Dual2<float> perlin_result;
        perlin<CGPolicyT>(perlin_result, h, px, py, pz);
        result = 0.5f * (perlin_result + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const {
        HashScalar h;        
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        Dual2<float> perlin_result;
        perlin<CGPolicyT>(perlin_result, h, px, py, pz, t);
        result = 0.5f * (perlin_result + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x) const {
        HashVector h;
        Dual2<Vec3> perlin_result;
        perlin<CGPolicyT>(perlin_result, h, x);
        result = Vec3(0.5f, 0.5f, 0.5f) * (perlin_result + Vec3(1, 1, 1));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y) const {
        HashVector h;
        Dual2<Vec3> perlin_result;
        perlin<CGPolicyT>(perlin_result, h, x, y);
        result = Vec3(0.5f, 0.5f, 0.5f) * (perlin_result + Vec3(1, 1, 1));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p) const {
        HashVector h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        Dual2<Vec3> perlin_result;
        perlin<CGPolicyT>(perlin_result, h, px, py, pz);
        result = Vec3(0.5f, 0.5f, 0.5f) * (perlin_result + Vec3(1, 1, 1));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const {
        HashVector h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        Dual2<Vec3> perlin_result;
        perlin<CGPolicyT>(perlin_result, h, px, py, pz, t);
        result = Vec3(0.5f, 0.5f, 0.5f) * (perlin_result + Vec3(1, 1, 1));
    }
};

struct Noise : NoiseImpl<CGDefault> {};
// Scalar version of Noise that is SIMD friendly suitable to be
// inlined inside of a SIMD loops
struct NoiseScalar : NoiseImpl<CGScalar> {};


template<typename CGPolicyT = CGDefault>
struct SNoiseImpl {
	OSL_FORCEINLINE OSL_HOSTDEVICE SNoiseImpl () { }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, float x) const {
        HashScalar h;
        perlin<CGPolicyT>(result, h, x);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, float x, float y) const {
        HashScalar h;
        perlin<CGPolicyT>(result, h, x, y);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p) const {
        HashScalar h;
        perlin<CGPolicyT>(result, h, p.x, p.y, p.z);
    }
    
    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, float t) const {
        HashScalar h;
        perlin<CGPolicyT>(result, h, p.x, p.y, p.z, t);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, float x) const {
        HashVector h;
        perlin<CGPolicyT>(result, h, x);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float y) const {
        HashVector h;
        perlin<CGPolicyT>(result, h, x, y);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p) const {
        HashVector h;
        perlin<CGPolicyT>(result, h, p.x, p.y, p.z);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, float t) const {
        HashVector h;
        perlin<CGPolicyT>(result, h, p.x, p.y, p.z, t);
    }


    // dual versions

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x) const {
        HashScalar h;
        perlin<CGPolicyT>(result, h, x);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x, const Dual2<float> &y) const {
        HashScalar h;
        perlin<CGPolicyT>(result, h, x, y);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p) const {
        HashScalar h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin<CGPolicyT>(result, h, px, py, pz);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const {
        HashScalar h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin<CGPolicyT>(result, h, px, py, pz, t);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x) const {
        HashVector h;
        perlin<CGPolicyT>(result, h, x);
    }


    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y) const {
        HashVector h;
        perlin<CGPolicyT>(result, h, x, y);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p) const {
        HashVector h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin<CGPolicyT>(result, h, px, py, pz);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const {
        HashVector h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin<CGPolicyT>(result, h, px, py, pz, t);
    }
};

struct SNoise : SNoiseImpl<CGDefault> {};
// Scalar version of SNoise that is SIMD friendly suitable to be
// inlined inside of a SIMD loops
struct SNoiseScalar : SNoiseImpl<CGScalar> {};



template<typename CGPolicyT = CGDefault>
struct PeriodicNoiseImpl {
	OSL_FORCEINLINE OSL_HOSTDEVICE PeriodicNoiseImpl () { }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, float x, float px) const {
        HashScalarPeriodic h(px);
        perlin<CGPolicyT>(result, h, x);
        result = 0.5f * (result + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, float x, float y, float px, float py) const {
        HashScalarPeriodic h(px, py);
        perlin<CGPolicyT>(result, h, x, y);
        result = 0.5f * (result + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, const Vec3 &pp) const {
        HashScalarPeriodic h(pp.x, pp.y, pp.z);
        perlin<CGPolicyT>(result, h, p.x, p.y, p.z);
        result = 0.5f * (result + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, float t, const Vec3 &pp, float pt) const {
        HashScalarPeriodic h(pp.x, pp.y, pp.z, pt);
        perlin<CGPolicyT>(result, h, p.x, p.y, p.z, t);
        result = 0.5f * (result + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float px) const {
        HashVectorPeriodic h(px);
        perlin<CGPolicyT>(result, h, x);
        result = 0.5f * (result + Vec3(1.0f, 1.0f, 1.0f));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float y, float px, float py) const {
        HashVectorPeriodic h(px, py);
        perlin<CGPolicyT>(result, h, x, y);
        result = 0.5f * (result + Vec3(1.0f, 1.0f, 1.0f));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, const Vec3 &pp) const {
        HashVectorPeriodic h(pp.x, pp.y, pp.z);
        perlin<CGPolicyT>(result, h, p.x, p.y, p.z);
        result = 0.5f * (result + Vec3(1.0f, 1.0f, 1.0f));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, float t, const Vec3 &pp, float pt) const {
        HashVectorPeriodic h(pp.x, pp.y, pp.z, pt);
        perlin<CGPolicyT>(result, h, p.x, p.y, p.z, t);
        result = 0.5f * (result + Vec3(1.0f, 1.0f, 1.0f));
    }

    // dual versions

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x, float px) const {
        HashScalarPeriodic h(px);
        perlin<CGPolicyT>(result, h, x);
        result = 0.5f * (result + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x, const Dual2<float> &y,
            float px, float py) const {
        HashScalarPeriodic h(px, py);
        perlin<CGPolicyT>(result, h, x, y);
        result = 0.5f * (result + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p, const Vec3 &pp) const {
        HashScalarPeriodic h(pp.x, pp.y, pp.z);
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin<CGPolicyT>(result, h, px, py, pz);
        result = 0.5f * (result + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p, const Dual2<float> &t,
            const Vec3 &pp, float pt) const {
        HashScalarPeriodic h(pp.x, pp.y, pp.z, pt);        
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin<CGPolicyT>(result, h, px, py, pz, t);
        result = 0.5f * (result + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x, float px) const {
        HashVectorPeriodic h(px);
        perlin<CGPolicyT>(result, h, x);
        result = Vec3(0.5f, 0.5f, 0.5f) * (result + Vec3(1.0f, 1.0f, 1.0f));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y,
            float px, float py) const {
        HashVectorPeriodic h(px, py);
        perlin<CGPolicyT>(result, h, x, y);
        result = Vec3(0.5f, 0.5f, 0.5f) * (result + Vec3(1.0f, 1.0f, 1.0f));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Vec3 &pp) const {
        HashVectorPeriodic h(pp.x, pp.y, pp.z);
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin<CGPolicyT>(result, h, px, py, pz);
        result = Vec3(0.5f, 0.5f, 0.5f) * (result + Vec3(1.0f, 1.0f, 1.0f));
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t,
            const Vec3 &pp, float pt) const {
        HashVectorPeriodic h(pp.x, pp.y, pp.z, pt);
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin<CGPolicyT>(result, h, px, py, pz, t);
        result = Vec3(0.5f, 0.5f, 0.5f) * (result + Vec3(1.0f, 1.0f, 1.0f));
    }

};

struct PeriodicNoise : PeriodicNoiseImpl<CGDefault> {};
// Scalar version of PeriodicNoise that is SIMD friendly suitable to be
// inlined inside of a SIMD loops
struct PeriodicNoiseScalar : PeriodicNoiseImpl<CGScalar> {};

template<typename CGPolicyT = CGDefault>
struct PeriodicSNoiseImpl {
	OSL_FORCEINLINE OSL_HOSTDEVICE PeriodicSNoiseImpl () { }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, float x, float px) const {
        HashScalarPeriodic h(px);
        perlin<CGPolicyT>(result, h, x);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, float x, float y, float px, float py) const {
        HashScalarPeriodic h(px, py);
        perlin<CGPolicyT>(result, h, x, y);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, const Vec3 &pp) const {
        HashScalarPeriodic h(pp.x, pp.y, pp.z);
        perlin<CGPolicyT>(result, h, p.x, p.y, p.z);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, float t, const Vec3 &pp, float pt) const {
        HashScalarPeriodic h(pp.x, pp.y, pp.z, pt);
        perlin<CGPolicyT>(result, h, p.x, p.y, p.z, t);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float px) const {
        HashVectorPeriodic h(px);
        perlin<CGPolicyT>(result, h, x);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float y, float px, float py) const {
        HashVectorPeriodic h(px, py);
        perlin<CGPolicyT>(result, h, x, y);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, const Vec3 &pp) const {
        HashVectorPeriodic h(pp.x, pp.y, pp.z);
        perlin<CGPolicyT>(result, h, p.x, p.y, p.z);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, float t, const Vec3 &pp, float pt) const {
        HashVectorPeriodic h(pp.x, pp.y, pp.z, pt);
        perlin<CGPolicyT>(result, h, p.x, p.y, p.z, t);
    }

    // dual versions

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x, float px) const {
        HashScalarPeriodic h(px);
        perlin<CGPolicyT>(result, h, x);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x, const Dual2<float> &y,
            float px, float py) const {
        HashScalarPeriodic h(px, py);
        perlin<CGPolicyT>(result, h, x, y);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p, const Vec3 &pp) const {
        HashScalarPeriodic h(pp.x, pp.y, pp.z);
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin<CGPolicyT>(result, h, px, py, pz);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p, const Dual2<float> &t,
            const Vec3 &pp, float pt) const {
        HashScalarPeriodic h(pp.x, pp.y, pp.z, pt);        
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin<CGPolicyT>(result, h, px, py, pz, t);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x, float px) const {
        HashVectorPeriodic h(px);
        perlin<CGPolicyT>(result, h, x);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y,
            float px, float py) const {
        HashVectorPeriodic h(px, py);
        perlin<CGPolicyT>(result, h, x, y);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Vec3 &pp) const {
        HashVectorPeriodic h(pp.x, pp.y, pp.z);
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin<CGPolicyT>(result, h, px, py, pz);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t,
            const Vec3 &pp, float pt) const {
        HashVectorPeriodic h(pp.x, pp.y, pp.z, pt);
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin<CGPolicyT>(result, h, px, py, pz, t);
    }

};

struct PeriodicSNoise : PeriodicSNoiseImpl<CGDefault> {};
// Scalar version of PeriodicSNoise that is SIMD friendly suitable to be
// inlined inside of a SIMD loops
struct PeriodicSNoiseScalar : PeriodicSNoiseImpl<CGScalar> {};


struct SimplexNoise {
    OSL_HOSTDEVICE SimplexNoise () { }

    inline OSL_HOSTDEVICE void operator() (float &result, float x) const {
        result = simplexnoise1 (x);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, float x, float y) const {
        result = simplexnoise2 (x, y);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p) const {
        result = simplexnoise3 (p.x, p.y, p.z);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, float t) const {
        result = simplexnoise4 (p.x, p.y, p.z, t);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, float x) const {
        result.x = simplexnoise1 (x, 0);
        result.y = simplexnoise1 (x, 1);
        result.z = simplexnoise1 (x, 2);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float y) const {
        result.x = simplexnoise2 (x, y, 0);
        result.y = simplexnoise2 (x, y, 1);
        result.z = simplexnoise2 (x, y, 2);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p) const {
        result.x = simplexnoise3 (p.x, p.y, p.z, 0);
        result.y = simplexnoise3 (p.x, p.y, p.z, 1);
        result.z = simplexnoise3 (p.x, p.y, p.z, 2);
    }
    
    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, float t) const {
        result.x = simplexnoise4 (p.x, p.y, p.z, t, 0);
        result.y = simplexnoise4 (p.x, p.y, p.z, t, 1);
        result.z = simplexnoise4 (p.x, p.y, p.z, t, 2);
    }


    // dual versions

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x,
                                           int seed=0) const {
        float r, dndx;
        r = simplexnoise1 (x.val(), seed, &dndx);
        result.set (r, dndx * x.dx(), dndx * x.dy());
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x,
                                           const Dual2<float> &y, int seed=0) const {
        float r, dndx, dndy;
        r = simplexnoise2 (x.val(), y.val(), seed, &dndx, &dndy);
        result.set (r, dndx * x.dx() + dndy * y.dx(),
                       dndx * x.dy() + dndy * y.dy());
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p,
                                           int seed=0) const {
        float r, dndx, dndy, dndz;
        r = simplexnoise3 (p.val().x, p.val().y, p.val().z,
                           seed, &dndx, &dndy, &dndz);
        result.set (r, dndx * p.dx().x + dndy * p.dx().y + dndz * p.dx().z,
                       dndx * p.dy().x + dndy * p.dy().y + dndz * p.dy().z);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p,
                                           const Dual2<float> &t, int seed=0) const {
        float r, dndx, dndy, dndz, dndt;
        r = simplexnoise4 (p.val().x, p.val().y, p.val().z, t.val(),
                           seed, &dndx, &dndy, &dndz, &dndt);
        result.set (r, dndx * p.dx().x + dndy * p.dx().y + dndz * p.dx().z + dndt * t.dx(),
                       dndx * p.dy().x + dndy * p.dy().y + dndz * p.dy().z + dndt * t.dy());
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x) const {
        Dual2<float> r0, r1, r2;
        (*this)(r0, x, 0);
        (*this)(r1, x, 1);
        (*this)(r2, x, 2);
        result = make_Vec3 (r0, r1, r2);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y) const {
        Dual2<float> r0, r1, r2;
        (*this)(r0, x, y, 0);
        (*this)(r1, x, y, 1);
        (*this)(r2, x, y, 2);
        result = make_Vec3 (r0, r1, r2);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p) const {
        Dual2<float> r0, r1, r2;
        (*this)(r0, p, 0);
        (*this)(r1, p, 1);
        (*this)(r2, p, 2);
        result = make_Vec3 (r0, r1, r2);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const {
        Dual2<float> r0, r1, r2;
        (*this)(r0, p, t, 0);
        (*this)(r1, p, t, 1);
        (*this)(r2, p, t, 2);
        result = make_Vec3 (r0, r1, r2);
    }
};


// Scalar version of SimplexNoise that is SIMD friendly suitable to be
// inlined inside of a SIMD loops
struct SimplexNoiseScalar {
    SimplexNoiseScalar () { }

    OSL_FORCEINLINE void operator() (float &result, float x) const {
        result = sfm::simplexnoise1<0/* seed */>(x);
    }

    OSL_FORCEINLINE void operator() (float &result, float x, float y) const {
        result = sfm::simplexnoise2<0/* seed */>(x, y);
    }

    OSL_FORCEINLINE void operator()(float &result, const Vec3 &p) const {
        result = sfm::simplexnoise3<0/* seed */>(p.x, p.y, p.z);
    }

    OSL_FORCEINLINE void operator()(float &result, const Vec3 &p, float t) const {
        result = sfm::simplexnoise4<0/* seed */>(p.x, p.y, p.z, t);
    }

    OSL_FORCEINLINE void operator() (Vec3 &result, float x) const {
        result.x = sfm::simplexnoise1<0/* seed */>(x);
        result.y = sfm::simplexnoise1<1/* seed */>(x);
        result.z = sfm::simplexnoise1<2/* seed */>(x);
    }

    OSL_FORCEINLINE void operator() (Vec3 &result, float x, float y) const {
        result.x = sfm::simplexnoise2<0/* seed */>(x, y);
        result.y = sfm::simplexnoise2<1/* seed */>(x, y);
        result.z = sfm::simplexnoise2<2/* seed */>(x, y);
    }

    OSL_FORCEINLINE void operator() (Vec3 &result, const Vec3 &p) const {
        result.x = sfm::simplexnoise3<0/* seed */>(p.x, p.y, p.z);
        result.y = sfm::simplexnoise3<1/* seed */>(p.x, p.y, p.z);
        result.z = sfm::simplexnoise3<2/* seed */>(p.x, p.y, p.z);
    }

    OSL_FORCEINLINE void operator() (Vec3 &result, const Vec3 &p, float t) const {
        result.x = sfm::simplexnoise4<0/* seed */>(p.x, p.y, p.z, t);
        result.y = sfm::simplexnoise4<1/* seed */>(p.x, p.y, p.z, t);
        result.z = sfm::simplexnoise4<2/* seed */>(p.x, p.y, p.z, t);
    }


    // dual versions
    template<int seed=0>
    OSL_FORCEINLINE void operator() (Dual2<float> &result, const Dual2<float> &x) const {
        float r, dndx;
        r = sfm::simplexnoise1<seed>(x.val(), sfm::DxRef(dndx));
        result.set (r, dndx * x.dx(), dndx * x.dy());
    }

    template<int seedT=0>
    OSL_FORCEINLINE void operator() (Dual2<float> &result, const Dual2<float> &x, const Dual2<float> &y) const {
        float r, dndx, dndy;
        r = sfm::simplexnoise2<seedT> (x.val(), y.val(), sfm::DxDyRef(dndx, dndy));
        result.set (r, dndx * x.dx() + dndy * y.dx(),
                       dndx * x.dy() + dndy * y.dy());
    }

    template<int seed=0>
    OSL_FORCEINLINE void operator() (Dual2<float> &result, const Dual2<Vec3> &p) const {
        float dndx, dndy, dndz;
        const Vec3 &p_val = p.val();
        float r = sfm::simplexnoise3<seed>(p_val.x, p_val.y, p_val.z, sfm::DxDyDzRef(dndx, dndy, dndz));
        result.set (r, dndx * p.dx().x + dndy * p.dx().y + dndz * p.dx().z,
                       dndx * p.dy().x + dndy * p.dy().y + dndz * p.dy().z);
    }

    template<int seed=0>
    OSL_FORCEINLINE void operator()(Dual2<float> &result, const Dual2<Vec3> &p,
                            const Dual2<float> &t) const {
        float dndx, dndy, dndz, dndt;
        float r = sfm::simplexnoise4<seed> (p.val().x, p.val().y, p.val().z, t.val(),
                           sfm::DxDyDzDwRef(dndx, dndy, dndz, dndt));
        result.set (r, dndx * p.dx().x + dndy * p.dx().y + dndz * p.dx().z + dndt * t.dx(),
                       dndx * p.dy().x + dndy * p.dy().y + dndz * p.dy().z + dndt * t.dy());
    }

    OSL_FORCEINLINE void operator() (Dual2<Vec3> &result, const Dual2<float> &x) const {
        Dual2<float> r0, r1, r2;
        operator()<0>(r0, x);
        operator()<1>(r1, x);
        operator()<2>(r2, x);
        result = make_Vec3 (r0, r1, r2);
    }

    OSL_FORCEINLINE void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y) const {
        Dual2<float> r0, r1, r2;
        operator()<0>(r0, x, y);
        operator()<1>(r1, x, y);
        operator()<2>(r2, x, y);
        result = make_Vec3 (r0, r1, r2);
    }

    OSL_FORCEINLINE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p) const {
        Dual2<float> r0, r1, r2;
        operator()<0>(r0, p);
        operator()<1>(r1, p);
        operator()<2>(r2, p);
        result = make_Vec3 (r0, r1, r2);
    }

    OSL_FORCEINLINE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const {
        Dual2<float> r0, r1, r2;
        operator()<0>(r0, p, t);
        operator()<1>(r1, p, t);
        operator()<2>(r2, p, t);
        result = make_Vec3 (r0, r1, r2);
    }
};

// Unsigned simplex noise
struct USimplexNoise {
    OSL_HOSTDEVICE USimplexNoise () { }

    inline OSL_HOSTDEVICE void operator() (float &result, float x) const {
        result = 0.5f * (simplexnoise1 (x) + 1.0f);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, float x, float y) const {
        result = 0.5f * (simplexnoise2 (x, y) + 1.0f);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p) const {
        result = 0.5f * (simplexnoise3 (p.x, p.y, p.z) + 1.0f);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, float t) const {
        result = 0.5f * (simplexnoise4 (p.x, p.y, p.z, t) + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, float x) const {
        result.x = 0.5f * (simplexnoise1 (x, 0) + 1.0f);
        result.y = 0.5f * (simplexnoise1 (x, 1) + 1.0f);
        result.z = 0.5f * (simplexnoise1 (x, 2) + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float y) const {
        result.x = 0.5f * (simplexnoise2 (x, y, 0) + 1.0f);
        result.y = 0.5f * (simplexnoise2 (x, y, 1) + 1.0f);
        result.z = 0.5f * (simplexnoise2 (x, y, 2) + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p) const {
        result.x = 0.5f * (simplexnoise3 (p.x, p.y, p.z, 0) + 1.0f);
        result.y = 0.5f * (simplexnoise3 (p.x, p.y, p.z, 1) + 1.0f);
        result.z = 0.5f * (simplexnoise3 (p.x, p.y, p.z, 2) + 1.0f);
    }

    OSL_FORCEINLINE OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, float t) const {
        result.x = 0.5f * (simplexnoise4 (p.x, p.y, p.z, t, 0) + 1.0f);
        result.y = 0.5f * (simplexnoise4 (p.x, p.y, p.z, t, 1) + 1.0f);
        result.z = 0.5f * (simplexnoise4 (p.x, p.y, p.z, t, 2) + 1.0f);
    }


    // dual versions

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x,
                                           int seed=0) const {
        float r, dndx;
        r = simplexnoise1 (x.val(), seed, &dndx);
        r = 0.5f * (r + 1.0f);
        dndx *= 0.5f;
        result.set (r, dndx * x.dx(), dndx * x.dy());
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x,
                                           const Dual2<float> &y, int seed=0) const {
        float r, dndx, dndy;
        r = simplexnoise2 (x.val(), y.val(), seed, &dndx, &dndy);
        r = 0.5f * (r + 1.0f);
        dndx *= 0.5f;
        dndy *= 0.5f;
        result.set (r, dndx * x.dx() + dndy * y.dx(),
                       dndx * x.dy() + dndy * y.dy());
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p,
                                           int seed=0) const {
        float r, dndx, dndy, dndz;
        r = simplexnoise3 (p.val().x, p.val().y, p.val().z,
                           seed, &dndx, &dndy, &dndz);
        r = 0.5f * (r + 1.0f);
        dndx *= 0.5f;
        dndy *= 0.5f;
        dndz *= 0.5f;
        result.set (r, dndx * p.dx().x + dndy * p.dx().y + dndz * p.dx().z,
                       dndx * p.dy().x + dndy * p.dy().y + dndz * p.dy().z);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p,
                                           const Dual2<float> &t, int seed=0) const {
        float r, dndx, dndy, dndz, dndt;
        r = simplexnoise4 (p.val().x, p.val().y, p.val().z, t.val(),
                           seed, &dndx, &dndy, &dndz, &dndt);
        r = 0.5f * (r + 1.0f);
        dndx *= 0.5f;
        dndy *= 0.5f;
        dndz *= 0.5f;
        dndt *= 0.5f;
        result.set (r, dndx * p.dx().x + dndy * p.dx().y + dndz * p.dx().z + dndt * t.dx(),
                       dndx * p.dy().x + dndy * p.dy().y + dndz * p.dy().z + dndt * t.dy());
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x) const {
        Dual2<float> r0, r1, r2;
        (*this)(r0, x, 0);
        (*this)(r1, x, 1);
        (*this)(r2, x, 2);
        result = make_Vec3 (r0, r1, r2);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y) const {
        Dual2<float> r0, r1, r2;
        (*this)(r0, x, y, 0);
        (*this)(r1, x, y, 1);
        (*this)(r2, x, y, 2);
        result = make_Vec3 (r0, r1, r2);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p) const {
        Dual2<float> r0, r1, r2;
        (*this)(r0, p, 0);
        (*this)(r1, p, 1);
        (*this)(r2, p, 2);
        result = make_Vec3 (r0, r1, r2);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const {
        Dual2<float> r0, r1, r2;
        (*this)(r0, p, t, 0);
        (*this)(r1, p, t, 1);
        (*this)(r2, p, t, 2);
        result = make_Vec3 (r0, r1, r2);
    }

};

// Scalar version of USimplexNoise that is SIMD friendly suitable to be
// inlined inside of a SIMD loops
struct USimplexNoiseScalar {
	OSL_FORCEINLINE USimplexNoiseScalar () { }

    OSL_FORCEINLINE void operator() (float &result, float x) const {
        result = 0.5f * (sfm::simplexnoise1<0/* seed */>(x) + 1.0f);
    }

    OSL_FORCEINLINE void operator() (float &result, float x, float y) const {
        result = 0.5f * (sfm::simplexnoise2<0/* seed */>(x, y) + 1.0f);
    }

    OSL_FORCEINLINE void operator() (float &result, const Vec3 &p) const {
        result = 0.5f * (sfm::simplexnoise3<0/* seed */>(p.x, p.y, p.z) + 1.0f);
    }

    OSL_FORCEINLINE void operator() (float &result, const Vec3 &p, float t) const {
        result = 0.5f * (sfm::simplexnoise4<0/* seed */> (p.x, p.y, p.z, t) + 1.0f);
    }

    OSL_FORCEINLINE void operator() (Vec3 &result, float x) const {
        result.x = 0.5f * (sfm::simplexnoise1<0/* seed */>(x) + 1.0f);
        result.y = 0.5f * (sfm::simplexnoise1<1/* seed */>(x) + 1.0f);
        result.z = 0.5f * (sfm::simplexnoise1<2/* seed */>(x) + 1.0f);
    }

    OSL_FORCEINLINE void operator() (Vec3 &result, float x, float y) const {
        result.x = 0.5f * (sfm::simplexnoise2<0>(x, y) + 1.0f);
        result.y = 0.5f * (sfm::simplexnoise2<1>(x, y) + 1.0f);
        result.z = 0.5f * (sfm::simplexnoise2<2>(x, y) + 1.0f);
    }

    OSL_FORCEINLINE void operator() (Vec3 &result, const Vec3 &p) const {
        result.x = 0.5f * (sfm::simplexnoise3<0/* seed */>(p.x, p.y, p.z) + 1.0f);
        result.y = 0.5f * (sfm::simplexnoise3<1/* seed */>(p.x, p.y, p.z) + 1.0f);
        result.z = 0.5f * (sfm::simplexnoise3<2/* seed */>(p.x, p.y, p.z) + 1.0f);
    }

    OSL_FORCEINLINE void operator() (Vec3 &result, const Vec3 &p, float t) const {
        result.x = 0.5f * (sfm::simplexnoise4<0/* seed */>(p.x, p.y, p.z, t) + 1.0f);
        result.y = 0.5f * (sfm::simplexnoise4<1/* seed */>(p.x, p.y, p.z, t) + 1.0f);
        result.z = 0.5f * (sfm::simplexnoise4<2/* seed */>(p.x, p.y, p.z, t) + 1.0f);
    }

    // dual versions
    template<int seed=0>
    OSL_FORCEINLINE void operator() (Dual2<float> &result, const Dual2<float> &x) const {
        float r, dndx;
        r = sfm::simplexnoise1<seed>(x.val(), sfm::DxRef(dndx));
        r = 0.5f * (r + 1.0f);
        dndx *= 0.5f;
        result.set (r, dndx * x.dx(), dndx * x.dy());
    }

    template<int seedT=0>
    OSL_FORCEINLINE void operator() (Dual2<float> &result, const Dual2<float> &x,
                             const Dual2<float> &y) const {
        float r, dndx, dndy;
        r = sfm::simplexnoise2<seedT> (x.val(), y.val(), sfm::DxDyRef(dndx, dndy));
        r = 0.5f * (r + 1.0f);
        dndx *= 0.5f;
        dndy *= 0.5f;
        result.set (r, dndx * x.dx() + dndy * y.dx(),
                       dndx * x.dy() + dndy * y.dy());
    }

    template<int seed=0>
    OSL_FORCEINLINE void operator() (Dual2<float> &result, const Dual2<Vec3> &p) const {
        float r, dndx, dndy, dndz;
        const Vec3 &p_val = p.val();
        r = sfm::simplexnoise3<seed>(p_val.x, p_val.y, p_val.z, sfm::DxDyDzRef(dndx, dndy, dndz));
        r = 0.5f * (r + 1.0f);
        dndx *= 0.5f;
        dndy *= 0.5f;
        dndz *= 0.5f;
        result.set (r, dndx * p.dx().x + dndy * p.dx().y + dndz * p.dx().z,
                       dndx * p.dy().x + dndy * p.dy().y + dndz * p.dy().z);
    }

    template<int seed=0>
    OSL_FORCEINLINE void operator()(Dual2<float> &result, const Dual2<Vec3> &p,
                            const Dual2<float> &t) const {
        float dndx, dndy, dndz, dndt;
        float r = sfm::simplexnoise4<seed> (p.val().x, p.val().y, p.val().z, t.val(),
                           sfm::DxDyDzDwRef(dndx, dndy, dndz, dndt));
        r = 0.5f * (r + 1.0f);
        dndx *= 0.5f;
        dndy *= 0.5f;
        dndz *= 0.5f;
        dndt *= 0.5f;
        result.set (r, dndx * p.dx().x + dndy * p.dx().y + dndz * p.dx().z + dndt * t.dx(),
                       dndx * p.dy().x + dndy * p.dy().y + dndz * p.dy().z + dndt * t.dy());
    }

    OSL_FORCEINLINE void operator() (Dual2<Vec3> &result, const Dual2<float> &x) const {
        Dual2<float> r0, r1, r2;
        operator()<0>(r0, x);
        operator()<1>(r1, x);
        operator()<2>(r2, x);
        result = make_Vec3 (r0, r1, r2);
    }

    OSL_FORCEINLINE void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y) const {
        Dual2<float> r0, r1, r2;
        operator()<0>(r0, x, y);
        operator()<1>(r1, x, y);
        operator()<2>(r2, x, y);
        result = make_Vec3 (r0, r1, r2);
    }

    OSL_FORCEINLINE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p) const {
        Dual2<float> r0, r1, r2;
        operator()<0>(r0, p);
        operator()<1>(r1, p);
        operator()<2>(r2, p);
        result = make_Vec3 (r0, r1, r2);
    }

    OSL_FORCEINLINE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const {
        Dual2<float> r0, r1, r2;
        operator()<0>(r0, p, t);
        operator()<1>(r1, p, t);
        operator()<2>(r2, p, t);
        result = make_Vec3 (r0, r1, r2);
    }

};
} // anonymous namespace



OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<float> gabor (const Dual2<Vec3> &P, const NoiseParams *opt);



OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<float> gabor (const Dual2<float> &x, const Dual2<float> &y,
                    const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<float> gabor (const Dual2<float> &x, const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<Vec3> gabor3 (const Dual2<Vec3> &P, const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<Vec3> gabor3 (const Dual2<float> &x, const Dual2<float> &y,
                    const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<Vec3> gabor3 (const Dual2<float> &x, const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<float> pgabor (const Dual2<Vec3> &P, const Vec3 &Pperiod,
                     const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<float> pgabor (const Dual2<float> &x, const Dual2<float> &y,
                     float xperiod, float yperiod, const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<float> pgabor (const Dual2<float> &x, float xperiod,
                     const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<Vec3> pgabor3 (const Dual2<Vec3> &P, const Vec3 &Pperiod,
                     const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<Vec3> pgabor3 (const Dual2<float> &x, const Dual2<float> &y,
                     float xperiod, float yperiod, const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<Vec3> pgabor3 (const Dual2<float> &x, float xperiod,
                     const NoiseParams *opt);



}; // namespace pvt

namespace oslnoise {

#define DECLNOISE(name,impl)                    \
    template <class S> OSL_HOSTDEVICE           \
    inline float name (S x) {                   \
        pvt::impl noise;                        \
        float r;                                \
        noise (r, x);                           \
        return r;                               \
    }                                           \
                                                \
    template <class S, class T> OSL_HOSTDEVICE  \
    inline float name (S x, T y) {              \
        pvt::impl noise;                        \
        float r;                                \
        noise (r, x, y);                        \
        return r;                               \
    }                                           \
                                                \
    template <class S> OSL_HOSTDEVICE           \
    inline Vec3 v ## name (S x) {               \
        pvt::impl noise;                        \
        Vec3 r;                                 \
        noise (r, x);                           \
        return r;                               \
    }                                           \
                                                \
    template <class S, class T> OSL_HOSTDEVICE  \
    inline Vec3 v ## name (S x, T y) {          \
        pvt::impl noise;                        \
        Vec3 r;                                 \
        noise (r, x, y);                        \
        return r;                               \
    }


DECLNOISE (snoise, SNoise)
DECLNOISE (noise, Noise)
DECLNOISE (cellnoise, CellNoise)
DECLNOISE (hashnoise, HashNoise)

#undef DECLNOISE
}   // namespace oslnoise


OSL_NAMESPACE_EXIT
