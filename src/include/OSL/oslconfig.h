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

#pragma once

/////////////////////////////////////////////////////////////////////////
/// \file
/// Various compile-time defaults are defined here that could, in
/// principle, be redefined if you are using OSL in some particular
/// renderer that wanted things a different way.
/////////////////////////////////////////////////////////////////////////

// Detect if we're C++11.
//
// Note: oslversion.h defined OSL_BUILD_CPP11 to be 1 if OSL was built
// using C++11. In contrast, OSL_CPLUSPLUS_VERSION defined below will be set
// to the right number for the C++ standard being compiled RIGHT NOW. These
// two things may be the same when compiling OSL, but they may not be the
// same if another packages is compiling against OSL and using these headers
// (OSL may be C++11 but the client package may be older, or vice versa --
// use these two symbols to differentiate these cases, when important).
#if (__cplusplus >= 201703L)
#    define OSL_CPLUSPLUS_VERSION 17
#    define OSL_CONSTEXPR14 constexpr
#    define OSL_CONSTEXPR17 constexpr
#    define OSL_CONSTEXPR20 /* not constexpr before C++20 */
#elif (__cplusplus >= 201402L)
#    define OSL_CPLUSPLUS_VERSION 14
#    define OSL_CONSTEXPR14 constexpr
#    define OSL_CONSTEXPR17 /* not constexpr before C++17 */
#    define OSL_CONSTEXPR20 /* not constexpr before C++20 */
#elif (__cplusplus >= 201103L) || _MSC_VER >= 1900
#    define OSL_CPLUSPLUS_VERSION 11
#    define OSL_CONSTEXPR14 /* not constexpr before C++14 */
#    define OSL_CONSTEXPR17 /* not constexpr before C++17 */
#    define OSL_CONSTEXPR20 /* not constexpr before C++20 */
#else
#    error "This version of OSL requires C++11"
#endif

#ifndef OSL_DEBUG
    #ifdef NDEBUG
        #define OSL_DEBUG 0
    #else
        #ifdef _DEBUG
            #define OSL_DEBUG _DEBUG
        #else
            #define OSL_DEBUG 0
        #endif
    #endif
#endif // OSL_DEBUG

#ifndef OSL_INLINE
    #if OSL_DEBUG
        #define OSL_INLINE inline
    #else
        #if __INTEL_COMPILER >= 1100 || (defined(_WIN32) || defined(_WIN64))
            #define OSL_INLINE __forceinline
        #else
            #define OSL_INLINE inline __attribute__((always_inline))
        #endif
    #endif
#endif

#ifndef OSL_NOINLINE
    #if (defined(_WIN32) || defined(_WIN64))
        #define OSL_NOINLINE __declspec(noinline)
    #else
        #define OSL_NOINLINE __attribute__((noinline))
    #endif
#endif

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    #define OSL_EXPECT_FALSE(EXPRESSION) EXPRESSION
    #define OSL_EXPECT_TRUE(EXPRESSION) EXPRESSION
#else
    #define OSL_EXPECT_FALSE(EXPRESSION) __builtin_expect(EXPRESSION, false)
    #define OSL_EXPECT_TRUE(EXPRESSION) __builtin_expect(EXPRESSION, true)
#endif

#ifndef OSL_HOSTDEVICE
#  ifdef __CUDACC__
#    define OSL_HOSTDEVICE __host__ __device__
#  else
#    define OSL_HOSTDEVICE
#  endif
#endif

#ifndef OSL_DEVICE
#  ifdef __CUDA_ARCH__
#    define OSL_DEVICE __device__
#  else
#    define OSL_DEVICE
#  endif
#endif

#ifndef OSL_CONSTANT_DATA
#  ifdef __CUDA_ARCH__
#    define OSL_CONSTANT_DATA __constant__
#  else
#    define OSL_CONSTANT_DATA
#  endif
#endif

// Symbol export defines
#include "export.h"

#include <OSL/oslversion.h>

// All the things we need from Imath
// When compiling for CUDA, we need to make sure the modified Imath
// headers are included before the stock versions.
#include <OSL/Imathx.h>

// Temporary bug fix: having trouble with cuda complaining about {fmt} lib.
// Work around by disabling it from oiio headers.
#ifdef __CUDA_ARCH__
#    define OIIO_USE_FMT 0
#endif

// All the things we need from OpenImageIO
#include <OpenImageIO/oiioversion.h>
#include <OpenImageIO/errorhandler.h>
#include <OpenImageIO/texture.h>
#include <OpenImageIO/typedesc.h>
#include <OpenImageIO/ustring.h>
#include <OpenImageIO/platform.h>
#include <OpenImageIO/span.h>

#if OSL_CPLUSPLUS_VERSION >= 17
    // fold expression to expand
    #define __OSL_EXPAND_PARAMETER_PACKS(EXPRESSION) \
        (void((EXPRESSION)) , ... );
#else
    // Choose to use initializer list to expand our parameter pack so that
    // the order they are evaluated in the order they appear
    #define __OSL_EXPAND_PARAMETER_PACKS(EXPRESSION) \
        { \
            using expander = int [sizeof...(IntListT)]; \
            (void)expander{((EXPRESSION),0)...}; \
        }
#endif

OSL_NAMESPACE_ENTER

namespace pvt {

#if (__cplusplus >= 201402L)
    template<int... IntegerListT>
    using int_sequence = std::integer_sequence<int, IntegerListT...>;

    template<int EndBeforeT>
    using make_int_sequence =  std::make_integer_sequence<int, EndBeforeT>;
#else
    // std::integer_sequence requires c++14 library support,
    // we have our own version here in case the environment is c++14 compiler
    // building against c++11 library.
    template<int... IntegerListT>
    struct int_sequence
    {
    };

    template<int StartAtT, int EndBeforeT, typename IntSequenceT>
    struct int_sequence_generator;

    template<int StartAtT, int EndBeforeT, int... IntegerListT>
    struct int_sequence_generator<StartAtT, EndBeforeT, int_sequence<IntegerListT...>>
    {
        typedef typename int_sequence_generator<StartAtT+1, EndBeforeT, int_sequence<IntegerListT..., StartAtT>>::type type;
    };

    template<int EndBeforeT, int... IntegerListT>
    struct int_sequence_generator<EndBeforeT, EndBeforeT, int_sequence<IntegerListT...>>
    {
        typedef int_sequence<IntegerListT...> type;
    };

    template<int EndBeforeT>
    using make_int_sequence = typename int_sequence_generator<0, EndBeforeT, int_sequence<> >::type;
#endif
} // namespace pvt

// Instead of relying on compiler loop unrolling, we can statically call functor
// for each integer in a sequence
template <template<int> class ConstantWrapperT, int... IntListT, typename FunctorT>
static OSL_INLINE void static_foreach(pvt::int_sequence<IntListT...>, const FunctorT &iFunctor) {
     __OSL_EXPAND_PARAMETER_PACKS( iFunctor(ConstantWrapperT<IntListT>{}) );
}

template <template<int> class ConstantWrapperT, int N, typename FunctorT>
static OSL_INLINE void static_foreach(const FunctorT &iFunctor) {
    static_foreach<ConstantWrapperT>(pvt::make_int_sequence<N>(), iFunctor);
}

#if (__cplusplus >= 201402L)
    template<int N>
    using ConstIndex = std::integral_constant<int, N>;
#else
    template<int N>
    struct ConstIndex : public std::integral_constant<int, N>
    {
        // C++14 adds this conversion operator we need to allow non-generic
        // lambda functions (pre C++14) to accept an "int" instead of a
        // typed ConstIndex<N> with "auto"
        constexpr int operator()() const noexcept {
            return N;
        }
    };
#endif

/// By default, we operate with single precision float.  Change this
/// definition to make a shading system that fundamentally operates
/// on doubles.
/// FIXME: it's very likely that all sorts of other things will break
/// if you do this, but eventually we should make sure it works.
typedef float Float;

/// By default, use the excellent Imath vector, matrix, and color types
/// from the IlmBase package from: http://www.openexr.com
///
/// It's permissible to override these types with the vector, matrix,
/// and color classes of your choice, provided that (a) your vectors
/// have the same data layout as a simple Float[n]; (b) your
/// matrices have the same data layout as Float[n][n]; and (c) your
/// classes have most of the obvious constructors and overloaded
/// operators one would expect from a C++ vector/matrix/color class.
typedef Imath::Vec3<Float>     Vec3;
typedef Imath::Matrix33<Float> Matrix33;
typedef Imath::Matrix44<Float> Matrix44;
typedef Imath::Color3<Float>   Color3;
typedef Imath::Vec2<Float>     Vec2;

typedef Imathx::Matrix22<Float> Matrix22;

/// Assume that we are dealing with OpenImageIO's texture system.  It
/// doesn't literally have to be OIIO's... it just needs to have the
/// same API as OIIO's TextureSystem class, it's a purely abstract class
/// anyway.
using OIIO::TextureSystem;
using OIIO::TextureOpt;

// And some other things we borrow from OIIO...
using OIIO::ErrorHandler;
using OIIO::TypeDesc;
using OIIO::ustring;
using OIIO::ustringHash;
using OIIO::string_view;
using OIIO::span;
using OIIO::cspan;


// In C++20 (and some compilers before that), __has_cpp_attribute can
// test for understand of [[attr]] tests.
#ifndef __has_cpp_attribute
#    define __has_cpp_attribute(x) 0
#endif

// On gcc & clang, __has_attribute can test for __attribute__((attr))
#ifndef __has_attribute
#    define __has_attribute(x) 0
#endif

// In C++17 (and some compilers before that), __has_include("blah.h") or
// __has_include(<blah.h>) can test for presence of an include file.
#ifndef __has_include
#    define __has_include(x) 0
#endif


#if OSL_CPLUSPLUS_VERSION >= 14 || __has_cpp_attribute(deprecated)
#  define OSL_DEPRECATED(msg) [[deprecated(msg)]]
#elif defined(__GNUC__) || defined(__clang__) || __has_attribute(deprecated)
#  define OSL_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
#  define OSL_DEPRECATED(msg) __declspec(deprecated(msg))
#else
#  define OSL_DEPRECATED(msg)
#endif

/// Work around bug in GCC with mixed __attribute__ and alignas parsing
/// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=69585
#ifdef __GNUC__
#  define OSL_ALIGNAS(size) __attribute__((aligned(size)))
#else
#  define OSL_ALIGNAS(size) alignas(size)
#endif


// OSL_PRETTY_FUNCTION gives a text string of the current function
// declaration.
#if defined(__PRETTY_FUNCTION__)
#    define OSL_PRETTY_FUNCTION __PRETTY_FUNCTION__ /* gcc, clang */
#elif defined(__FUNCSIG__)
#    define OSL_PRETTY_FUNCTION __FUNCSIG__ /* MS gotta be different */
#else
#    define OSL_PRETTY_FUNCTION __FUNCTION__
#endif



/// OSL_ABORT_IF_DEBUG is a call to abort() for debug builds, but does
/// nothing for release builds.
#ifndef NDEBUG
#    define OSL_ABORT_IF_DEBUG abort()
#else
#    define OSL_ABORT_IF_DEBUG (void)0
#endif

/// OSL_ASSERT(condition) checks if the condition is met, and if not,
/// prints an error message indicating the module and line where the error
/// occurred, and additionally aborts if in debug mode. When in release
/// mode, it prints the error message if the condition fails, but does not
/// abort.
///
/// OSL_ASSERT_MSG(condition,msg,...) lets you add formatted output (a la
/// printf) to the failure message.
#ifndef __CUDA_ARCH__
#    define OSL_ASSERT_PRINT(...) (std::fprintf(stderr, __VA_ARGS__))
#else
#    define OSL_ASSERT_PRINT(...) (printf(__VA_ARGS__))
#endif

#define OSL_ASSERT(x)                                                          \
    (OIIO_LIKELY(x)                                                            \
         ? ((void)0)                                                           \
         : (OSL_ASSERT_PRINT("%s:%u: %s: Assertion '%s' failed.\n",            \
                             __FILE__, __LINE__, OSL_PRETTY_FUNCTION, #x),     \
            OSL_ABORT_IF_DEBUG))
#define OSL_ASSERT_MSG(x, msg, ...)                                            \
    (OIIO_LIKELY(x)                                                            \
         ? ((void)0)                                                           \
         : (OSL_ASSERT_PRINT("%s:%u: %s: Assertion '%s' failed: " msg "\n",    \
                             __FILE__, __LINE__, OSL_PRETTY_FUNCTION, #x,      \
                             __VA_ARGS__),                                     \
            OSL_ABORT_IF_DEBUG))

/// OSL_DASSERT and OSL_DASSERT_MSG are the same as OSL_ASSERT for debug
/// builds (test, print error, abort), but do nothing at all in release
/// builds (not even perform the test). This is similar to C/C++ assert(),
/// but gives us flexibility in improving our error messages. It is also ok
/// to use regular assert() for this purpose if you need to eliminate the
/// dependency on this header from a particular place (and don't mind that
/// assert won't format identically on all platforms).
#ifndef NDEBUG
#    define OSL_DASSERT OSL_ASSERT
#    define OSL_DASSERT_MSG OSL_ASSERT_MSG
#else
#    define OSL_DASSERT(x) ((void)sizeof(x))          /*NOLINT*/
#    define OSL_DASSERT_MSG(x, ...) ((void)sizeof(x)) /*NOLINT*/
#endif


OSL_NAMESPACE_EXIT
