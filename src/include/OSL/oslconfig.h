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

// Symbol export defines
#include "export.h"

// All the things we need from Imath
#ifdef __CUDACC__
// When compiling for CUDA, we need to make sure the modified Imath
// headers are included before the stock versions.
#  include <OSL/ImathVec_cuda.h>
#  include <OSL/ImathMatrix_cuda.h>
#else
#  include <OpenEXR/ImathVec.h>
#  include <OpenEXR/ImathMatrix.h>
#endif

#include <OpenEXR/ImathColor.h>

// All the things we need from OpenImageIO
#include <OpenImageIO/oiioversion.h>
#include <OpenImageIO/errorhandler.h>
#include <OpenImageIO/texture.h>
#include <OpenImageIO/typedesc.h>
#include <OpenImageIO/ustring.h>
#include <OpenImageIO/platform.h>

// Make sure we can use OIIO::cspan
#if OIIO_VERSION >= 10904
#  include <OpenImageIO/span.h>
#else
#  include <OpenImageIO/array_view.h>
#endif

// If we're using an old version of OIIO prior to the introduction of
// Strutil::sprintf, define it ourselves to be a synonym for format.
#ifndef OIIO_HAS_SPRINTF
OIIO_NAMESPACE_BEGIN
namespace Strutil {
template<typename... Args>
inline std::string sprintf (const char* fmt, const Args&... args) {
    return Strutil::format (fmt, args...);
}
} // namespace strutil
OIIO_NAMESPACE_END
#endif

// Extensions to Imath
#include <OSL/matrix22.h>

#include <OSL/oslversion.h>

OSL_NAMESPACE_ENTER


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

// Make sure we can use OIIO::cspan
#if OIIO_VERSION >= 10904
  using OIIO::cspan;
#else
  template<typename T> using cspan = OIIO::array_view<const T>;
#endif


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

OSL_NAMESPACE_EXIT
