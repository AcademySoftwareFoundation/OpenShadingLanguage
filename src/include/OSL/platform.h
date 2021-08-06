// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

/////////////////////////////////////////////////////////////////////////
// \file
// platform.h is where we put all the platform-specific macros.
// Things like:
//
//   * Detecting which compiler is being used.
//   * Detecting which C++ standard is being used and what features are
//     available.
//   * Various helpers that need to be defined differently per compiler,
//     language version, OS, etc.
/////////////////////////////////////////////////////////////////////////

// clang-format off

#pragma once

#include <memory>

#include <OSL/oslversion.h>


/////////////////////////////////////////////////////////////////////////
// Detect which compiler and version we're using


// Define OSL_GNUC_VERSION to hold an encoded gcc version (e.g. 40802 for
// 4.8.2), or 0 if not a GCC release. N.B.: This will be 0 for clang.
#if defined(__GNUC__) && !defined(__clang__)
#  define OSL_GNUC_VERSION (10000*__GNUC__ + 100*__GNUC_MINOR__ + __GNUC_PATCHLEVEL__)
#else
#  define OSL_GNUC_VERSION 0
#endif

// Define OSL_CLANG_VERSION to hold an encoded generic Clang version (e.g.
// 30402 for clang 3.4.2), or 0 if not a generic Clang release.
// N.B. This will be 0 for the clang Apple distributes (which has different
// version numbers entirely).
#if defined(__clang__) && !defined(__apple_build_version__)
#  define OSL_CLANG_VERSION (10000*__clang_major__ + 100*__clang_minor__ + __clang_patchlevel__)
#else
#  define OSL_CLANG_VERSION 0
#endif

// Define OSL_APPLE_CLANG_VERSION to hold an encoded Apple Clang version
// (e.g. 70002 for clang 7.0.2), or 0 if not an Apple Clang release.
#if defined(__clang__) && defined(__apple_build_version__)
#  define OSL_APPLE_CLANG_VERSION (10000*__clang_major__ + 100*__clang_minor__ + __clang_patchlevel__)
#else
#  define OSL_APPLE_CLANG_VERSION 0
#endif

// Define OSL_INTEL_COMPILER to hold an encoded Intel compiler version
// (e.g. 1900), or 0 if not an Intel compiler.
#if defined(__INTEL_COMPILER)
#  define OSL_INTEL_COMPILER __INTEL_COMPILER
#else
#  define OSL_INTEL_COMPILER 0
#endif

// Intel's compiler on OSX may still define __clang__
// and we have need to know when using a true clang compiler
#if !defined(__INTEL_COMPILER) && defined(__clang__)
    #define OSL_NON_INTEL_CLANG  __clang__
#else
	#define OSL_NON_INTEL_CLANG  0
#endif

// Tests for MSVS versions, always 0 if not MSVS at all.
#if defined(_MSC_VER)
#  define OSL_MSVS_AT_LEAST_2013 (_MSC_VER >= 1800)
#  define OSL_MSVS_BEFORE_2013   (_MSC_VER <  1800)
#  define OSL_MSVS_AT_LEAST_2015 (_MSC_VER >= 1900)
#  define OSL_MSVS_BEFORE_2015   (_MSC_VER <  1900)
#  define OSL_MSVS_AT_LEAST_2017 (_MSC_VER >= 1910)
#  define OSL_MSVS_BEFORE_2017   (_MSC_VER <  1910)
#  if OSL_MSVS_BEFORE_2017
#    error "This version of OSL is meant to work only with Visual Studio 2017 or later"
#  endif
#else
#  define OSL_MSVS_AT_LEAST_2013 0
#  define OSL_MSVS_BEFORE_2013   0
#  define OSL_MSVS_AT_LEAST_2015 0
#  define OSL_MSVS_BEFORE_2015   0
#  define OSL_MSVS_AT_LEAST_2017 0
#  define OSL_MSVS_BEFORE_2017   0
#endif


// Note about __CUDACC__ versus __CUDA_ARCH__:
//
// __CUDACC__     defined any time we are compiling a module for Cuda
//                (both for the host pass and the device pass). "Do this
//                when using nvcc or clang with ptx target."
// __CUDA_ARCH__  is only defined when doing the device pass. "Do this only
//                for code that will actually run on the GPU."


/////////////////////////////////////////////////////////////////////////
// Detect which C++ standard we're using, and handy macros.
// See https://en.cppreference.com/w/cpp/compiler_support
//
// Note: oslversion.h defines OSL_BUILD_CPP to be 14, 17, etc., to reflect
// the version that OSL itself was built with. In contrast,
// OSL_CPLUSPLUS_VERSION defined below will be set to the right number for
// the C++ standard being compiled RIGHT NOW by whomever is parsing these
// header files. These two things may be the same when compiling OSL, but
// they may not be the same if another package is compiling against OSL and
// using these headers (e.g., OSL may be C++14 but the client package may be
// newer, or vice versa -- use these two symbols to differentiate these
// cases, when important).
#if (__cplusplus >= 202001L)
#    define OSL_CPLUSPLUS_VERSION 20
#    define OSL_CONSTEXPR17 constexpr
#    define OSL_CONSTEXPR20 constexpr
#elif (__cplusplus >= 201703L)
#    define OSL_CPLUSPLUS_VERSION 17
#    define OSL_CONSTEXPR17 constexpr
#    define OSL_CONSTEXPR20 /* not constexpr before C++20 */
#elif (__cplusplus >= 201402L) || (defined(_MSC_VER) && _MSC_VER >= 1914)
#    define OSL_CPLUSPLUS_VERSION 14
#    define OSL_CONSTEXPR17 /* not constexpr before C++17 */
#    define OSL_CONSTEXPR20 /* not constexpr before C++20 */
#else
#    error "This version of OSL requires C++14 or above"
#endif

// DEPRECATED(1.12): use C++14 constexpr
#define OSL_CONSTEXPR14 constexpr



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



/////////////////////////////////////////////////////////////////////////
// Pragmas and attributes that vary per platform

// Generic pragma definition
#if defined(_MSC_VER)
    // Of course MS does it in a quirky way
    #define OSL_PRAGMA(UnQuotedPragma) __pragma(UnQuotedPragma)
#else
    // All other compilers seem to support C99 _Pragma
    #define OSL_PRAGMA(UnQuotedPragma) _Pragma(#UnQuotedPragma)
#endif

// Compiler-specific pragmas
#if defined(__GNUC__) /* gcc, clang, icc */
#    define OSL_PRAGMA_WARNING_PUSH    OSL_PRAGMA(GCC diagnostic push)
#    define OSL_PRAGMA_WARNING_POP     OSL_PRAGMA(GCC diagnostic pop)
#    define OSL_PRAGMA_VISIBILITY_PUSH OSL_PRAGMA(GCC visibility push(default))
#    define OSL_PRAGMA_VISIBILITY_POP  OSL_PRAGMA(GCC visibility pop)
#    define OSL_GCC_PRAGMA(UnQuotedPragma) OSL_PRAGMA(UnQuotedPragma)
#    if defined(__clang__)
#        define OSL_CLANG_PRAGMA(UnQuotedPragma) OSL_PRAGMA(UnQuotedPragma)
#    else
#        define OSL_CLANG_PRAGMA(UnQuotedPragma)
#    endif
#    if defined(__INTEL_COMPILER)
#        define OSL_INTEL_PRAGMA(UnQuotedPragma) OSL_PRAGMA(UnQuotedPragma)
#    else
#        define OSL_INTEL_PRAGMA(UnQuotedPragma)
#    endif
#    define OSL_MSVS_PRAGMA(UnQuotedPragma)
#elif defined(_MSC_VER)
#    define OSL_PRAGMA_WARNING_PUSH __pragma(warning(push))
#    define OSL_PRAGMA_WARNING_POP  __pragma(warning(pop))
#    define OSL_PRAGMA_VISIBILITY_PUSH /* N/A on MSVS */
#    define OSL_PRAGMA_VISIBILITY_POP  /* N/A on MSVS */
#    define OSL_GCC_PRAGMA(UnQuotedPragma)
#    define OSL_CLANG_PRAGMA(UnQuotedPragma)
#    define OSL_INTEL_PRAGMA(UnQuotedPragma)
#    define OSL_MSVS_PRAGMA(UnQuotedPragma) OSL_PRAGMA(UnQuotedPragma)
#else
#    define OSL_PRAGMA_WARNING_PUSH
#    define OSL_PRAGMA_WARNING_POP
#    define OSL_PRAGMA_VISIBILITY_PUSH
#    define OSL_PRAGMA_VISIBILITY_POP
#    define OSL_GCC_PRAGMA(UnQuotedPragma)
#    define OSL_CLANG_PRAGMA(UnQuotedPragma)
#    define OSL_INTEL_PRAGMA(UnQuotedPragma)
#    define OSL_MSVS_PRAGMA(UnQuotedPragma)
#endif

#ifdef __clang__
    #define OSL_CLANG_ATTRIBUTE(value) __attribute__((value))
#else
    #define OSL_CLANG_ATTRIBUTE(value)
#endif

#if OSL_GNUC_VERSION
    #define OSL_GNUC_ATTRIBUTE(value) __attribute__((value))
#else
    #define OSL_GNUC_ATTRIBUTE(value)
#endif


#ifndef OSL_DEBUG
    #ifdef NDEBUG
        #define OSL_DEBUG 0
    #else
        #ifdef _DEBUG
            #define OSL_DEBUG _DEBUG
        #else
            #define OSL_DEBUG 1
        #endif
    #endif
#endif // OSL_DEBUG


#if OSL_DEBUG
	// Disable OPENMP_SIMD when debugging
	#undef OSL_OPENMP_SIMD
#endif

#ifdef OSL_OPENMP_SIMD
    #define OSL_OMP_PRAGMA(aUnQuotedPragma) OSL_PRAGMA(aUnQuotedPragma)
#else
    #define OSL_OMP_PRAGMA(aUnQuotedPragma)
#endif

// OSL_FORCEINLINE is a function attribute that attempts to make the
// function always inline. On many compilers regular 'inline' is only
// advisory. Put this attribute before the function return type, just like
// you would use 'inline'. Note that if OSL_DEBUG is true, it just becomes
// ordinary inline.
#if OSL_DEBUG
#    define OSL_FORCEINLINE inline
#elif defined(__CUDACC__)
#    define OSL_FORCEINLINE __inline__
#elif defined(__GNUC__) || defined(__clang__) || __has_attribute(always_inline)
#    define OSL_FORCEINLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)
#    define OSL_FORCEINLINE __forceinline
#else
#    define OSL_FORCEINLINE inline
#endif


// OSL_NOINLINE hints to the compiler that the functions should never be
// inlined.
#if defined(__GNUC__) || defined(__clang__) || __has_attribute(noinline)
#    define OSL_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)
#    define OSL_NOINLINE __declspec(noinline)
#else
#    define OSL_NOINLINE
#endif

// OSL_FORCEINLINE_BLOCK tells the compiler that any calls in the next
// statement (or compound statement/code block)should be inlined whenever
// the compiler is capable of doing so.  Furthermore, any calls made by those
// calls should be inlined, and so on recursively.  Has no affect on function calls to
// whose definition exists in different translation units or calls that are not
// legal to inline (possibly by OSL_NOINLINE).  Intent is to ensure that entire
// contents of a SIMD loop body is are fully inlined avoiding compiler heuristics.
// However if any debugging code exists inside the code block, then the amount
// of code inlined could explode.  So great care should be used when placing
// code inside one of these forced inline blocks.  Note that if OSL_DEBUG is true,
// it does nothing to avoid over-inlining.
// NOTE:  only supported by Intel C++ compiler.
// Unsupported compilers may instead use command line flags to increase their inlining
// limits/thresholds
#if OSL_DEBUG
	#define OSL_FORCEINLINE_BLOCK
#else
	#define OSL_FORCEINLINE_BLOCK OSL_INTEL_PRAGMA(forceinline recursive)
#endif


// OSL_MAYBE_UNUSED is a function or variable attribute that assures the
// compiler that it's fine for the item to appear to be unused.
#if OSL_CPLUSPLUS_VERSION >= 17 || __has_cpp_attribute(maybe_unused)
#    define OSL_MAYBE_UNUSED [[maybe_unused]]
#elif defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER) || __has_attribute(unused)
#    define OSL_MAYBE_UNUSED __attribute__((unused))
#else
#    define OSL_MAYBE_UNUSED
#endif


// Some compilers define a special intrinsic to use in conditionals that can
// speed up extremely performance-critical spots if the conditional is
// usually (or rarely) is true.  You use it by replacing
//     if (x) ...
// with
//     if (OSL_LIKELY(x)) ...     // if you think x will usually be true
// or
//     if (OSL_UNLIKELY(x)) ...   // if you think x will rarely be true
// Caveat: Programmers are notoriously bad at guessing this, so it
// should be used only with thorough benchmarking.
#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#    define OSL_LIKELY(x) (__builtin_expect(bool(x), true))
#    define OSL_UNLIKELY(x) (__builtin_expect(bool(x), false))
#else
#    define OSL_LIKELY(x) (x)
#    define OSL_UNLIKELY(x) (x)
#endif


#ifndef OSL_HOSTDEVICE
#  ifdef __CUDACC__
#    define OSL_HOSTDEVICE __host__ __device__
#  else
#    define OSL_HOSTDEVICE
#  endif
#endif

#ifndef OSL_DEVICE
#  ifdef __CUDACC__
#    define OSL_DEVICE __device__
#  else
#    define OSL_DEVICE
#  endif
#endif

#ifndef OSL_CONSTANT_DATA
#  ifdef __CUDACC__
#    define OSL_CONSTANT_DATA __constant__
#  else
#    define OSL_CONSTANT_DATA
#  endif
#endif


// OSL_DEPRECATED before a function declaration marks it as deprecated in
// a way that will generate compile warnings if it is called (but will
// preserve linkage compatibility).
#if OSL_CPLUSPLUS_VERSION >= 14 || __has_cpp_attribute(deprecated)
#  define OSL_DEPRECATED(msg) [[deprecated(msg)]]
#elif defined(__GNUC__) || defined(__clang__) || __has_attribute(deprecated)
#  define OSL_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
#  define OSL_DEPRECATED(msg) __declspec(deprecated(msg))
#else
#  define OSL_DEPRECATED(msg)
#endif

// OSL_FALLTHROUGH at the end of a `case` label's statements documents that
// the switch statement case is intentionally falling through to the code for
// the next case.
#if OSL_CPLUSPLUS_VERSION >= 17 || __has_cpp_attribute(fallthrough)
#    define OSL_FALLTHROUGH [[fallthrough]]
#else
#    define OSL_FALLTHROUGH
#endif


// OSL_NODISCARD following a function declaration documents that the
// function's return value should never be ignored.
#if OSL_CPLUSPLUS_VERSION >= 17 || __has_cpp_attribute(nodiscard)
#    define OSL_NODISCARD [[nodiscard]]
#else
#    define OSL_NODISCARD
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
    (OSL_LIKELY(x)                                                             \
         ? ((void)0)                                                           \
         : (OSL_ASSERT_PRINT("%s:%u: %s: Assertion '%s' failed.\n",            \
                             __FILE__, __LINE__, OSL_PRETTY_FUNCTION, #x),     \
            OSL_ABORT_IF_DEBUG))
#define OSL_ASSERT_MSG(x, msg, ...)                                             \
    (OSL_LIKELY(x)                                                              \
         ? ((void)0)                                                            \
         : (std::fprintf(stderr, "%s:%u: %s: Assertion '%s' failed: " msg "\n", \
                         __FILE__, __LINE__, OSL_PRETTY_FUNCTION, #x,           \
                         __VA_ARGS__),                                          \
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

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    #define OSL_STACK_ARRAY(TYPE, NAME, DYNAMIC_COUNT) TYPE * NAME = reinterpret_cast<TYPE *>(alloca(sizeof(TYPE)*(DYNAMIC_COUNT)))
#else
    // Utilize variable length array compiler extension when available
    #define OSL_STACK_ARRAY(TYPE, NAME, DYNAMIC_COUNT) TYPE NAME[DYNAMIC_COUNT]
#endif

OSL_NAMESPACE_ENTER

#if OSL_CPLUSPLUS_VERSION >= 20
using std::assume_aligned;
#else

// Until we have c++20, we can provide our own version of std::assume_aligned
// http://open-std.org/JTC1/SC22/WG21/docs/papers/2018/p1007r0.pdf
template<int Alignment, class DataT>
OSL_NODISCARD
#ifndef __INTEL_COMPILER // constexpr not compatible with ICC's __assume_aligned
    constexpr
#endif
DataT* assume_aligned(DataT* ptr)
{
#ifdef __INTEL_COMPILER
    __assume_aligned(ptr, Alignment);
    return ptr;
#elif defined(__clang__) || defined(__GNUC__)
    return reinterpret_cast<DataT*>(__builtin_assume_aligned(ptr, Alignment));
#elif defined(_MSC_VER)
    constexpr std::uintptr_t unaligned_bit_mask = (Alignment - 1);
    if ((reinterpret_cast<std::uintptr_t>(ptr) & unaligned_bit_mask) == 0)
        return ptr;
    else
        __assume(0); // let compiler know we should never reach this branch
#else
    #error unknown compiler, update assume_aligned to handle (or explicitly ignore) emitting a pointer alignment hint
    return ptr;
#endif
}

#endif

static constexpr int MaxSupportedSimdLaneCount = 16;

OSL_NAMESPACE_EXIT
