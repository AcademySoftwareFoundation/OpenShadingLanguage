// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

/// \file
/// OSLPUBLIC and OSLEXPORT macros that are necessary for proper symbol
/// export when doing multi-platform development.


///
/// On Windows, when compiling code that will end up in a DLL, symbols
/// must be marked as 'exported' (i.e. __declspec(dllexport)) or they
/// won't be visible to programs linking against the DLL.
///
/// In addition, when compiling the application code that calls the DLL,
/// if a routine is marked as 'imported' (i.e. __declspec(dllimport)),
/// the compiler can be smart about eliminating a level of calling
/// indirection.  But you DON'T want to use __declspec(dllimport) when
/// calling a function from within its own DLL (it will still compile
/// correctly, just not with maximal efficiency).  Which is quite the
/// dilemma since the same header file is used by both the library and
/// its clients.  Sheesh!
///
/// But on Linux/OSX as well, we want to only have the DSO export the
/// symbols we designate as the public interface.  So we link with
/// -fvisibility=hidden to default to hiding the symbols.  See
/// http://gcc.gnu.org/wiki/Visibility
///
/// We solve this awful mess by defining these macros:
///
/// OSL*PUBLIC - normally, assumes that it's being seen by a client
///              of the library, and therefore declare as 'imported'.
///              But if OSL_EXPORT_PUBLIC is defined, change the declaration
///              to 'exported' -- you want to define this macro when
///              compiling the module that actually defines the class.
///
/// There is a separate define for each library, because there inter-
/// dependencies, and so what is exported for one may be imported for
/// another.

#if defined(_WIN32) || defined(__CYGWIN__)
#    if defined(OSL_STATIC_BUILD) || defined(OSL_STATIC_DEFINE) \
        || defined(__CUDACC__)
#        define OSL_DLL_IMPORT
#        define OSL_DLL_EXPORT
#        define OSL_DLL_LOCAL
#    else
#        define OSL_DLL_IMPORT __declspec(dllimport)
#        define OSL_DLL_EXPORT __declspec(dllexport)
#        define OSL_DLL_LOCAL
#    endif
//#define OSL_LLVM_EXPORT __declspec(dllexport)
#    define OSL_LLVM_EXPORT OSL_DLL_LOCAL
#else
#    define OSL_DLL_IMPORT  __attribute__((visibility("default")))
#    define OSL_DLL_EXPORT  __attribute__((visibility("default")))
#    define OSL_DLL_LOCAL   __attribute__((visibility("hidden")))
#    define OSL_LLVM_EXPORT OSL_DLL_LOCAL
#endif



#if defined(oslcomp_EXPORTS) || defined(oslexec_EXPORTS) || defined(OSL_EXPORTS)
#    define OSLCOMPPUBLIC OSL_DLL_EXPORT
#else
#    define OSLCOMPPUBLIC OSL_DLL_IMPORT
#endif

#if defined(oslexec_EXPORTS) || defined(OSL_EXPORTS)
#    define OSLEXECPUBLIC OSL_DLL_EXPORT
#else
#    define OSLEXECPUBLIC OSL_DLL_IMPORT
#endif

#if defined(oslquery_EXPORTS) || defined(oslexec_EXPORTS) \
    || defined(OSL_EXPORTS)
#    define OSLQUERYPUBLIC OSL_DLL_EXPORT
#else
#    define OSLQUERYPUBLIC OSL_DLL_IMPORT
#endif

#if defined(oslnoise_EXPORTS) || defined(oslexec_EXPORTS) \
    || defined(OSL_EXPORTS)
#    define OSLNOISEPUBLIC OSL_DLL_EXPORT
#else
#    define OSLNOISEPUBLIC OSL_DLL_IMPORT
#endif
