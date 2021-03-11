// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#ifndef __OSL_WIDTH
#    error must define __OSL_WIDTH to number of SIMD lanes before including this header
#endif

#ifndef __OSL_TARGET_ISA
#    error must define __OSL_TARGET_ISA to AVX512, AVX2, AVX, SSE4_2, or x64 before including this header
#endif

#include <OSL/export.h>
// Prefix for batched OSL shade op declarations.
// Make them externally visibility, so their addresses
// can be dynamically discoverred (DLSYM).
// Also use "C" linkage (no C++ name mangling).
#define OSL_BATCHOP extern "C" OSL_DLL_EXPORT

// Macro helpers for xmacro include files
#define __OSL_EXPAND(A) A
#define __OSL_XMACRO_ARG1(A,...) A
#define __OSL_XMACRO_ARG2(A,B,...) B
#define __OSL_XMACRO_ARG3(A,B,C,...) C
#define __OSL_XMACRO_ARG4(A,B,C,D,...) D


#define __OSL_LIBRARY_SELECTOR \
    __OSL_CONCAT5(b, __OSL_WIDTH, _, __OSL_TARGET_ISA, _)


#define __OSL_OP(NAME) __OSL_CONCAT3(osl_, __OSL_LIBRARY_SELECTOR, NAME)
#define __OSL_MASKED_OP(NAME) \
    __OSL_CONCAT4(osl_, __OSL_LIBRARY_SELECTOR, NAME, _masked)

#define __OSL_OP1(NAME, A) \
    __OSL_CONCAT5(osl_, __OSL_LIBRARY_SELECTOR, NAME, _, A)
#define __OSL_MASKED_OP1(NAME, A) \
    __OSL_CONCAT6(osl_, __OSL_LIBRARY_SELECTOR, NAME, _, A, _masked)

#define __OSL_OP2(NAME, A, B) \
    __OSL_CONCAT6(osl_, __OSL_LIBRARY_SELECTOR, NAME, _, A, B)
#define __OSL_MASKED_OP2(NAME, A, B) \
    __OSL_CONCAT7(osl_, __OSL_LIBRARY_SELECTOR, NAME, _, A, B, _masked)

#define __OSL_OP3(NAME, A, B, C) \
    __OSL_CONCAT7(osl_, __OSL_LIBRARY_SELECTOR, NAME, _, A, B, C)
#define __OSL_MASKED_OP3(NAME, A, B, C) \
    __OSL_CONCAT8(osl_, __OSL_LIBRARY_SELECTOR, NAME, _, A, B, C, _masked)

#define __OSL_OP4(NAME, A, B, C, D) \
    __OSL_CONCAT8(osl_, __OSL_LIBRARY_SELECTOR, NAME, _, A, B, C, D)
#define __OSL_MASKED_OP4(NAME, A, B, C, D) \
    __OSL_CONCAT9(osl_, __OSL_LIBRARY_SELECTOR, NAME, _, A, B, C, D, _masked)

#define __OSL_OP5(NAME, A, B, C, D, E) \
    __OSL_CONCAT9(osl_, __OSL_LIBRARY_SELECTOR, NAME, _, A, B, C, D, E)
#define __OSL_MASKED_OP5(NAME, A, B, C, D, E)                            \
    __OSL_CONCAT10(osl_, __OSL_LIBRARY_SELECTOR, NAME, _, A, B, C, D, E, \
                   _masked)
