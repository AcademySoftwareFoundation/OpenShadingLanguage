// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause

// CUDA 13.2 added _NV_RSQRT_SPECIFIER to math_functions.hpp, but older Clang
// CUDA wrappers include that file without defining the macro first. Match the
// definition used by newer Clang wrappers, including glibc 2.42's exception
// specifier.
// Upstream fix: https://github.com/llvm/llvm-project/pull/185701
#ifndef _NV_RSQRT_SPECIFIER
#    if defined(__GNUC__) && defined(__GLIBC_PREREQ)
#        if __GLIBC_PREREQ(2, 42)
#            define _NV_RSQRT_SPECIFIER noexcept(true)
#        endif
#    endif
#    ifndef _NV_RSQRT_SPECIFIER
#        define _NV_RSQRT_SPECIFIER
#    endif
#endif

// Clang's CUDA wrapper intentionally includes this header more than once.
#include_next <crt/math_functions.hpp>
