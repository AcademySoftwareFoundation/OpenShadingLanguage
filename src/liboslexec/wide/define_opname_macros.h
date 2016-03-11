/*
Copyright (c) 2009-2018 Sony Pictures Imageworks Inc., et al.
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

#ifndef __OSL_WIDTH
#    error must define __OSL_WIDTH to number of SIMD lanes before including this header
#endif

#ifndef __OSL_TARGET_ISA
#    error must define __OSL_TARGET_ISA to AVX512, AVX2, AVX, SSE4_2, or x64 before including this header
#endif


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
