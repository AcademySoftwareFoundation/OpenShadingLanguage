/*
Copyright (c) 2012 Sony Pictures Imageworks Inc., et al.
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

#include <OSL/oslconfig.h>
#include <OSL/dual_vec.h>
#include <OSL/wide.h>

OSL_NAMESPACE_ENTER


struct NoiseParams;

namespace pvt {

// Foward declaration, implementation is in liboslnoise/wide_gabornoise.h
struct DisabledFilterPolicy
{
	static constexpr bool active = false;
};

struct EnabledFilterPolicy
{
	static constexpr bool active = true;
};


template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE void
wide_gabor (
        MaskedAccessor<Dual2<float>,WidthT> wResult,
        ConstWideAccessor<Dual2<float>,WidthT> wX,
        NoiseParams const *opt);

// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE void
wide_gabor (
        MaskedAccessor<Dual2<float>,WidthT> wResult,
        ConstWideAccessor<Dual2<float>,WidthT> wX,
        ConstWideAccessor<Dual2<float>,WidthT> wY,
        NoiseParams const *opt);

template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE void
wide_gabor (
        MaskedAccessor<Dual2<float>,WidthT> wResult,
        ConstWideAccessor<Dual2<Vec3>,WidthT> wP,
        NoiseParams const *opt);

// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE  void
wide_gabor3 (
        MaskedAccessor<Dual2<Vec3>,WidthT> wResult,
        ConstWideAccessor<Dual2<float>, WidthT> wX,
        NoiseParams const *opt);

// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE  void
wide_gabor3 (
        MaskedAccessor<Dual2<Vec3>,WidthT> wResult,
        ConstWideAccessor<Dual2<float>, WidthT> wX,
        ConstWideAccessor<Dual2<float>, WidthT> wY,
        NoiseParams const *opt);

// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE  void
wide_gabor3 (
        MaskedAccessor<Dual2<Vec3>,WidthT> wResult,
		ConstWideAccessor<Dual2<Vec3>, WidthT> wP,
		NoiseParams const *opt);

// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE void
wide_pgabor (
        MaskedAccessor<Dual2<float>,WidthT> wResult,
        ConstWideAccessor<Dual2<float>,WidthT> wX,
        ConstWideAccessor<float,WidthT> wXp,
        NoiseParams const *opt);

// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE void
wide_pgabor (
        MaskedAccessor<Dual2<float>,WidthT> wResult,
        ConstWideAccessor<Dual2<float>,WidthT> wX,
        ConstWideAccessor<Dual2<float>,WidthT> wY,
        ConstWideAccessor<float,WidthT> wXp,
        ConstWideAccessor<float,WidthT> wYp,
        NoiseParams const *opt);

// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE void
wide_pgabor (
        MaskedAccessor<Dual2<float>,WidthT> wResult,
        ConstWideAccessor<Dual2<Vec3>,WidthT> wP,
        ConstWideAccessor<Vec3,WidthT> wPp,
        NoiseParams const *opt);






// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE void
wide_pgabor3 (
        MaskedAccessor<Dual2<Vec3>,WidthT> wResult,
        ConstWideAccessor<Dual2<float>,WidthT> wX,
        ConstWideAccessor<float,WidthT> wXp,
        NoiseParams const *opt);

// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE void
wide_pgabor3 (
        MaskedAccessor<Dual2<Vec3>,WidthT> wResult,
        ConstWideAccessor<Dual2<float>,WidthT> wX,
        ConstWideAccessor<Dual2<float>,WidthT> wY,
        ConstWideAccessor<float,WidthT> wXp,
        ConstWideAccessor<float,WidthT> wYp,
        NoiseParams const *opt);

// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE void
wide_pgabor3 (
        MaskedAccessor<Dual2<Vec3>,WidthT> wResult,
        ConstWideAccessor<Dual2<Vec3>,WidthT> wP,
        ConstWideAccessor<Vec3,WidthT> wPp,
        NoiseParams const *opt);




} // namespace pvt

OSL_NAMESPACE_EXIT
