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

#include <OSL/wide_gabornoise_fwd.h>
#include "sfm_gabornoise.h"


OSL_NAMESPACE_ENTER

namespace pvt {

template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE  void
wide_gabor (
        MaskedAccessor<Dual2<float>,WidthT> wResult,
        ConstWideAccessor<Dual2<float>,WidthT> wX,
        NoiseParams const *opt)
{
    DASSERT (opt);

    OSL_INTEL_PRAGMA(forceinline recursive)
    {

        sfm::GaborUniformParams gup(*opt);

        // Complicated code caused compilation issues with icc17u2
        // but verified fixed in icc17u4
#if __INTEL_COMPILER >= 1700 && __INTEL_COMPILER_UPDATE >= 4
        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
#endif
        for(int i=0; i< WidthT; ++i) {
            const Dual2<float> x = wX[i];
            const Dual2<Vec3> P = make_Vec3(x);

            wResult[i] = sfm::scalar_gabor<AnisotropicT, FilterPolicyT>(P, gup);
        }
    }
}

template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE  void
wide_gabor (
        MaskedAccessor<Dual2<float>,WidthT> wResult,
        ConstWideAccessor<Dual2<float>,WidthT> wX,
        ConstWideAccessor<Dual2<float>,WidthT> wY,
        NoiseParams const *opt)
{
    DASSERT (opt);

    OSL_INTEL_PRAGMA(forceinline recursive)
    {

        sfm::GaborUniformParams gup(*opt);

        // Complicated code caused compilation issues with icc17u2
        // but verified fixed in icc17u4
#if __INTEL_COMPILER >= 1700 && __INTEL_COMPILER_UPDATE >= 4
        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
#endif
        for(int i=0; i< WidthT; ++i) {

            const Dual2<float> x = wX[i];
            const Dual2<float> y = wY[i];
            const Dual2<Vec3> P = make_Vec3(x, y);

            wResult[i] = sfm::scalar_gabor<AnisotropicT, FilterPolicyT>(P, gup);
        }
    }
}

template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE  void
wide_gabor (
        MaskedAccessor<Dual2<float>,WidthT> wResult,
        ConstWideAccessor<Dual2<Vec3>,WidthT> wP,
        NoiseParams const *opt)
{
    DASSERT (opt);

    OSL_INTEL_PRAGMA(forceinline recursive)
    {

        sfm::GaborUniformParams gup(*opt);

        // Complicated code caused compilation issues with icc17u2
        // but verified fixed in icc17u4
#if __INTEL_COMPILER >= 1700 && __INTEL_COMPILER_UPDATE >= 4
        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
#endif
        for(int i=0; i< WidthT; ++i) {
            const Dual2<Vec3> P = wP[i];
            wResult[i] = sfm::scalar_gabor<AnisotropicT, FilterPolicyT>(P, gup);
        }
    }

}


template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE /*OSL_CLANG_ATTRIBUTE(flatten)*/ void
wide_gabor3 (
        MaskedAccessor<Dual2<Vec3>,WidthT> wResult,
        ConstWideAccessor<Dual2<float>, WidthT> wX,
        NoiseParams const *opt)
{
    DASSERT (opt);

    OSL_INTEL_PRAGMA(forceinline recursive)
    {

        sfm::GaborUniformParams gup(*opt);

        // Complicated code caused compilation issues with icc17u2
        // but verified fixed in icc17u4
//#if (!defined(__INTEL_COMPILER)) || (__INTEL_COMPILER >= 1700 && __INTEL_COMPILER_UPDATE >= 4)
//        #ifdef __AVX512F__
//            OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
//        #else
//            // remark #15547: simd loop was not vectorized: code size was too large for vectorization. Consider reducing the number of distinct variables use
//            // So don't mandate interleaved loop unrolling by forcing a simdlen wider than ISA
//            OSL_OMP_PRAGMA(omp simd)
//        #endif
//#endif
        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
        for(int i=0; i< WidthT; ++i) {

            const Dual2<float> x = wX[i];
            const Dual2<Vec3> P = make_Vec3(x);

            wResult[i] = sfm::scalar_gabor3<AnisotropicT, FilterPolicyT>(P, gup);
        }
    }

}

template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE /*OSL_CLANG_ATTRIBUTE(flatten)*/ void
wide_gabor3 (
        MaskedAccessor<Dual2<Vec3>,WidthT> wResult,
        ConstWideAccessor<Dual2<float>, WidthT> wX,
        ConstWideAccessor<Dual2<float>, WidthT> wY,
        NoiseParams const *opt)
{
    DASSERT (opt);

    OSL_INTEL_PRAGMA(forceinline recursive)
    {

        sfm::GaborUniformParams gup(*opt);

//        #if !defined(__INTEL_COMPILER) || defined(__AVX512F__)
//            OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
//        #else
//            // remark #15547: simd loop was not vectorized: code size was too large for vectorization. Consider reducing the number of distinct variables use
//            // So don't mandate interleaved loop unrolling by forcing a simdlen wider than ISA
//            OSL_OMP_PRAGMA(omp simd)
//        #endif
        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
        for(int i=0; i< WidthT; ++i) {

            const Dual2<float> x = wX[i];
            const Dual2<float> y = wY[i];
            const Dual2<Vec3> P = make_Vec3(x, y);

            wResult[i] = sfm::scalar_gabor3<AnisotropicT, FilterPolicyT>(P, gup);
        }
    }
}

template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE /*OSL_CLANG_ATTRIBUTE(flatten)*/ void
wide_gabor3 (
        MaskedAccessor<Dual2<Vec3>,WidthT> wResult,
		ConstWideAccessor<Dual2<Vec3>, WidthT> wP,
		NoiseParams const *opt)
{
    DASSERT (opt);

	OSL_INTEL_PRAGMA(forceinline recursive)
	{

    	sfm::GaborUniformParams gup(*opt);

    	// Complicated code caused compilation issues with icc17u2
    	// but verified fixed in icc17u4
//#if (!defined(__INTEL_COMPILER)) || (__INTEL_COMPILER >= 1700 && __INTEL_COMPILER_UPDATE >= 4)
//		#ifdef __AVX512F__
//			OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
//		#else
//			// remark #15547: simd loop was not vectorized: code size was too large for vectorization. Consider reducing the number of distinct variables use
//			// So don't mandate interleaved loop unrolling by forcing a simdlen wider than ISA
//			OSL_OMP_PRAGMA(omp simd)
//		#endif
//#endif
        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
		for(int i=0; i< WidthT; ++i) {

			const Dual2<Vec3> P = wP[i];

            wResult[i] = sfm::scalar_gabor3<AnisotropicT, FilterPolicyT>(P, gup);
		}
	}

}


template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE  void
wide_pgabor (
        MaskedAccessor<Dual2<float>,WidthT> wResult,
        ConstWideAccessor<Dual2<float>,WidthT> wX,
        ConstWideAccessor<float,WidthT> wXp,
        NoiseParams const *opt)
{
    DASSERT (opt);

    OSL_INTEL_PRAGMA(forceinline recursive)
    {

        sfm::GaborUniformParams gup(*opt);

        // Complicated code caused compilation issues with icc17u2
        // but verified fixed in icc17u4
#if __INTEL_COMPILER >= 1700 && __INTEL_COMPILER_UPDATE >= 4
        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
#endif
        for(int i=0; i< WidthT; ++i) {

            const Dual2<float> x = wX[i];
            const Dual2<Vec3> P = make_Vec3(x);

            const float xperiod = wXp[i];
            Vec3 Pperiod(xperiod,0.0f,0.0f);

            wResult[i] = sfm::scalar_pgabor<AnisotropicT, FilterPolicyT>(P, Pperiod, gup);
        }
    }
}


template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE  void
wide_pgabor (
        MaskedAccessor<Dual2<float>,WidthT> wResult,
        ConstWideAccessor<Dual2<float>,WidthT> wX,
        ConstWideAccessor<Dual2<float>,WidthT> wY,
        ConstWideAccessor<float,WidthT> wXp,
        ConstWideAccessor<float,WidthT> wYp,
        NoiseParams const *opt)
{
    DASSERT (opt);

    // (make_Vec3(x,y), Vec3(xperiod,yperiod,0.0f), opt)

    DASSERT (opt);

    OSL_INTEL_PRAGMA(forceinline recursive)
    {

        sfm::GaborUniformParams gup(*opt);

        // Complicated code caused compilation issues with icc17u2
        // but verified fixed in icc17u4
#if __INTEL_COMPILER >= 1700 && __INTEL_COMPILER_UPDATE >= 4
        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
#endif
        for(int i=0; i< WidthT; ++i) {

            const Dual2<float> x = wX[i];
            const Dual2<float> y = wY[i];
            const Dual2<Vec3> P = make_Vec3(x,y);

            const float xperiod = wXp[i];
            const float yperiod = wYp[i];
            Vec3 Pperiod(xperiod,yperiod,0.0f);

            wResult[i] = sfm::scalar_pgabor<AnisotropicT, FilterPolicyT>(P, Pperiod, gup);
        }
    }
}

template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE  void
wide_pgabor (
        MaskedAccessor<Dual2<float>,WidthT> wResult,
        ConstWideAccessor<Dual2<Vec3>,WidthT> wP,
        ConstWideAccessor<Vec3,WidthT> wPp,
        NoiseParams const *opt)
{
    DASSERT (opt);

    OSL_INTEL_PRAGMA(forceinline recursive)
    {

        sfm::GaborUniformParams gup(*opt);

        // Complicated code caused compilation issues with icc17u2
        // but verified fixed in icc17u4
#if __INTEL_COMPILER >= 1700 && __INTEL_COMPILER_UPDATE >= 4
        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
#else
        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
#endif
        for(int i=0; i< WidthT; ++i) {
            const Dual2<Vec3> P = wP[i];
            Vec3 Pperiod = wPp[i];

            wResult[i] = sfm::scalar_pgabor<AnisotropicT, FilterPolicyT>(P, Pperiod, gup);
        }
    }
}



template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE  void
wide_pgabor3 (
        MaskedAccessor<Dual2<Vec3>,WidthT> wResult,
        ConstWideAccessor<Dual2<float>,WidthT> wX,
        ConstWideAccessor<float,WidthT> wXp,
        NoiseParams const *opt)
{
    DASSERT (opt);

    OSL_INTEL_PRAGMA(forceinline recursive)
    {

        sfm::GaborUniformParams gup(*opt);

        // Complicated code caused compilation issues with icc17u2
        // but verified fixed in icc17u4
#if __INTEL_COMPILER >= 1700 && __INTEL_COMPILER_UPDATE >= 4
        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
#endif
        for(int i=0; i< WidthT; ++i) {

            const Dual2<float> x = wX[i];
            const Dual2<Vec3> P = make_Vec3(x);

            const float xperiod = wXp[i];
            Vec3 Pperiod(xperiod,0.0f,0.0f);

            wResult[i] = sfm::scalar_pgabor3<AnisotropicT, FilterPolicyT>(P, Pperiod, gup);
        }
    }
}


template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE  void
wide_pgabor3 (
        MaskedAccessor<Dual2<Vec3>,WidthT> wResult,
        ConstWideAccessor<Dual2<float>,WidthT> wX,
        ConstWideAccessor<Dual2<float>,WidthT> wY,
        ConstWideAccessor<float,WidthT> wXp,
        ConstWideAccessor<float,WidthT> wYp,
        NoiseParams const *opt)
{
    DASSERT (opt);

    // (make_Vec3(x,y), Vec3(xperiod,yperiod,0.0f), opt)

    DASSERT (opt);

    OSL_INTEL_PRAGMA(forceinline recursive)
    {

        sfm::GaborUniformParams gup(*opt);

        // Complicated code caused compilation issues with icc17u2
        // but verified fixed in icc17u4
#if __INTEL_COMPILER >= 1700 && __INTEL_COMPILER_UPDATE >= 4
        //OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
#endif
        OSL_OMP_PRAGMA(omp simd)
        for(int i=0; i< WidthT; ++i) {

            const Dual2<float> x = wX[i];
            const Dual2<float> y = wY[i];
            const Dual2<Vec3> P = make_Vec3(x,y);

            const float xperiod = wXp[i];
            const float yperiod = wYp[i];
            Vec3 Pperiod(xperiod,yperiod,0.0f);

            wResult[i] = sfm::scalar_pgabor3<AnisotropicT, FilterPolicyT>(P, Pperiod, gup);
        }
    }
}

template<int AnisotropicT, typename FilterPolicyT, int WidthT>
OSL_NOINLINE  void
wide_pgabor3 (
        MaskedAccessor<Dual2<Vec3>,WidthT> wResult,
        ConstWideAccessor<Dual2<Vec3>,WidthT> wP,
        ConstWideAccessor<Vec3,WidthT> wPp,
        NoiseParams const *opt)
{
    DASSERT (opt);

    OSL_INTEL_PRAGMA(forceinline recursive)
    {

        sfm::GaborUniformParams gup(*opt);

        // Complicated code caused compilation issues with icc17u2
        // but verified fixed in icc17u4
#if __INTEL_COMPILER >= 1700 && __INTEL_COMPILER_UPDATE >= 4
        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
#else
        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
#endif
        for(int i=0; i< WidthT; ++i) {

            const Dual2<Vec3> P = wP[i];
            Vec3 Pperiod = wPp[i];

            wResult[i] = sfm::scalar_pgabor3<AnisotropicT, FilterPolicyT>(P, Pperiod, gup);
        }
    }
}



} // namespace pvt

OSL_NAMESPACE_EXIT
