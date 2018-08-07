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

Portions Copyright (c) 2017 Intel Inc., et al. All Rights Reserved.
*/

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of spline
/// operator
///
/////////////////////////////////////////////////////////////////////////



// If enabled, derivatives associated with the knot vectors are
// ignored
//#define SKIP_KNOT_DERIVS 1


#include <iostream>

#include <OpenImageIO/fmath.h>

#include "oslexec_pvt.h"
#include "OSL/dual_vec.h"
#include "splineimpl.h"
#include "sfm_staticmatrix.h"

using namespace std;
namespace {

// ========================================================
//
// Interpolation bases for splines
//
// ========================================================

static const int kNumSplineTypes = 6;
static const int kLinearSpline = kNumSplineTypes - 1;
static Spline::SplineBasis gBasisSet[kNumSplineTypes] = {
   { ustring("catmull-rom"), 1, Matrix44( (-1.0f/2.0f),  ( 3.0f/2.0f), (-3.0f/2.0f), ( 1.0f/2.0f),
                                          ( 2.0f/2.0f),  (-5.0f/2.0f), ( 4.0f/2.0f), (-1.0f/2.0f),
                                          (-1.0f/2.0f),  ( 0.0f/2.0f), ( 1.0f/2.0f), ( 0.0f/2.0f),
                                          ( 0.0f/2.0f),  ( 2.0f/2.0f), ( 0.0f/2.0f), ( 0.0f/2.0f))  },
   { ustring("bezier"),      3, Matrix44(  -1,  3, -3,  1,
                                            3, -6,  3,  0,
                                           -3,  3,  0,  0,
                                            1,  0,  0,  0) },
   { ustring("bspline"),     1, Matrix44( (-1.0f/6.0f), ( 3.0f/6.0f),  (-3.0f/6.0f),  (1.0f/6.0f),
                                          ( 3.0f/6.0f), (-6.0f/6.0f),  ( 3.0f/6.0f),  (0.0f/6.0f),
                                          (-3.0f/6.0f), ( 0.0f/6.0f),  ( 3.0f/6.0f),  (0.0f/6.0f),
                                          ( 1.0f/6.0f), ( 4.0f/6.0f),  ( 1.0f/6.0f),  (0.0f/6.0f)) },
   { ustring("hermite"),     2, Matrix44(  2,  1, -2,  1,
                                          -3, -2,  3, -1,
                                           0,  1,  0,  0,
                                           1,  0,  0,  0) },
   { ustring("linear"),      1, Matrix44(  0,  0,  0,  0,
                                           0,  0,  0,  0,
                                           0, -1,  1,  0,
                                           0,  1,  0,  0) },
   { ustring("constant"),    1, Matrix44(0.0f) }  // special marker for constant
};

};  // End anonymous namespace


OSL_NAMESPACE_ENTER

namespace pvt {

namespace fast {

template<class T>
OSL_INLINE void clamp_in_place (Dual2<T> &x, const Dual2<T> minv, const Dual2<T> maxv)
{
   const float xval = x.val();
   if (xval < minv.val()) x = minv;
   if (xval > maxv.val()) x = maxv;
}


OSL_INLINE void clamp_in_place(float &x, float minv, float maxv) {
    const float xval = x;
    if (xval < minv) x = minv;
    if (xval > maxv) x = maxv;
};

int getSplineBasisType(const ustring &basis_name)
{
    int basis_type = -1;
    for (basis_type = 0; basis_type < kNumSplineTypes &&
        basis_name != gBasisSet[basis_type].basis_name; basis_type++);
    // If unrecognizable spline type, then default to Linear
    if (basis_type == kNumSplineTypes)
        basis_type = kLinearSpline;

    return basis_type;
}

}
const Spline::SplineBasis *Spline::getSplineBasis(const ustring &basis_name)
{
    int basis_type = -1;
    for (basis_type = 0; basis_type < kNumSplineTypes &&
        basis_name != gBasisSet[basis_type].basis_name; basis_type++);
    // If unrecognizable spline type, then default to Linear
    if (basis_type == kNumSplineTypes)
        basis_type = kLinearSpline;

    return &gBasisSet[basis_type];
}



#define USTR(cstr) (*((ustring *)&cstr))
#define DFLOAT(x) (*(Dual2<Float> *)x)
#define DVEC(x) (*(Dual2<Vec3> *)x)

OSL_SHADEOP void  osl_spline_fff(void *out, const char *spline_, void *x,
                                 float *knots, int knot_count, int knot_arraylen)
{
   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<float, float, float, float, false>
      (spline, *(float *)out, *(float *)x, knots, knot_count, knot_arraylen);
}

namespace fast {

template <class K_T, bool IsBasisUConstantT, int BasisStepT, class MatrixT, class R_T, class X_T, class KArrayT>
OSL_INLINE
void spline_weighted_evaluate(
					 const MatrixT &M,
                     R_T &result,
                     X_T &xval,
                     KArrayT knots,
                     int knot_count)
{
#if __clang__
    // Clang was unhappy tyring to SIMD a loop with min/max on Dual2 return type
    // so instead use a function that works on a reference instead of returning
    X_T x(xval);
    fast::clamp_in_place(x, X_T(0.0), X_T(1.0));
#else
    X_T x = Spline::Clamp(xval, X_T(0.0), X_T(1.0));
#endif
    int nsegs = ((knot_count - 4) / BasisStepT) + 1;
    x = x*(float)nsegs;
    float seg_x = removeDerivatives(x);
    int segnum = (int)seg_x;
    if (segnum < 0)
        segnum = 0;
    if (segnum > (nsegs-1))
       segnum = nsegs-1;

    if (IsBasisUConstantT) {
        // Special case for "constant" basis
        R_T P = removeDerivatives (K_T(knots[segnum+1]));
        assignment (result, P);
        return;
    }
    // x is the position along segment 'segnum'
    x = x - float(segnum);
    int s = segnum*BasisStepT;

    // extract the knot elements

    K_T P0 = knots[s];
    K_T P1 = knots[s+1];
    K_T P2 = knots[s+2];
    K_T P3 = knots[s+3];

    auto tk0 = M.m00 * P0 +
            M.m01 * P1 +
            M.m02 * P2 +
            M.m03 * P3;

    auto tk1 = M.m10 * P0 +
            M.m11 * P1 +
            M.m12 * P2 +
            M.m13 * P3;

    auto tk2 = M.m20 * P0 +
            M.m21 * P1 +
            M.m22 * P2 +
            M.m23 * P3;

    auto tk3 = M.m30 * P0 +
            M.m31 * P1 +
            M.m32 * P2 +
            M.m33 * P3;

    R_T tresult = sfm::unproxy_element(((tk0*x + tk1)*x + tk2)*x + tk3);
    assignment(result, tresult);
}


template <
    bool IsBasisUConstantT,
    int BasisStepT,
    typename MatrixT,
    typename RAccessorT,
    typename XAccessorT,
    typename KAccessorT>
OSL_NOINLINE
void spline_evaluate_loop_over_wide(
    const MatrixT &M,
    RAccessorT wR,
    XAccessorT wX,
    KAccessorT wK)
{
    static constexpr int vec_width = RAccessorT::width;

    typedef typename XAccessorT::value_type X_Type;
    typedef typename RAccessorT::value_type R_Type;
    typedef typename KAccessorT::value_type K_Type;

    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(vec_width))
        OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(vec_width))
        for(int lane=0; lane < wR.width; ++lane) {
            X_Type x = wX[lane];
            auto knots = wK[lane];

            R_Type result;
            spline_weighted_evaluate<
                K_Type,
                IsBasisUConstantT,
                BasisStepT>(M, result, x, knots, knots.length());

            wR[lane] = result;
        }
    }
}

template <
	typename KAccessor_T,
	typename RAccessorT,
	typename XAccessorT>
void spline_evaluate_wide(
	RAccessorT wR,
	ustring spline_basis,
	XAccessorT wX,
	KAccessor_T wK
	)
{

	int basis_type = fast::getSplineBasisType(spline_basis);
	switch(basis_type)
	{
	case 0:  // catmull-rom
	{
        sfm::StaticMatrix44<-1, 3, -3, 1,
                            2, -5, 4, -1,
                            -1, 0, 1, 0,
                            0, 2, 0, 0,
                            2 /* divisor */> catmullRomWeights;
        spline_evaluate_loop_over_wide<
            false /*is_basis_u_constant */,
            1 /* basis_step */>
            (catmullRomWeights, wR, wX, wK);
        break;
	}
	case 1:  // bezier
	{
        sfm::StaticMatrix44<-1, 3, -3, 1,
                                    3, -6, 3, 0,
                                    -3, 3, 0, 0,
                                    1, 0, 0, 0, 1 /*divisor*/> bezierWeights;
        spline_evaluate_loop_over_wide<
            false /*is_basis_u_constant */,
            3 /* basis_step */>
            (bezierWeights, wR, wX, wK);
		break;

	}

	case 2:  // bspline
	{
        sfm::StaticMatrix44<-1, 3, -3, 1,
                                    3, -6, 3, 0,
                                    -3, 0, 3, 0,
                                    1, 4, 1, 0, 6 /*bspline*/> bsplineWeights;

        spline_evaluate_loop_over_wide<
            false /*is_basis_u_constant */,
            1 /* basis_step */>
            (bsplineWeights, wR, wX, wK);
		break;
	}
	case 3:  // hermite
	{
        sfm::StaticMatrix44<2, 1, -2, 1,
                                    -3, -2, 3, -1,
                                     0, 1, 0, 0,
                                     1, 0, 0, 0, 1 /*Divisor*/> hermiteWeights;

        spline_evaluate_loop_over_wide<
            false /*is_basis_u_constant */,
            2 /* basis_step */>
            (hermiteWeights, wR, wX, wK);
		break;
	}
	case 4:  // linear
	{
        sfm::StaticMatrix44< 0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    0, -1, 1, 0,
                                    0, 1, 0, 0, 1 /*Divisor*/> linearWeights;

        spline_evaluate_loop_over_wide<
            false /*is_basis_u_constant */,
            1 /* basis_step */>
            (linearWeights, wR, wX, wK);
		break;
	}

	case 5:  // constant
	{
        // NOTE:  when basis is constant the weights are ignored,
        // just pass in 0's for the compiler to ignore
        sfm::StaticMatrix44< 0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0, 1 /*Divisor*/> constantWeights;

        spline_evaluate_loop_over_wide<
        true /*is_basis_u_constant */,
            1 /* basis_step */>
            (constantWeights, wR, wX, wK);
		break;
	}

	default:
		ASSERT(0 && "unsupported spline basis");
		break;
	};
}


template <class RTYPE, class XTYPE, class KTYPE>
void spline_evaluate_scalar(
	RTYPE &result,
	ustring spline_basis,
	XTYPE x,
	KTYPE *knots,
	int knot_count)
{

	int basis_type = fast::getSplineBasisType(spline_basis);

	OSL_INTEL_PRAGMA(forceinline recursive)
	switch(basis_type)
	{

	case 0:  // catmull-rom
	{
		sfm::StaticMatrix44<-1, 3, -3, 1,
							2, -5, 4, -1,
							-1, 0, 1, 0,
							0, 2, 0, 0,
							2 /* divisor */> catmullRomWeights;

		spline_weighted_evaluate<KTYPE,
							  false /*is_basis_u_constant */,
							  1 /* basis_step */>
		   (catmullRomWeights, result, x, knots, knot_count);
		break;
	}



	case 1:  // bezier
	{

		sfm::StaticMatrix44<-1, 3, -3, 1,
									3, -6, 3, 0,
									-3, 3, 0, 0,
									1, 0, 0, 0, 1 /*divisor*/> bezierWeights;
		spline_weighted_evaluate<KTYPE,
							  false /*is_basis_u_constant */,
							  3 /* basis_step */>
		   (bezierWeights, result, x, knots, knot_count);
		break;
	}

	case 2:  // bspline
	{
		sfm::StaticMatrix44<-1, 3, -3, 1,
									3, -6, 3, 0,
									-3, 0, 3, 0,
									1, 4, 1, 0, 6 /*bspline*/> bsplineWeights;
		spline_weighted_evaluate<KTYPE,
							  false /*is_basis_u_constant */,
							  1 /* basis_step */>
		   (bsplineWeights, result, x, knots, knot_count);
		break;
	}

	case 3:  // hermite
	{
		sfm::StaticMatrix44<2, 1, -2, 1,
									-3, -2, 3, -1,
									 0, 1, 0, 0,
									 1, 0, 0, 0, 1 /*Divisor*/> hermiteWeights;

		spline_weighted_evaluate<KTYPE,
							  false /*is_basis_u_constant */,
							  2 /* basis_step */>
		   (hermiteWeights, result, x, knots, knot_count);
		break;
	}

	case 4:  // linear
	{
		sfm::StaticMatrix44< 0, 0, 0, 0,
									0, 0, 0, 0,
									0, -1, 1, 0,
									0, 1, 0, 0, 1 /*Divisor*/> linearWeights;
		spline_weighted_evaluate<KTYPE,
							  false /*is_basis_u_constant */,
							  1 /* basis_step */>
		   (linearWeights, result, x, knots, knot_count);
		break;
	}

	case 5:  // constant
	{
		// NOTE:  when basis is constant the weights are ignored,
		// just pass in 0's for the compiler to ignore
		sfm::StaticMatrix44< 0, 0, 0, 0,
								0, 0, 0, 0,
								0, 0, 0, 0,
								0, 0, 0, 0, 1 /*Divisor*/> constantWeights;
		spline_weighted_evaluate<KTYPE,
							  true /*is_basis_u_constant */,
							  1 /* basis_step */>
		   (constantWeights, result, x, knots, knot_count);
		break;
	}

	default:
		ASSERT(0 && "unsupported spline basis");
		break;
	};
}//spline_eval ends

} // namespace fast
OSL_SHADEOP void  osl_spline_w16fw16ff_masked(void *wout_, const char *spline_, void *wx_,
                                 float *knots, int knot_count, int knot_arraylen, unsigned int mask_value)
{



	fast::template spline_evaluate_wide(
        MaskedAccessor<float>(wout_, Mask(mask_value)),
        USTR(spline_),
        ConstWideAccessor<float>(wx_),
        ConstUniformUnboundedArrayAccessor<float>(knots, knot_count));
}

OSL_SHADEOP void  osl_spline_w16ffw16f(void *wout_, const char *spline_, void *wx_,
                                 float *knots, int knot_count, int knot_arraylen)
{
	//WideAccessor<Matrix44> mout(wout_, Mask(mask_value));

	fast::template spline_evaluate_wide(
        WideAccessor<float>(wout_),
        USTR(spline_),
        ConstUniformAccessor<float>(wx_),
        ConstWideUnboundArrayAccessor<float>(knots,knot_count));
}


OSL_SHADEOP void  osl_spline_w16ffw16f_masked(void *wout_, const char *spline_, void *wx_,
                                 float *knots, int knot_count, int knot_arraylen, unsigned int mask_value)
{
	fast::template spline_evaluate_wide(
	    MaskedAccessor<float>(wout_, Mask(mask_value)),
	    USTR(spline_),
	    ConstUniformAccessor<float> (wx_),
	    ConstWideUnboundArrayAccessor<float>(knots, knot_count));
}


//OSL_SHADEOP void  osl_spline_w16ffw16f_masked(void *wout_, const char *spline_, void *wx_,
//                                 float *knots, int knot_count, int knot_arraylen, unsigned int mask_value)
//{
//	MaskedAccessor<Matrix44> mout(wout_, Mask(mask_value));
//
//	fast::template spline_evaluate_wide
//	(WideAccessor<float>(mout), USTR(spline_), ConstUniformAccessor<float>(wx_), ConstWideUnboundArrayAccessor<float>(knots, knot_count));
//}

OSL_SHADEOP void osl_spline_w16fff_masked(
	void *wout_,
	const char *spline_,
	void *x,
	float *knots, int knot_count, int knot_arraylen, int mask_value)
{

//   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
//   float result;
//   Spline::spline_evaluate<float, float, float, float, false>
//      (spline, result, *(float *)x, knots, knot_count, knot_arraylen);
//   Wide<float>  & wr = *reinterpret_cast<Wide<float> *>(out);
//   wr.set_all(result);

	float scalar_result;
	fast::template spline_evaluate_scalar<
		float,
		float,
		float>
		(scalar_result, USTR(spline_), *reinterpret_cast<float *>(x), knots, knot_count);

	//Broadcast to a wide wout_
	MaskedAccessor<float> wr(wout_, Mask(mask_value));
	make_uniform(wr, scalar_result);
}

OSL_SHADEOP void  osl_spline_dfdfdf(void *out, const char *spline_, void *x,
                                    float *knots, int knot_count, int knot_arraylen)
{
   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
       Spline::spline_evaluate<Dual2<float>, Dual2<float>, Dual2<float>, float, true>
     (spline, DFLOAT(out), DFLOAT(x), knots, knot_count, knot_arraylen);
}

OSL_SHADEOP void osl_spline_w16dfw16dfw16df_masked(void *wout_, const char *spline_, void *wx_,
									float *knots, int knot_count, int knot_arraylen, unsigned int mask_value)
{

	fast::template spline_evaluate_wide(
        MaskedAccessor<Dual2<float>>(wout_, Mask(mask_value)),
	    USTR(spline_),
	    ConstWideAccessor<Dual2<float>>(wx_),
	    ConstWideUnboundArrayAccessor<Dual2<float>>(knots, knot_count));
}

OSL_SHADEOP void osl_spline_w16dfw16dfdf_masked(void *wout_, const char *spline_, void *wx_,
                                    float *knots, int knot_count, int knot_arraylen, unsigned int mask_value)
{

    fast::template spline_evaluate_wide(
        MaskedAccessor<Dual2<float>>(wout_, Mask(mask_value)),
        USTR(spline_),
        ConstWideAccessor<Dual2<float>>(wx_),
        ConstUniformUnboundedArrayAccessor<Dual2<float>>(knots, knot_count));
}


OSL_SHADEOP void osl_spline_w16dfdfw16df_masked(void *wout_, const char *spline_, void *wx_,
                                    float *knots, int knot_count, int knot_arraylen, unsigned int mask_value)
{

    fast::template spline_evaluate_wide(
        MaskedAccessor<Dual2<float>>(wout_, Mask(mask_value)),
        USTR(spline_),
        ConstUniformAccessor<Dual2<float>>(wx_),
        ConstWideUnboundArrayAccessor<Dual2<float>>(knots, knot_count));
}





//===========================================================================
OSL_SHADEOP void  osl_spline_dffdf(void *out, const char *spline_, void *x,
                                   float *knots, int knot_count, int knot_arraylen)
{



   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Dual2<float>, float, Dual2<float>, float, true>
      (spline, DFLOAT(out), *(float *)x, knots, knot_count, knot_arraylen);
}

OSL_SHADEOP void  osl_spline_w16dffw16df_masked(void *wout_, const char *spline_, void *wx_,
                                   float *knots, int knot_count, int knot_arraylen, unsigned int  mask_value)
{

//		fast::template spline_evaluate<
//		ConstWideUnboundArrayAccessor<Dual2<float>>, true>(
//	MaskedAccessor<Dual2<float>>(wout_, Mask(mask_value)),
//	USTR(spline_),
//	ConstUniformAccessor<float>(wx_),
//	knots, knot_count);
//
//
//		fast::template spline_evaluate<
//	ConstWideUnboundArrayAccessor<Dual2<float>>, true>(
//	MaskedAccessor<Dual2<float>>(wout_, Mask(mask_value)),
//	USTR(spline_),
//	ConstUniformAccessor<float>(wx_),
//	knots, knot_count);


    fast::template spline_evaluate_wide(
        MaskedAccessor<Dual2<float>>(wout_, Mask(mask_value)),
        USTR(spline_),
        ConstUniformAccessor<float>(wx_),
        ConstWideUnboundArrayAccessor<Dual2<float>>(knots, knot_count));
}



OSL_SHADEOP void  osl_spline_w16dfw16fw16df_masked(void *wout_, const char *spline_, void *wx_,
                                   float *knots, int knot_count, int knot_arraylen, unsigned int  mask_value)
{

    fast::template spline_evaluate_wide(
        MaskedAccessor<Dual2<float>>(wout_, Mask(mask_value)),
        USTR(spline_),
        ConstWideAccessor<float>(wx_),
        ConstWideUnboundArrayAccessor<Dual2<float>>(knots, knot_count));
}

//===========================================================================

OSL_SHADEOP void  osl_spline_dfdff(void *out, const char *spline_, void *x,
                                   float *knots, int knot_count, int knot_arraylen)
{


   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Dual2<float>, Dual2<float>, float, float, false>
      (spline, DFLOAT(out), DFLOAT(x), knots, knot_count, knot_arraylen);
}
OSL_SHADEOP void  osl_spline_w16dfw16dff_masked(void *wout_, const char *spline_, void *wx_,
                                   float *knots, int knot_count, int knot_arraylen, unsigned int mask_value)
{

	fast::template spline_evaluate_wide(
		MaskedAccessor<Dual2<float>>(wout_, Mask(mask_value)),
		USTR(spline_),
		ConstWideAccessor<Dual2<float>>(wx_),
		ConstUniformUnboundedArrayAccessor<float>(knots, knot_count));
}



OSL_SHADEOP void  osl_spline_w16fw16fw16f_masked(void *wout_, const char *spline_, void *wx_,
										  void *wknots_, int knot_count, int knot_arraylen, unsigned int mask_value)
{
	fast::template spline_evaluate_wide(
        MaskedAccessor<float>(wout_, Mask(mask_value)),
        USTR(spline_),
        ConstWideAccessor<float>(wx_),
        ConstWideUnboundArrayAccessor<float>(wknots_, knot_count));
}

/*
OSL_SHADEOP void  osl_spline_w16dfw16f(void *wout_, const char *spline_, void *wx_,
										  void *wknots_, int knot_count, int knot_arraylen)
{
	fast::template spline_evaluate_wide
	(WideAccessor<float>(wout_), USTR(spline_), ConstUniformAccessor<float>(wx_), ConstWideUnboundArrayAccessor<float>(wknots_, knot_count));
}
*/

/*
OSL_SHADEOP void  osl_spline_w16dfw16dfw16f(void *wout_, const char *spline_, void *wx_,
										  void *wknots_, int knot_count, int knot_arraylen)
{
	fast::template spline_evaluate_wide
	(WideAccessor<float>(wout_), USTR(spline_), ConstWideAccessor<float>(wx_), ConstWideUnboundArrayAccessor<float>(wknots_, knot_count));
}
*/
//=======================================================================
OSL_SHADEOP void  osl_spline_vfv(void *out, const char *spline_, void *x,
                                 Vec3 *knots, int knot_count, int knot_arraylen)
{

   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Vec3, float, Vec3, Vec3, false>
      (spline, *(Vec3 *)out, *(float *)x, knots, knot_count, knot_arraylen);
}

OSL_SHADEOP void  osl_spline_w16vw16fv_masked(void *wout_, const char *spline_, void *wx_,
                                 Vec3 *knots, int knot_count, int knot_arraylen, unsigned int mask_value)
{

	fast::template spline_evaluate_wide(
        MaskedAccessor<Vec3>(wout_, Mask(mask_value)),
        USTR(spline_),
        ConstWideAccessor<float>(wx_),
        ConstUniformUnboundedArrayAccessor<Vec3>(knots, knot_count));
}





OSL_SHADEOP void  osl_spline_w16vw16fw16v_masked(void *wout_, const char *spline_, void *wx_,
                                 Vec3 *knots, int knot_count, int knot_arraylen, unsigned int mask_value)
{

	fast::template spline_evaluate_wide(
        MaskedAccessor<Vec3>(wout_, Mask(mask_value)),
        USTR(spline_),
        ConstWideAccessor<float>(wx_),
        ConstWideUnboundArrayAccessor<Vec3>(knots, knot_count));
}




OSL_SHADEOP void  osl_spline_w16vfw16v_masked(void *wout_, const char *spline_, void *wx_,
                                 Vec3 *knots, int knot_count, int knot_arraylen, unsigned int mask_value)
{
	fast::template spline_evaluate_wide(
        MaskedAccessor<Vec3>(wout_, Mask(mask_value)),
        USTR(spline_),
        ConstUniformAccessor<float>(wx_),
        ConstWideUnboundArrayAccessor<Vec3>(knots, knot_count));

}
//=======================================================================

OSL_SHADEOP void  osl_spline_dvdfv(void *out, const char *spline_, void *x,
                                   Vec3 *knots, int knot_count, int knot_arraylen)
{



   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Dual2<Vec3>, Dual2<float>, Vec3, Vec3, false>
      (spline, DVEC(out), DFLOAT(x), knots, knot_count, knot_arraylen);
}


OSL_SHADEOP void osl_spline_w16dvw16dfv_masked (void *wout_, const char *spline_, void *wx_,
        Vec3 *knots, int knot_count, int knot_arraylen, unsigned int mask_value)
{



	fast::template spline_evaluate_wide(
        MaskedAccessor<Dual2<Vec3>>(wout_, Mask(mask_value)),
        USTR(spline_),
        ConstWideAccessor<Dual2<float>>(wx_),
        ConstUniformUnboundedArrayAccessor<Vec3>(knots, knot_count));
}




OSL_SHADEOP void osl_spline_w16dvw16dfw16v_masked (void *wout_, const char *spline_, void *wx_,
        Vec3 *knots, int knot_count, int knot_arraylen, unsigned int mask_value )
{

	//std::cout<<"Knot count is: "<<knot_count<<std::endl;
	fast::template spline_evaluate_wide(
        MaskedAccessor<Dual2<Vec3>>(wout_, Mask(mask_value)),
        USTR(spline_),
        ConstWideAccessor<Dual2<float>>(wx_),
        ConstWideUnboundArrayAccessor<Vec3>(knots, knot_count));
}


OSL_SHADEOP void osl_spline_w16dvdfw16v_masked (void *wout_, const char *spline_, void *wx_,
        Vec3 *knots, int knot_count, int knot_arraylen, unsigned int mask_value)
{
//
//	fast::template spline_evaluate_wide
//	(WideAccessor<Dual2<Vec3>>(wout_), USTR(spline_), ConstUniformAccessor<Dual2<float>>(wx_), ConstWideUnboundArrayAccessor<Vec3>(knots, knot_count));


    fast::template spline_evaluate_wide(
        MaskedAccessor<Dual2<Vec3>>(wout_, Mask(mask_value)),
        USTR(spline_),
        ConstUniformAccessor<Dual2<float>>(wx_),
        ConstWideUnboundArrayAccessor<Vec3>(knots, knot_count));
}

//=======================================================================
OSL_SHADEOP void  osl_spline_dvfdv(void *out, const char *spline_, void *x,
                                    Vec3 *knots, int knot_count, int knot_arraylen)
{

   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Dual2<Vec3>, float, Dual2<Vec3>, Vec3, true>
      (spline, DVEC(out), *(float *)x, knots, knot_count, knot_arraylen);
}

OSL_SHADEOP void  osl_spline_w16dvfw16dv_masked(void *wout_, const char *spline_, void *wx_,
                                    Vec3 *knots, int knot_count, int knot_arraylen, unsigned int mask_value)
{
    fast::template spline_evaluate_wide(
	    MaskedAccessor<Dual2<Vec3>>(wout_, Mask(mask_value)),
        USTR(spline_),
        ConstUniformAccessor<float>(wx_),
        ConstWideUnboundArrayAccessor<Dual2<Vec3>>(knots, knot_count));
}

OSL_SHADEOP void  osl_spline_w16dvw16fw16dv_masked(void *wout_, const char *spline_, void *wx_,
                                    Vec3 *knots, int knot_count, int knot_arraylen, unsigned int mask_value)
{
	fast::template spline_evaluate_wide(
        MaskedAccessor<Dual2<Vec3>>(wout_, Mask(mask_value)),
        USTR(spline_),
        ConstWideAccessor<float>(wx_),
        ConstWideUnboundArrayAccessor<Dual2<Vec3>>(knots, knot_count));
}

OSL_SHADEOP void  osl_spline_w16dvw16fdv_masked(void *wout_, const char *spline_, void *wx_,
                                    Vec3 *knots, int knot_count, int knot_arraylen, unsigned int mask_value)
{

	fast::template spline_evaluate_wide(
        MaskedAccessor<Dual2<Vec3>>(wout_, Mask(mask_value)),
        USTR(spline_),
        ConstWideAccessor<float>(wx_),
        ConstUniformUnboundedArrayAccessor<Dual2<Vec3>>(knots, knot_count));
}

//=======================================================================
OSL_SHADEOP void  osl_spline_dvdfdv(void *out, const char *spline_, void *x,
                                    Vec3 *knots, int knot_count, int knot_arraylen)
{

   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Dual2<Vec3>, Dual2<float>, Dual2<Vec3>, Vec3, true>
      (spline, DVEC(out), DFLOAT(x), knots, knot_count, knot_arraylen);
}

OSL_SHADEOP void  osl_spline_w16dvw16dfw16dv_masked(void *wout_, const char *spline_, void *wx_,
                                    Vec3 *knots, int knot_count, int knot_arraylen, unsigned int mask_value)
{
	fast::template spline_evaluate_wide(
        MaskedAccessor<Dual2<Vec3>>(wout_, Mask(mask_value)),
        USTR(spline_),
        ConstWideAccessor<Dual2<float>>(wx_),
        ConstWideUnboundArrayAccessor<Dual2<Vec3>>(knots, knot_count));
}

OSL_SHADEOP void  osl_spline_w16dvw16dfdv_masked(void *wout_, const char *spline_, void *wx_,
                                    Vec3 *knots, int knot_count, int knot_arraylen, unsigned int mask_value)
{
	fast::template spline_evaluate_wide(
        MaskedAccessor<Dual2<Vec3>>(wout_, Mask(mask_value)),
        USTR(spline_),
        ConstWideAccessor<Dual2<float>>(wx_),
        ConstUniformUnboundedArrayAccessor<Dual2<Vec3>>(knots, knot_count));
}

OSL_SHADEOP void  osl_spline_w16dvdfw16dv_masked(void *wout_, const char *spline_, void *wx_,
                                    Vec3 *knots, int knot_count, int knot_arraylen, unsigned int mask_value)
{
    fast::template spline_evaluate_wide(
        MaskedAccessor<Dual2<Vec3>>(wout_, Mask(mask_value)),
        USTR(spline_),
        ConstUniformAccessor<Dual2<float>>(wx_),
        ConstWideUnboundArrayAccessor<Dual2<Vec3>>(knots, knot_count));
}

OSL_SHADEOP void osl_splineinverse_fff(void *out, const char *spline_, void *x,
                                       float *knots, int knot_count, int knot_arraylen)
{
    // Version with no derivs

    const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
    Spline::spline_inverse<float> (spline, *(float *)out, *(float *)x, knots, knot_count, knot_arraylen);
}

OSL_SHADEOP void osl_splineinverse_w16fw16fw16f_masked(void *wout_, const char *spline_, void *wx_,
                                       void *wknots_, int knot_count, int /*knot_arraylen*/, unsigned int mask_value)
{
    // Version with no derivs

	const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
	ConstWideAccessor<float> wX(wx_);
	ConstWideUnboundArrayAccessor<float> wK (wknots_, knot_count);
    MaskedAccessor<float> wR(wout_, Mask(mask_value));

    for(int lane = 0; lane<wR.width; ++lane){

        if (wR.mask().is_on(lane)) {
    	float x = wX[lane];

    	auto knot_array = wK[lane];
    	float knots[knot_array.length()];
    	for (int k =0; k< knot_array.length(); ++k){
    	    knots[k] = knot_array[k];
    	}

    	float result;

		Spline::spline_inverse<float> (spline, result, x, knots, knot_array.length(), knot_array.length());
		wR[lane] = result;

        }


    }


  //  const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));

  //  Spline::spline_inverse<float> (spline, *(float *)out, *(float *)x, knots, knot_count, knot_arraylen);
}

#if 0
OSL_SHADEOP void  osl_spline_w16fw16ff(void *wout_, const char *spline_, void *wx_,
                                 float *knots, int knot_count, int knot_arraylen)
{
	//printf("Inside  osl_spline_w16fw16fff\n");
	const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
	ConstWideAccessor<float> wX(wx_);

	WideAccessor<float> wR(wout_);

	// calling a function below, don't bother vectorizing
	//OSL_INTEL_PRAGMA("novector")
	for(int lane=0; lane < wR.width; ++lane) {
	    float x = wX[lane];

	    // TODO: investigate removing this function call to enable SIMD
	    float result;
	    Spline::spline_evaluate<float, float, float, float, false>
	       (spline, result, x, knots, knot_count, knot_arraylen);

	    wR[lane] = result;
	}
}
#endif

OSL_SHADEOP void osl_splineinverse_w16fw16ff_masked(void *wout_, const char *spline_, void *wx_,
                                       void *wknots_, int knot_count, int /*knot_arraylen*/, unsigned int mask_value)
{
    // Version with no derivs

    const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
    ConstWideAccessor <float> wX(wx_);
    ConstUniformUnboundedArrayAccessor <float> wK (wknots_, knot_count);
    MaskedAccessor<float> wR(wout_, Mask (mask_value));

    for(int lane = 0; lane<wR.width; ++lane){
        if (wR.mask().is_on(lane)) {
            float x = wX[lane];

            auto knot_array = wK[lane];
            float knots[knot_array.length()];
            for(int k = 0; k<knot_array.length(); ++k){
                knots[k] = knot_array[k];
            }
            float result;

            Spline::spline_inverse<float> (spline, result,x, knots, knot_array.length(), knot_array.length());
        //	Spline::spline_inverse<float> (spline, MaskedAccessor<float>(wout_, Mask(mask_value)), x, knots, knot_count, knot_arraylen);
            wR[lane] = result;
        }
    }
  //  Spline::spline_inverse<float> (spline, *(float *)out, *(float *)x, knots, knot_count, knot_arraylen);
}

OSL_SHADEOP void osl_splineinverse_w16ffw16f_masked(void *wout_, const char *spline_, void *wx_,
                                       void *wknots_, int knot_count, int /*knot_arraylen*/, unsigned int mask_value)
{
    // Version with no derivs

   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   ConstUniformAccessor <float> wX(wx_);
   ConstWideUnboundArrayAccessor <float> wK (wknots_, knot_count);
   MaskedAccessor<float> wR(wout_, Mask (mask_value));

   for(int lane = 0; lane<wR.width; ++lane){
       if (wR.mask().is_on(lane)) {

		   float x = wX[lane];

		   auto knot_array = wK[lane];
		   float knots[knot_array.length()];
		   for(int k = 0; k< knot_array.length(); ++k) {
		       knots[k] = knot_array[k];
		   }

		   float result;
		   Spline::spline_inverse<float> (spline, result, x, knots, knot_array.length(), knot_array.length());
		   wR[lane] = result;
       }
   }
  //  Spline::spline_inverse<float> (spline, *(float *)out, *(float *)x, knots, knot_count, knot_arraylen);
}

OSL_SHADEOP void osl_splineinverse_w16fff_masked(void *wout_, const char *spline_, void *wx_,
                                       void *wknots_, int knot_count, int /*knot_arraylen*/, unsigned int mask_value)
{
    // Version with no derivs

   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   ConstUniformAccessor <float> wX(wx_);
   ConstUniformUnboundedArrayAccessor <float> wK (wknots_, knot_count);
   MaskedAccessor<float> wR(wout_, Mask (mask_value));

   for(int lane = 0; lane<wR.width; ++lane){
       if (wR.mask().is_on(lane)) {
           //float k = wK[lane];
           //float *kp = &k;]
           float x = wX[lane];
           auto knot_array = wK[lane];
           float knots[knot_array.length()];
           for(int k = 0; k< knot_array.length(); ++k) {
                knots[k] = knot_array[k];
                      }
           float result;
           Spline::spline_inverse<float> (spline, result, x, knots, knot_array.length(), knot_array.length());
           wR[lane] = result;
       }
   }
  //  Spline::spline_inverse<float> (spline, *(float *)out, *(float *)x, knots, knot_count, knot_arraylen);
}

OSL_SHADEOP void osl_splineinverse_dfdff(void *out, const char *spline_, void *x,
                                         float *knots, int knot_count, int knot_arraylen)
{
    // x has derivs, so return derivs as well

    const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
    Spline::spline_inverse<Dual2<float> > (spline, DFLOAT(out), DFLOAT(x), knots, knot_count, knot_arraylen);
}

OSL_SHADEOP void osl_splineinverse_w16dfw16dff_masked(void *wout_, const char *spline_, void *wx_,
                                         void *wknots_, int knot_count, int /*knot_arraylen*/, unsigned int mask_value)
{
    // x has derivs, so return derivs as well

	const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
    ConstWideAccessor<Dual2<float>> wX (wx_);
    ConstUniformUnboundedArrayAccessor<float> wK (wknots_, knot_count);
    MaskedAccessor<Dual2<float>> wR(wout_, Mask (mask_value));

    for(int lane = 0; lane<wR.width; ++lane){
        if (wR.mask().is_on(lane)) {
            Dual2<float> x = wX[lane]; //This has x, dx, and dy

            auto knot_array = wK[lane];
            float knots[knot_array.length()];

            for(int k = 0; k<knot_array.length(); ++k){
                knots[k] = knot_array[k];
            }

            Dual2<float> result;

            Spline::spline_inverse<Dual2<float> > (spline, result, x,
                    knots, knot_array.length(), knot_array.length());

            wR[lane] = result;
        }
    }


//    const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
//    Spline::spline_inverse<Dual2<float> > (spline, DFLOAT(out), DFLOAT(x), knots, knot_count, knot_arraylen);
}



OSL_SHADEOP void osl_splineinverse_dfdfdf(void *out, const char *spline_, void *x,
                                          float *knots, int knot_count, int knot_arraylen)
{

    // Ignore knot derivatives
    osl_splineinverse_dfdff (out, spline_, x, knots, knot_count, knot_arraylen);

}

//OSL_SHADEOP void osl_splineinverse_w16dfw16dfw16df_masked(void *wout_, const char *spline_, void *wx_,
//                                          float *knots, int knot_count, int knot_arraylen, unsigned int mask_value)
//{
//
//    // Ignore knot derivatives
//   // osl_splineinverse_dfdff (out, spline_, x, knots, knot_count, knot_arraylen);
//
//	    const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
//	    ConstWideAccessor<Dual2<float>> wX (wx_);
//	 //   WideAccessor<Dual2<float>> wR(wout_);
//	    MaskedAccessor<Dual2<float>> wR(wout_, Mask (mask_value));
//	  //  ConstWideAccessor<Dual2<float>> wK (knots);
//
//	    for(int lane = 0; lane<wR.width; ++lane){
//          if (wR.mask().is_on(lane)) {
    //	    	Dual2<float> x = wX[lane]; //This has x, dx, and dy
    //	    	//Dual2<float> k = wK[lane];
    //
    //	    	//Dual2<float> *kp = &k;
    //	    	Dual2<float> result;
    //
    //	    	Spline::spline_inverse<Dual2<float> > (spline, result, x, knots, knot_count, knot_arraylen);
    //
    //	    	wR[lane] = result;
//            }
//
//	    }
//}


OSL_SHADEOP void osl_splineinverse_w16dfw16dfw16df_masked(void *wout_, const char *spline_, void *wx_,
                                       void *wknots_, int knot_count, int /*knot_arraylen*/, unsigned int mask_value)
{
    // Version with no derivs

    const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
    ConstWideAccessor<Dual2<float>> wX(wx_);
    ConstWideUnboundArrayAccessor<Dual2<float>> wK(wknots_, knot_count); //Dual2 knots are treated as float knots
    MaskedAccessor<Dual2<float>> wR(wout_, Mask (mask_value));


    for(int lane = 0; lane<wR.width; ++lane){
        if (wR.mask().is_on(lane)) {

            Dual2<float> x = wX[lane];
            auto knot_array = wK[lane];
            float knots[knot_array.length()];
            for (int k=0; k < knot_array.length();++k) {
                // Ignore knot derivatives
                knots[k] = unproxy(knot_array[k]).val();
            }
          //  float *kp = &k;
            Dual2<float> result;

            Spline::spline_inverse<Dual2<float>> (spline, result, x, knots, knot_array.length(), knot_array.length());
            wR[lane] = result;
        }

    }

 }


OSL_SHADEOP void osl_splineinverse_w16dfw16dfdf_masked(void *wout_, const char *spline_, void *wx_,
                                          void *wknots_, int knot_count, int /*knot_arraylen*/, unsigned int mask_value)
{

    // Ignore knot derivatives
   // osl_splineinverse_dfdff (out, spline_, x, knots, knot_count, knot_arraylen);
//treated like dfdff

        const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
        ConstWideAccessor<Dual2<float>> wX (wx_);
        ConstUniformUnboundedArrayAccessor<Dual2<float>> wK (wknots_, knot_count);
        MaskedAccessor<Dual2<float>> wR(wout_, Mask (mask_value));

        for(int lane = 0; lane<wR.width; ++lane){
            if (wR.mask().is_on(lane)) {

                Dual2<float> x = wX[lane]; //This has x, dx, and dy
                auto knot_array = wK[lane];
                float knots[knot_array.length()];

                for(int k=0; k < knot_array.length();++k){
                    knots[k] = knot_array[k].val();
                }

                Dual2<float> result;

                Spline::spline_inverse<Dual2<float> > (spline, result, x, knots,
                        knot_array.length(), knot_array.length());

                wR[lane] = result;
            }
        }
}


OSL_SHADEOP void osl_splineinverse_w16dfdfw16df_masked(void *wout_, const char *spline_, void *wx_,
                                          void *wknots_, int knot_count, int /*knot_arraylen*/, unsigned int mask_value)
{

    // Ignore knot derivatives
   // osl_splineinverse_dfdff (out, spline_, x, knots, knot_count, knot_arraylen);

        const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
        ConstUniformAccessor<Dual2<float>> wX (wx_);
        ConstWideUnboundArrayAccessor<Dual2<float>> wK (wknots_, knot_count);
        MaskedAccessor<Dual2<float>> wR(wout_, Mask (mask_value));

        for(int lane = 0; lane<wR.width; ++lane){
            if (wR.mask().is_on(lane)) {
                Dual2<float> x = wX[lane]; //This has x, dx, and dy

                auto knot_array = wK[lane];
                float knots[knot_array.length()];
                for (int k=0; k < knot_array.length();++k) {
                knots[k] = unproxy(knot_array[k]).val();
                }


                Dual2<float> result;
                Spline::spline_inverse<Dual2<float>> (spline, result, x, knots, knot_array.length(), knot_array.length());
                wR[lane] = result;
            }
        }
}


OSL_SHADEOP void osl_splineinverse_dffdf(void *out, const char *spline_, void *x,
                                         float *knots, int knot_count, int knot_arraylen)
{

    // Ignore knot derivs
    float outtmp = 0;
    osl_splineinverse_fff (&outtmp, spline_, x, knots, knot_count, knot_arraylen);
    DFLOAT(out) = outtmp;
}

OSL_SHADEOP void osl_splineinverse_w16dffw16df_masked (void *wout_, const char *spline_, void *wx_,
void *wknots_, int knot_count, int /*knot_arraylen*/, unsigned int mask_value)
{
    // treated as fff

      const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
      ConstUniformAccessor<float> wX (wx_);
      ConstWideUnboundArrayAccessor <Dual2<float>> wK (wknots_, knot_count);
      MaskedAccessor<float> wR(wout_, Mask (mask_value));


      for(int lane = 0; lane<wR.width; ++lane){
          if (wR.mask().is_on(lane)) {
              float x = wX[lane];
              //Step1: Get knots array; whatever type they may be

              auto knot_array = wK[lane];

              //Step2: Put them in an array
              float knots[knot_array.length()];

              for(int k = 0; k<knot_array.length(); ++k){
                  knots[k] = unproxy(knot_array[k]).val();
              }

              float result; //The result is a dual2,  but I only need value.
              //Spline::spline_inverse<float> (spline, result, *(float *)wx_, kp, knot_count, knot_arraylen);
              Spline::spline_inverse<float> (spline, result, x,
                      knots, knot_array.length(), knot_array.length());
              wR[lane] = result;//We're not feeding it Duals so no chance of getting a dual op
          }
      }
}


OSL_SHADEOP void osl_splineinverse_w16dfw16fw16df_masked (void *wout_, const char *spline_, void *wx_,
float *knots, int knot_count, int /*knot_arraylen*/, unsigned int mask_value)
{
    // treated as fff

      const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
      ConstWideAccessor<Dual2<float>> wX (wx_);
      ConstWideUnboundArrayAccessor<Dual2<float>> wK (knots, knot_count);
      MaskedAccessor<float> wR(wout_, Mask (mask_value));


      for(int lane = 0; lane<wR.width; ++lane){
          if (wR.mask().is_on(lane)) {

             Dual2<float> d_x = wX[lane];
              float x = d_x.val();
              auto knot_array = wK[lane];
              float knots[knot_array.length()];
              for (int k = 0; k<knot_array.length(); ++k){
                  knots[k] = unproxy(knot_array[k]).val();
              }
              float result;
              //Spline::spline_inverse<float> (spline, result, *(float *)wx_, kp, knot_count, knot_arraylen);
              Spline::spline_inverse<float> (spline, result, x,
                      knots, knot_array.length(), knot_array.length());
              wR[lane] = result;
          }
      }
}

OSL_SHADEOP void osl_splineinverse_w16dfdffw16df_masked (void *wout_, const char *spline_, void *wx_,
void *wknots_, int knot_count, int /*knot_arraylen*/, unsigned int mask_value)
{
/*
      const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
      ConstWideAccessor<float> wX (wx_);
      ConstWideUnboundArrayAccessor <float> wK (wknots, knot_count);
      MaskedAccessor<float> wR(wout_, Mask (mask_value));


      for(int lane = 0; lane<wR.width; ++lane){
          if (wR.mask().is_on(lane)) {
              float x = wX[lane];
              float k = wK[lane];

              float result;
              //Spline::spline_inverse<float> (spline, result, *(float *)wx_, kp, knot_count, knot_arraylen);
              Spline::spline_inverse<float> (spline, result, x,
                      kp, knot_count, knot_arraylen);
              wR[lane] = result;
          }
      }
      */
}

} // namespace pvt
OSL_NAMESPACE_EXIT
