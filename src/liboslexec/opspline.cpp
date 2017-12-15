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

OSL_SHADEOP void  osl_spline_w16fff(void *out, const char *spline_, void *x,
                                 float *knots, int knot_count, int knot_arraylen)
{

   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   float result;
   Spline::spline_evaluate<float, float, float, float, false>
      (spline, result, *(float *)x, knots, knot_count, knot_arraylen);
   Wide<float>  & wr = *reinterpret_cast<Wide<float> *>(out);
   wr.set_all(result);
}






namespace fast {

template <int MT, int DivisorT>
struct Multiplier {
	static float multiply(float value) { return value*(static_cast<float>(MT)/static_cast<float>(DivisorT)) ; }

};

template <int DivisorT>
struct Multiplier<0, DivisorT> {
	static float multiply(float value) { return 0.0f; }

};

template <int DivisorT>
struct Multiplier<DivisorT, DivisorT> {
	static float multiply(float value) { return value; }

};

template <int ValueT>
struct Negative{
	static constexpr int value=-ValueT;
};



template <int MT, int DivisorT>
struct ProxyElement {

	OSL_INLINE float operator * (float value) const
	{
		return value*(static_cast<float>(MT)/static_cast<float>(DivisorT));
	}

	OSL_INLINE float to_float() const { return static_cast<float>(MT)/static_cast<float>(DivisorT); }
};

template <int DivisorT>
struct ProxyElement<0, DivisorT> {
	OSL_INLINE ProxyElement<0, DivisorT> operator * (float) const
	{
		return ProxyElement<0, DivisorT>();
	}

	OSL_INLINE float operator + (float value) const
	{
		return value;
	}

	template<int OtherMt, int OtherDivisorT>
	OSL_INLINE ProxyElement<OtherMt, OtherDivisorT> operator +
	(ProxyElement<OtherMt, OtherDivisorT>) const
	{
		return ProxyElement<OtherMt, OtherDivisorT>();
	}

	OSL_INLINE float to_float() const { return 0.0f; }

};

template <int MT>
struct ProxyElement<MT, 1> {

	OSL_INLINE float operator * (float value)
	{
		return static_cast<float>(MT)*value;
	}


	template<int OtherDivisorT>
	OSL_INLINE float operator + (ProxyElement<OtherDivisorT, OtherDivisorT>) const
	{
		return static_cast<float>(MT + 1);
	}

	template<int OtherMT>
	OSL_INLINE ProxyElement<OtherMT+MT, 1> operator + (ProxyElement<OtherMT, 1>) const
	{
		return ProxyElement<OtherMT+MT, 1>();
	}

	template<int OtherDivisorT>
	OSL_INLINE ProxyElement<MT, 1> operator + (ProxyElement<0, OtherDivisorT>) const
	{
		return ProxyElement<MT, 1>();
	}

	OSL_INLINE float to_float() const { return static_cast<float>(MT); }

};


template <>
struct ProxyElement<0, 1> {
	OSL_INLINE ProxyElement<0, 1> operator * (float) const
	{
		return ProxyElement<0, 1>();
	}

	OSL_INLINE float operator + (float value) const
	{
		return value;
	}

	template<int OtherMt, int OtherDivisorT>
	OSL_INLINE ProxyElement<OtherMt, OtherDivisorT> operator +
	(ProxyElement<OtherMt, OtherDivisorT>) const
	{
		return ProxyElement<OtherMt, OtherDivisorT>();
	}

	OSL_INLINE float to_float() const { return 0.0f; }

};

template <int DivisorT>
struct ProxyElement<DivisorT, DivisorT> {
	OSL_INLINE float operator * (float value) const
	{
		return value;
	}

	OSL_INLINE float operator + (float value) const
	{
		return 1.0f + value;//
	}

	template<int OtherDivisorT>
	OSL_INLINE ProxyElement<2, 1> operator + (ProxyElement<OtherDivisorT, OtherDivisorT>) const
	{
		return ProxyElement<2, 1>();
	}

	template<int OtherDivisorT>
	OSL_INLINE ProxyElement<0, DivisorT> operator + (ProxyElement<0, OtherDivisorT>) const
	{
		return ProxyElement<0, DivisorT>;
	}

	OSL_INLINE float to_float() const { return 1.0f; }

};
//
template <>
struct ProxyElement<1, 1> {

	OSL_INLINE float
	operator * (float value) const
	{
		return value;
	}

	OSL_INLINE float
	operator + (float value) const
	{
		return 1.0f + value;
	}

	template<int OtherDivisorT>
	OSL_INLINE ProxyElement<2, 1>
	operator + (ProxyElement<OtherDivisorT, OtherDivisorT>) const
	{
		return ProxyElement<2, 1>();
	}

	template<int OtherDivisorT>
	OSL_INLINE
	ProxyElement<0, 1> operator + (ProxyElement<0, OtherDivisorT>) const
	{
		return ProxyElement<0, 1>;
	}

	OSL_INLINE float
	to_float() const { return 1.0f; }

};


template <>
struct ProxyElement<-1, 1> {

	OSL_INLINE float
	operator * (float value) const
	{
		return -value;
	}

	OSL_INLINE float
	operator + (float value) const
	{
		return -1.0f + value;
	}//

	template<int OtherDivisorT>
	OSL_INLINE ProxyElement<0, 1>
	operator + (ProxyElement<OtherDivisorT, OtherDivisorT>) const
	{
		return ProxyElement<0, 1>();
	}

	template<int OtherDivisorT>
	OSL_INLINE
	ProxyElement<-1, 1> operator + (ProxyElement<0, OtherDivisorT>) const
	{
		return ProxyElement<-1>;
	}

	OSL_INLINE float
	to_float() const { return -1.0f; }

};

//Vec3
template<typename ProxyElementX_T, typename ProxyElementY_T, typename ProxyElementZ_T>
struct ProxyVec3
{
	ProxyElementX_T x;
	ProxyElementY_T y;
	ProxyElementZ_T z;
};

template<typename ProxyElementX_T, typename ProxyElementY_T, typename ProxyElementZ_T>
OSL_INLINE ProxyVec3<ProxyElementX_T, ProxyElementY_T, ProxyElementZ_T>
makeProxyVec3(ProxyElementX_T x, ProxyElementY_T y, ProxyElementZ_T z)
{
	ProxyVec3<ProxyElementX_T, ProxyElementY_T, ProxyElementZ_T> r = {x,y,z};
	return r;
}

OSL_INLINE
Vec3
makeProxyVec3(float x, float y, float z)
{
	return Vec3(x,y,z);
}

template <int MT, int DivisorT>
OSL_INLINE
decltype(std::declval<ProxyElement<MT, DivisorT>>() + std::declval<float>())
operator + (float a, ProxyElement<MT, DivisorT> b)
{
	return b + a;
}

template <int MT, int DivisorT>
OSL_INLINE
decltype(std::declval<ProxyElement<MT, DivisorT>>() + std::declval<Vec3>())
operator + (Vec3 a, ProxyElement<MT, DivisorT> b)
{
	return b + a;
}


template<typename XT, typename YT, typename ZT>
OSL_INLINE
decltype(std::declval<ProxyVec3<XT, YT, ZT>>() + std::declval<Vec3>())
operator + (Vec3 a, ProxyVec3<XT, YT, ZT> b)
{
	return b + a;
}

template <int MT, int DivisorT>
OSL_INLINE
decltype(std::declval<ProxyElement<MT, DivisorT>>() * std::declval<float>())
operator * (float a, ProxyElement<MT, DivisorT> b)
{
	return b * a;
}

template <int MT, int DivisorT>
OSL_INLINE
decltype(std::declval<ProxyElement<MT, DivisorT>>() * std::declval<Vec3>())
operator * (Vec3 a, ProxyElement<MT, DivisorT> b)
{
	return b * a;
}

OSL_INLINE
float unproxy_element(float value)
{
	return value;
}



OSL_INLINE
Vec3 unproxy_element(const Vec3 &value)
{
	return value;
}

template<typename ProxyElementX_T, typename ProxyElementY_T, typename ProxyElementZ_T>
OSL_INLINE Vec3
unproxy_element(const ProxyVec3<ProxyElementX_T, ProxyElementY_T, ProxyElementZ_T> &value)
{
	return Vec3(unproxy_element(value.x), unproxy_element(value.y), unproxy_element(value.z));
}


template <int MT, int DivisorT>
OSL_INLINE
float unproxy_element(ProxyElement<MT, DivisorT> value)
{
	return value.to_float();
}



//Specialize operators for Vec3 to interact with ProxyElements:

template <int MT, int DivisorT>
OSL_INLINE decltype(makeProxyVec3 (std::declval<ProxyElement<MT, DivisorT>>()*float(),
		                       std::declval<ProxyElement<MT, DivisorT>>()*float(),
		                       std::declval<ProxyElement<MT, DivisorT>>()*float()))
operator* (ProxyElement<MT, DivisorT> a, const Vec3 &b)
{
    return makeProxyVec3 (a*b.x,
                     	  a*b.y,
						  a*b.z);
}


template<int MT, int DivisorT>
OSL_INLINE decltype(makeProxyVec3 (std::declval<ProxyElement<MT, DivisorT>>()+float(),
        std::declval<ProxyElement<MT, DivisorT>>()+float(),
        std::declval<ProxyElement<MT, DivisorT>>()+float()))

operator+ (ProxyElement<MT, DivisorT> a, const Vec3 &b)
{
    return makeProxyVec3 (a + b.x, a + b.y, a + b.z);
}

template<typename XT, typename YT, typename ZT>
OSL_INLINE decltype(makeProxyVec3 (std::declval<XT>()+float(),
        std::declval<YT>()+float(),
        std::declval<ZT>()+float()))

operator+ (ProxyVec3<XT, YT, ZT> a, const Vec3 &b)
{
    return makeProxyVec3 (a.x + b.x, a.y + b.y, a.z + b.z);
}


template<typename XT, typename YT, typename ZT,
         typename X2T, typename Y2T, typename Z2T>
OSL_INLINE decltype(makeProxyVec3 (std::declval<XT>()+std::declval<X2T>(),
        std::declval<YT>()+std::declval<Y2T>(),
        std::declval<ZT>()+std::declval<Z2T>()))

operator+ (ProxyVec3<XT, YT, ZT> a, const ProxyVec3<X2T, Y2T, Z2T> &b)
{
    return makeProxyVec3 (a.x + b.x, a.y + b.y, a.z + b.z);
}

template<typename XT, typename YT, typename ZT>
OSL_INLINE decltype(makeProxyVec3 (std::declval<XT>()*float(),
        std::declval<YT>()*float(),
        std::declval<ZT>()*float()))

operator* (ProxyVec3<XT, YT, ZT> a, float b)
{
    return makeProxyVec3 (a.x*b, a.y*b, a.z*b);
}

//Dual2

// Specialize operators for Dual2 to interact with ProxyElements
template <class T, int MT, int DivisorT>
OSL_INLINE
Dual2<T> operator* (ProxyElement<MT, DivisorT> b, const Dual2<T> &a)
{
    return Dual2<T> (unproxy_element(a.val()*b),
                     unproxy_element(a.dx()*b),
					 unproxy_element(a.dy()*b));
}


template<class T, int MT, int DivisorT>
OSL_INLINE
Dual2<T> operator+ (const Dual2<T> &a, ProxyElement<MT, DivisorT> b)
{
    return Dual2<T> (a.val()+b, a.dx(), a.dy());
}

OSL_INLINE
Dual2<float> unproxy_element(const Dual2<float> &value)
{
	return value;
}

template <int M00, int M01, int M02, int M03,
          int M10, int M11, int M12, int M13,
		  int M20, int M21, int M22, int M23,
		  int M30, int M31, int M32, int M33, int DivisorT>
struct StaticMatrix44
{
	static ProxyElement<M00,DivisorT> m00;
	static ProxyElement<M01,DivisorT> m01;
	static ProxyElement<M02,DivisorT> m02;
	static ProxyElement<M03,DivisorT> m03;

	static ProxyElement<M10,DivisorT> m10;
	static ProxyElement<M11,DivisorT> m11;
	static ProxyElement<M12,DivisorT> m12;
	static ProxyElement<M13,DivisorT> m13;

	static ProxyElement<M20,DivisorT> m20;
	static ProxyElement<M21,DivisorT> m21;
	static ProxyElement<M22,DivisorT> m22;
	static ProxyElement<M23,DivisorT> m23;

	static ProxyElement<M30,DivisorT> m30;
	static ProxyElement<M31,DivisorT> m31;
	static ProxyElement<M32,DivisorT> m32;
	static ProxyElement<M33,DivisorT> m33;

};




template <class RTYPE, class XTYPE, class CTYPE, class KTYPE, bool knot_derivs, bool is_basis_u_constant, int basis_step, class MatrixType>
void spline_weighted_evaluate(
					 const MatrixType &M,
                     RTYPE &result,
                     XTYPE &xval,
                     const KTYPE *knots,
                     int knot_count)
{
    XTYPE x = Spline::Clamp(xval, XTYPE(0.0), XTYPE(1.0));
    int nsegs = ((knot_count - 4) / basis_step) + 1;
    x = x*(float)nsegs;
    float seg_x = removeDerivatives(x);
    int segnum = (int)seg_x;
    if (segnum < 0)
        segnum = 0;
    if (segnum > (nsegs-1))
       segnum = nsegs-1;


    if (is_basis_u_constant) {
        // Special case for "constant" basis
        RTYPE P = removeDerivatives (knots[segnum+1]);
        assignment (result, P);
        return;
    }

    // x is the position along segment 'segnum'
    x = x - float(segnum);
    int s = segnum*basis_step;



    // create a functor so we can cleanly(!) extract
    // the knot elements
    Spline::extractValueFromArray<CTYPE, KTYPE, knot_derivs> myExtract;
    CTYPE P[4];

    P[0] = knots[s];
    P[1] = knots[s+1];
    P[2] = knots[s+2];
    P[3] = knots[s+3];
   // P[0] = myExtract(knots, knot_arraylen, s + 0);
//    P[1] = myExtract(knots, knot_arraylen, s + 1);
//    P[2] = myExtract(knots, knot_arraylen, s + 2);
//    P[3] = myExtract(knots, knot_arraylen, s + 3);

        auto tk0 = M.m00 * P[0] +
        		M.m01 * P[1] +
				M.m02 * P[2] +
				M.m03 * P[3];
        auto tk1 = M.m10 * P[0] +
        		M.m11 * P[1] +
				M.m12 * P[2] +
				M.m13 * P[3];
        auto tk2 = M.m20 * P[0] +
        		M.m21 * P[1] +
				M.m22 * P[2] +
				M.m23 * P[3];


        auto tk3 = M.m30 * P[0] +
        		M.m31 * P[1] +
				M.m32 * P[2] +
				M.m33 * P[3];


    RTYPE tresult;

    tresult = unproxy_element(((tk0*x + tk1)*x + tk2)*x + tk3);
    assignment(result, tresult);
}

template <class RTYPE, class XTYPE, class CTYPE, class KTYPE, bool knot_derivs>
void spline_evaluate(
	void *wout_,
	ustring spline_basis,
	void *wx_,
	KTYPE *knots,
	int knot_count)
{

	int basis_type = fast::getSplineBasisType(spline_basis);
	static constexpr int vec_width = WideAccessor<RTYPE>::width;
	OSL_INTEL_PRAGMA("forceinline recursive")
	switch(basis_type)
	{

	case 0:  // catmull-rom
	{
		ConstWideAccessor<XTYPE> wX(wx_);
		WideAccessor<RTYPE> wR(wout_);
		OSL_INTEL_PRAGMA("nofusion")
		OSL_INTEL_PRAGMA("omp simd simdlen(vec_width)")
		for(int lane=0; lane < wR.width; ++lane) {
			XTYPE x = wX[lane];

			fast::StaticMatrix44<-1, 3, -3, 1,
								2, -5, 4, -1,
								-1, 0, 1, 0,
								0, 2, 0, 0,
								2 /* divisor */> catmullRomWeights;

			RTYPE result;
			spline_weighted_evaluate<RTYPE, XTYPE, CTYPE, KTYPE,
								  false /* knot_derivs */,
								  false /*is_basis_u_constant */,
								  1 /* basis_step */>
			   (catmullRomWeights, result, x, knots, knot_count);

			wR[lane] = result;
		}
		break;
	}



	case 1:  // bezier
	{
		//std::cout<<"Inside Dual bezier"<<std::endl;
		ConstWideAccessor<XTYPE> wX(wx_);
		WideAccessor<RTYPE> wR(wout_);
		OSL_INTEL_PRAGMA("nofusion")
		OSL_INTEL_PRAGMA("omp simd simdlen(vec_width)")
		for(int lane=0; lane < wR.width; ++lane) {
			XTYPE x = wX[lane];


			fast::StaticMatrix44<-1, 3, -3, 1,
										3, -6, 3, 0,
										-3, 3, 0, 0,
										1, 0, 0, 0, 1 /*divisor*/> bezierWeights;
			RTYPE result;
			spline_weighted_evaluate<RTYPE, XTYPE, CTYPE, KTYPE,
								  false /* knot_derivs */,
								  false /*is_basis_u_constant */,
								  3 /* basis_step */>
			   (bezierWeights, result, x, knots, knot_count);


			wR[lane] = result;
		}
		break;

	}

	case 2:  // bspline
	{
		ConstWideAccessor<XTYPE> wX(wx_);
		WideAccessor<RTYPE> wR(wout_);
		OSL_INTEL_PRAGMA("nofusion")
		OSL_INTEL_PRAGMA("omp simd simdlen(vec_width)")
		for(int lane=0; lane < wR.width; ++lane) {
			XTYPE x = wX[lane];



			fast::StaticMatrix44<-1, 3, -3, 1,
										3, -6, 3, 0,
										-3, 0, 3, 0,
										1, 4, 1, 0, 6 /*bspline*/> bsplineWeights;
			RTYPE result;
			spline_weighted_evaluate<RTYPE, XTYPE, CTYPE, KTYPE,
								  false /* knot_derivs */,
								  false /*is_basis_u_constant */,
								  1 /* basis_step */>
			   (bsplineWeights, result, x, knots, knot_count);

			wR[lane] = result;
		}
		break;
	}

	case 3:  // hermite
	{
		ConstWideAccessor<XTYPE> wX(wx_);
		WideAccessor<RTYPE> wR(wout_);
		OSL_INTEL_PRAGMA("nofusion")
		OSL_INTEL_PRAGMA("omp simd simdlen(vec_width)")
		for(int lane=0; lane < wR.width; ++lane) {
			XTYPE x = wX[lane];


			fast::StaticMatrix44<2, 1, -2, 1,
										-3, -2, 3, -1,
										 0, 1, 0, 0,
										 1, 0, 0, 0, 1 /*Divisor*/> hermiteWeights;

			RTYPE result;
			spline_weighted_evaluate<RTYPE, XTYPE, CTYPE, KTYPE,
								  false /* knot_derivs */,
								  false /*is_basis_u_constant */,
								  2 /* basis_step */>
			   (hermiteWeights, result, x, knots, knot_count);

			wR[lane] = result;
		}
		break;
	}

	case 4:  // linear
	{
		ConstWideAccessor<XTYPE> wX(wx_);
		WideAccessor<RTYPE> wR(wout_);
		OSL_INTEL_PRAGMA("nofusion")
		OSL_INTEL_PRAGMA("omp simd simdlen(vec_width)")
		for(int lane=0; lane < wR.width; ++lane) {
			XTYPE x = wX[lane];



			fast::StaticMatrix44< 0, 0, 0, 0,
										0, 0, 0, 0,
										0, -1, 1, 0,
										0, 1, 0, 0, 1 /*Divisor*/> linearWeights;

			RTYPE result;

			spline_weighted_evaluate<RTYPE, XTYPE, CTYPE, KTYPE,
								  false /* knot_derivs */,
								  false /*is_basis_u_constant */,
								  1 /* basis_step */>
			   (linearWeights, result, x, knots, knot_count);


			wR[lane] = result;
		}
		break;
	}

	case 5:  // constant
	{
		ConstWideAccessor<XTYPE> wX(wx_);
		WideAccessor<RTYPE> wR(wout_);
		OSL_INTEL_PRAGMA("nofusion")
		OSL_INTEL_PRAGMA("omp simd simdlen(vec_width)")
		for(int lane=0; lane < wR.width; ++lane) {
			XTYPE x = wX[lane];

			// NOTE:  when basis is constant the weights are ignored,
			// just pass in 0's for the compiler to ignore
			fast::StaticMatrix44< 0, 0, 0, 0,
									0, 0, 0, 0,
									0, 0, 0, 0,
									0, 0, 0, 0, 1 /*Divisor*/> constantWeights;


			RTYPE result;
			spline_weighted_evaluate<RTYPE, XTYPE, CTYPE, KTYPE,
								  false /* knot_derivs */,
								  true /*is_basis_u_constant */,
								  1 /* basis_step */>
			   (constantWeights, result, x, knots, knot_count);

			wR[lane] = result;
		}
		break;
	}

	default:
		ASSERT(0 && "unsupported spline basis");
		break;
	};
}



} // namespace fast

OSL_SHADEOP void  osl_spline_w16fw16ff(void *wout_, const char *spline_, void *wx_,
                                 float *knots, int knot_count, int knot_arraylen)
{
	fast::template spline_evaluate<float, float, float, float, false>(wout_, USTR(spline_), wx_, knots, knot_count);
#if 0
		int basis_type = fast::getSplineBasisType(USTR(spline_));
		OSL_INTEL_PRAGMA("forceinline recursive")
		switch(basis_type)
		{

		case 0:  // catmull-rom
		{
			ConstWideAccessor<float> wX(wx_);
			WideAccessor<float> wR(wout_);
			// calling a function below, don't bother vectorizing
			OSL_INTEL_PRAGMA("nofusion")
			OSL_INTEL_PRAGMA("omp simd simdlen(wR.width)")
			for(int lane=0; lane < wR.width; ++lane) {
				float x = wX[lane];

								fast::StaticMatrix44<-1, 3, -3, 1,
													2, -5, 4, -1,
													-1, 0, 1, 0,
													0, 2, 0, 0,
													2 /* divisor */> catmullRomWeights;

				float result;
				fast::spline_evaluate<float, float, float, float,
									  false /* knot_derivs */,
				                      false /*is_basis_u_constant */,
									  1 /* basis_step */>
				   (catmullRomWeights, result, x, knots, knot_count, knot_arraylen);

				wR[lane] = result;
			}
			break;
		}



		case 1:  // bezier
		{
			ConstWideAccessor<float> wX(wx_);
			WideAccessor<float> wR(wout_);
			// calling a function below, don't bother vectorizing
			OSL_INTEL_PRAGMA("nofusion")
			OSL_INTEL_PRAGMA("omp simd simdlen(wR.width)")
			for(int lane=0; lane < wR.width; ++lane) {
				float x = wX[lane];


				fast::StaticMatrix44<-1, 3, -3, 1,
											3, -6, 3, 0,
											-3, 3, 0, 0,
											1, 0, 0, 0, 1 /*divisor*/> bezierWeights;
				float result;
				fast::spline_evaluate<float, float, float,float,
									  false /* knot_derivs */,
				                      false /*is_basis_u_constant */,
									  3 /* basis_step */>
				   (bezierWeights, result, x, knots, knot_count, knot_arraylen);


				wR[lane] = result;
			}
			break;

		}

		case 2:  // bspline
		{
			ConstWideAccessor<float> wX(wx_);
			WideAccessor<float> wR(wout_);
			// calling a function below, don't bother vectorizing
			OSL_INTEL_PRAGMA("nofusion")
			OSL_INTEL_PRAGMA("omp simd simdlen(wR.width)")
			for(int lane=0; lane < wR.width; ++lane) {
				float x = wX[lane];



				fast::StaticMatrix44<-1, 3, -3, 1,
											3, -6, 3, 0,
											-3, 0, 3, 0,
											1, 4, 1, 0, 6 /*bspline*/> bsplineWeights;
				float result;
				fast::spline_evaluate<float, float, float, float,
									  false /* knot_derivs */,
				                      false /*is_basis_u_constant */,
									  1 /* basis_step */>
				   (bsplineWeights, result, x, knots, knot_count, knot_arraylen);

				wR[lane] = result;
			}
			break;
		}

		case 3:  // hermite
		{
			ConstWideAccessor<float> wX(wx_);
			WideAccessor<float> wR(wout_);
			// calling a function below, don't bother vectorizing
			OSL_INTEL_PRAGMA("nofusion")
			OSL_INTEL_PRAGMA("omp simd simdlen(wR.width)")
			for(int lane=0; lane < wR.width; ++lane) {
				float x = wX[lane];


				fast::StaticMatrix44<2, 1, -2, 1,
											-3, -2, 3, -1,
											 0, 1, 0, 0,
											 1, 0, 0, 0, 1 /*Divisor*/> hermiteWeights;

				float result;
				fast::spline_evaluate<float, float, float, float,
									  false /* knot_derivs */,
				                      false /*is_basis_u_constant */,
									  2 /* basis_step */>
				   (hermiteWeights, result, x, knots, knot_count, knot_arraylen);

				wR[lane] = result;
			}
			break;
		}

		case 4:  // linear
		{
			ConstWideAccessor<float> wX(wx_);
			WideAccessor<float> wR(wout_);
			// calling a function below, don't bother vectorizing
			OSL_INTEL_PRAGMA("nofusion")
			OSL_INTEL_PRAGMA("omp simd simdlen(wR.width)")
			for(int lane=0; lane < wR.width; ++lane) {
				float x = wX[lane];



				fast::StaticMatrix44< 0, 0, 0, 0,
											0, 0, 0, 0,
											0, -1, 1, 0,
											0, 1, 0, 0, 1 /*Divisor*/> linearWeights;




				float result;

				fast::spline_evaluate<float, float, float, float,
									  false /* knot_derivs */,
				                      false /*is_basis_u_constant */,
									  1 /* basis_step */>
				   (linearWeights, result, x, knots, knot_count, knot_arraylen);


				wR[lane] = result;
			}
			break;
		}

		case 5:  // constant
		{
			ConstWideAccessor<float> wX(wx_);
			WideAccessor<float> wR(wout_);
			// calling a function below, don't bother vectorizing
			OSL_INTEL_PRAGMA("nofusion")
			OSL_INTEL_PRAGMA("omp simd simdlen(wR.width)")
			for(int lane=0; lane < wR.width; ++lane) {
				float x = wX[lane];

				// NOTE:  when basis is constant the weights are ignored,
				// just pass in 0's for the compiler to ignore
				fast::StaticMatrix44< 0, 0, 0, 0,
										0, 0, 0, 0,
										0, 0, 0, 0,
										0, 0, 0, 0, 1 /*Divisor*/> constantWeights;


				float result;
				fast::spline_evaluate<float, float, float, float,
									  false /* knot_derivs */,
				                      true /*is_basis_u_constant */,
									  1 /* basis_step */>
				   (constantWeights, result, x, knots, knot_count, knot_arraylen);

				wR[lane] = result;
			}
			break;
		}

		default:
			ASSERT(0 && "unsupported spline basis");
			break;
		};
#endif
}



OSL_SHADEOP void  osl_spline_dfdfdf(void *out, const char *spline_, void *x,
                                    float *knots, int knot_count, int knot_arraylen)
{

   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Dual2<float>, Dual2<float>, Dual2<float>, float, true>
      (spline, DFLOAT(out), DFLOAT(x), knots, knot_count, knot_arraylen);
}


OSL_SHADEOP void osl_spline_w16dfw16dfdf(void *wout, const char *spline_, void *wx_,
									float *knots, int knot_count, int knot_arraylen)
{

	/*template <class RTYPE, class XTYPE, class CTYPE, class KTYPE, bool knot_derivs>
		void spline_evaluate(
			void *wout_,
			ustring spline_basis,
			void *wx_,
			float *knots, //Here knots are vec3
			int knot_count)
		*/
//	fast::template spline_evaluate<Dual2<float>, Dual2<float>, Dual2<float>, float, true>(wout_, USTR(spline_), wx_, knots, knot_count);
}

OSL_SHADEOP void  osl_spline_dffdf(void *out, const char *spline_, void *x,
                                   float *knots, int knot_count, int knot_arraylen)
{

   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Dual2<float>, float, Dual2<float>, float, true>
      (spline, DFLOAT(out), *(float *)x, knots, knot_count, knot_arraylen);
}

OSL_SHADEOP void  osl_spline_dfdff(void *out, const char *spline_, void *x,
                                   float *knots, int knot_count, int knot_arraylen)
{

   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Dual2<float>, Dual2<float>, float, float, false>
      (spline, DFLOAT(out), DFLOAT(x), knots, knot_count, knot_arraylen);
}
OSL_SHADEOP void  osl_spline_w16dfw16dff(void *wout_, const char *spline_, void *wx_,
                                   float *knots, int knot_count, int knot_arraylen)
{
	fast::template spline_evaluate<Dual2<float>, Dual2<float>, float, float, false>(wout_, USTR(spline_), wx_, knots, knot_count);

#if 0
   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Dual2<float>, Dual2<float>, float, float, false>
      (spline, DFLOAT(out), DFLOAT(x), knots, knot_count, knot_arraylen);
#endif
}

OSL_SHADEOP void  osl_spline_w16fw16fw16f(void *wout_, const char *spline_, void *wx_,
										  void *wknots_, int knot_count, int knot_arraylen)
{
	const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
	ConstWideAccessor<float> wX(wx_);

	WideAccessor<float> wR(wout_);

	ConstWideUnboundArrayAccessor<float> wKnots(wknots_, knot_count);

	// calling a function below, don't bother vectorizing
	//OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
	OSL_INTEL_PRAGMA(novector)
	for(int lane=0; lane < wR.width; ++lane) {
	    float x = wX[lane];
	    auto knotsArrayProxy = wKnots[lane];

	    // Existing spline evaluate is expecting a float array
	    // in development version will be templatized.
	    // For now we will extract the knots out to a linear array
	    float knots[knot_count];
	    for(int k=0; k< knot_count; ++k) {
	    	knots[k] = knotsArrayProxy[k];
	    }

	    // TODO: investigate removing this function call to enable SIMD
	    float result;
	    Spline::spline_evaluate<float, float, float, float, false>
	       (spline, result, x, knots, knot_count, knot_arraylen);

	    wR[lane] = result;
	}
}

OSL_SHADEOP void  osl_spline_vfv(void *out, const char *spline_, void *x,
                                 Vec3 *knots, int knot_count, int knot_arraylen)
{

   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Vec3, float, Vec3, Vec3, false>
      (spline, *(Vec3 *)out, *(float *)x, knots, knot_count, knot_arraylen);
}

OSL_SHADEOP void  osl_spline_w16vw16fv(void *wout_, const char *spline_, void *wx_,
                                 Vec3 *knots, int knot_count, int knot_arraylen)
{
    //Knots were floats in the other cases; here, Vec3.
	fast::template spline_evaluate<Vec3, float, Vec3, Vec3, false>(wout_, USTR(spline_), wx_, knots, knot_count);
/*template <class RTYPE, class XTYPE, class CTYPE, class KTYPE, bool knot_derivs>
	void spline_evaluate(
		void *wout_,
		ustring spline_basis,
		void *wx_,
		float *knots, //Here knots are vec3
		int knot_count)
	*/
#if 0
   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Vec3, float, Vec3, Vec3, false>
      (spline, *(Vec3 *)out, *(float *)x, knots, knot_count, knot_arraylen);
#endif
}

OSL_SHADEOP void  osl_spline_dvdfv(void *out, const char *spline_, void *x,
                                   Vec3 *knots, int knot_count, int knot_arraylen)
{


   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Dual2<Vec3>, Dual2<float>, Vec3, Vec3, false>
      (spline, DVEC(out), DFLOAT(x), knots, knot_count, knot_arraylen);
}

OSL_SHADEOP void osl_spline_w16dvw16dfv (void *wout_, const char *spline_, void *wx_,
        Vec3 *knots, int knot_count, int knot_arraylen)
{

	//fast::template spline_evaluate<Dual2<Vec3>, Dual2<float>, Vec3, Vec3, false>(wout_, USTR(spline_), wx_, knots, knot_count);

#if 0
const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
Spline::spline_evaluate<Dual2<Vec3>, Dual2<float>, Vec3, Vec3, false>
(spline, DVEC(out), DFLOAT(x), knots, knot_count, knot_arraylen);
#endif
}

OSL_SHADEOP void  osl_spline_dvfdv(void *out, const char *spline_, void *x,
                                    Vec3 *knots, int knot_count, int knot_arraylen)
{

   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Dual2<Vec3>, float, Dual2<Vec3>, Vec3, true>
      (spline, DVEC(out), *(float *)x, knots, knot_count, knot_arraylen);
}

OSL_SHADEOP void  osl_spline_dvdfdv(void *out, const char *spline_, void *x,
                                    Vec3 *knots, int knot_count, int knot_arraylen)
{

   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Dual2<Vec3>, Dual2<float>, Dual2<Vec3>, Vec3, true>
      (spline, DVEC(out), DFLOAT(x), knots, knot_count, knot_arraylen);
}

OSL_SHADEOP void  osl_spline_w16dvw16dfdv(void *wout_, const char *spline_, void *wx_,
                                    Vec3 *knots, int knot_count, int knot_arraylen)
{

	//fast::template spline_evaluate<Dual2<Vec3>, Dual2<float>, Dual2<Vec3>, Vec3, true>(wout_, USTR(spline_), wx_, knots, knot_count);
/*template <class RTYPE, class XTYPE, class CTYPE, class KTYPE, bool knot_derivs>
	void spline_evaluate(
		void *wout_,
		ustring spline_basis,
		void *wx_,
		float *knots, //Here knots are vec3
		int knot_count)
	*/
  /* const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Dual2<Vec3>, Dual2<float>, Dual2<Vec3>, Vec3, true>
      (spline, DVEC(out), DFLOAT(x), knots, knot_count, knot_arraylen);
*/
}



OSL_SHADEOP void osl_splineinverse_fff(void *out, const char *spline_, void *x,
                                       float *knots, int knot_count, int knot_arraylen)
{
    // Version with no derivs

    const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
    Spline::spline_inverse<float> (spline, *(float *)out, *(float *)x, knots, knot_count, knot_arraylen);
}

OSL_SHADEOP void osl_splineinverse_dfdff(void *out, const char *spline_, void *x,
                                         float *knots, int knot_count, int knot_arraylen)
{
    // x has derivs, so return derivs as well

    const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
    Spline::spline_inverse<Dual2<float> > (spline, DFLOAT(out), DFLOAT(x), knots, knot_count, knot_arraylen);
}

OSL_SHADEOP void osl_splineinverse_dfdfdf(void *out, const char *spline_, void *x,
                                          float *knots, int knot_count, int knot_arraylen)
{

    // Ignore knot derivatives
    osl_splineinverse_dfdff (out, spline_, x, knots, knot_count, knot_arraylen);
}

OSL_SHADEOP void osl_splineinverse_dffdf(void *out, const char *spline_, void *x,
                                         float *knots, int knot_count, int knot_arraylen)
{

    // Ignore knot derivs
    float outtmp = 0;
    osl_splineinverse_fff (&outtmp, spline_, x, knots, knot_count, knot_arraylen);
    DFLOAT(out) = outtmp;
}



} // namespace pvt
OSL_NAMESPACE_EXIT
