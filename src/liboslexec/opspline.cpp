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

#include "oslexec_pvt.h"
#include "oslops.h"
#include "dual_vec.h"

#include "OpenImageIO/varyingref.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {

namespace {

// ========================================================
//
// Interpolation bases for splines
//
// ========================================================

struct SplineBasis {
   ustring  basis_name;
   int      basis_step;
   Matrix44 basis;
};

static const int kNumSplineTypes = 5;
static const int kLinearSpline = kNumSplineTypes - 1;
static SplineBasis gBasisSet[kNumSplineTypes] = {
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
   { ustring("hermite"),     2, Matrix44(  1,   1, -3,  1,
                                          -1,  -2,  4, -1,
                                          -1,   1,  0,  0,
                                           1,   0,  0,  0) },
   { ustring("linear"),      1, Matrix44(  0,  0,  0,  0,
                                           0,  0,  0,  0,
                                           0, -1,  1,  0,
                                           0,  1,  0,  0) }
};



// ========================================================
//
// Silly helper functions for handling operations involving
// various types
//
// ========================================================

inline Dual2<float> Clamp(Dual2<float> x, Dual2<float> minv, Dual2<float> maxv)
{
    return dual_clamp(x, minv, maxv);
}

inline float Clamp(float x, float minv, float maxv) {
    if (x < minv) return minv;
    else if (x > maxv) return maxv;
    else return x;
};



// If necessary, eliminate the derivatives of a number
template<class T> T removeDerivatives (const T x)              { return x;       }
template<class T> T removeDerivatives (const Dual2<T> &x) { return x.val(); }



// simple templated "copy" function
template <class T>
inline void assignment(T &a, T &b)
{
   a = b;
}

template <class T>
inline void assignment(T &a, Dual2<T> &b)
{
   a = b.val();
}



// We need to know explicitly whether the knots have
// derivatives associated with them because of the way
// Dual2<T> forms of arrays are stored..  Arrays with 
// derivatives are stored:
//   T T T... T.dx T.dx T.dx... T.dy T.dy T.dy...
// This means, we need to explicitly construct the Dual2<T>
// form of the knots on the fly.
// if 'is_dual' == true, then OUTTYPE == Dual2<INTYPE>
// if 'is_dual' == false, then OUTTYPE == INTYPE

// This functor will extract a T or a Dual2<T> type from a VaryingRef array
template <class OUTTYPE, class INTYPE, bool is_dual> struct extractValueFromArray
{
    OUTTYPE operator()(VaryingRef<INTYPE> &value, int array_length, int i, int idx);
};

template <class OUTTYPE, class INTYPE>
struct extractValueFromArray<OUTTYPE, INTYPE, true> 
{
    OUTTYPE operator()(VaryingRef<INTYPE> &value, int array_length, int i, int idx)
    {
        return OUTTYPE( (&value[i])[idx + 0*array_length], 
                        (&value[i])[idx + 1*array_length],
                        (&value[i])[idx + 2*array_length] );
    }
};

template <class OUTTYPE, class INTYPE>
struct extractValueFromArray<OUTTYPE, INTYPE, false> 
{
    OUTTYPE operator()(VaryingRef<INTYPE> &value, int array_length, int i, int idx)
    {
        return OUTTYPE( (&value[i])[idx] );
    }
};



// Functors for copying the final interpolated point to the
// spline-op Result field.  This might involve eliminating
// derivatives.

struct DualFloatToFloat
{
   float operator()(Dual2<float> a) { return a.val(); }
};

struct DualVec3ToVec3
{
   Vec3 operator()(Dual2<Vec3> a) { return a.val(); }
};

template<class T>
struct CopySelf
{
   T operator()(T a) { return a; }
};



// This is the special-case version of the spline interpolation for
// when there are derivatives.
//
// This shade-op is tricky because of the way arrays with derivatives
// are stored.  They are not stored as an array of Dual2<*> elements, 
// instead they are stored as three consecutive arrays: base-elements,
// dx-elements, dy-elements. This means we need to construct Dual2<> forms
// of the knots on-the-fly so that we can perform Dual2<> math.
//
// ATYPE -- the result type
// BTYPE -- the interpolator type
// CTYPE -- the type of the knots:  float, Dual2<float> Vec3, Dual2<Vec3>.
// DTYPE -- the base type of the knoes, either 'Vec3' or 'float'
// knot_derivs -- bool determining whether or not the knots have
//                derivatives associated with them -- this is used when
//                extracting knots from the knot array
// KNOT_TO_RESULT -- functor which casts from CTYPE to ATYPE (in case we 
//                   need to "downgrade" from Dual2<float> to float, for 
//                   example)
//
template <class ATYPE, class BTYPE, class CTYPE, class DTYPE, bool knot_derivs, class KNOT_TO_RESULT>
inline void 
spline_op_guts_generic(Symbol &Result, Symbol Spline, int array_length, Symbol &Value, int num_knots, Symbol &Knots,
      ShadingExecution *exec, bool zero_derivs=true)
{
    VaryingRef<ATYPE> result ((ATYPE *)Result.data(), Result.step());
    VaryingRef<BTYPE> xval   ((BTYPE *)Value.data(),  Value.step());
    VaryingRef<DTYPE> knots  ((DTYPE *)Knots.data(),  Knots.step());
    VaryingRef<ustring> spline_name ((ustring *)Spline.data(), Spline.step());

    bool varying_spline = Spline.is_varying();
    int nsegs = -1;
    int basis_type = -1;
    if (!varying_spline)
    {
        for (basis_type = 0; basis_type < kNumSplineTypes && 
            spline_name[0] != gBasisSet[basis_type].basis_name; basis_type++);
        // If unrecognizable spline type, then default to Linear
        if (basis_type == kNumSplineTypes)
            basis_type = kLinearSpline;
        int knot_count = (num_knots > 0) ? std::min(num_knots, Knots.typespec().arraylength()) : Knots.typespec().arraylength();
        nsegs = ((knot_count - 4) / gBasisSet[basis_type].basis_step) + 1;
    }

    // assuming spline-type is uniform
    KNOT_TO_RESULT castToResult;

    SHADE_LOOP_BEGIN
        if (varying_spline)
        {
            for (basis_type = 0; basis_type < kNumSplineTypes && 
                spline_name[i] != gBasisSet[basis_type].basis_name; basis_type++);
            // If unrecognizable spline type, then default to Linear
            if (basis_type == kNumSplineTypes)
                basis_type = kLinearSpline;
            int knot_count = (num_knots > 0) ? std::min(num_knots, Knots.typespec().arraylength()) : Knots.typespec().arraylength();
            nsegs = ((knot_count - 4) / gBasisSet[basis_type].basis_step) + 1;
        }
        SplineBasis &spline = gBasisSet[basis_type];

        BTYPE x = Clamp(xval[i], BTYPE(0.0f), BTYPE(1.0f));
        x = x*(float)nsegs;
        float seg_x = removeDerivatives(x);
        int segnum = (int)seg_x;
        if (segnum > (nsegs-1))
           segnum = nsegs-1;
        // x is the position along segment 'segnum'
        x = x - float(segnum);
        int s = segnum*spline.basis_step;
        int len = array_length;

        // create a functor so we can cleanly(!) extract
        // the knot elements
        extractValueFromArray<CTYPE, DTYPE, knot_derivs> myExtract;
        CTYPE P[4];
        for (int k = 0; k < 4; k++) {
            P[k] = myExtract(knots, len, i, s + k);
        }

        CTYPE tk[4];
        for (int k = 0; k < 4; k++) {
            tk[k] = spline.basis[k][0] * P[0] +
                    spline.basis[k][1] * P[1] +
                    spline.basis[k][2] * P[2] + 
                    spline.basis[k][3] * P[3];
        }

        ATYPE tresult;
        // The following is what we want, but this gives me template errors
        // which I'm too lazy to decipher:
        //    tresult = ((tk[0]*x + tk[1])*x + tk[2])*x + tk[3];
        tresult = castToResult (tk[0]   * x + tk[1]);
        tresult = castToResult (tresult * x + tk[2]);
        tresult = castToResult (tresult * x + tk[3]);
        assignment(result[i], tresult);
    SHADE_LOOP_END
}



};  // End anonymous namespace



#define RES_DERIVS  (1 << 0)
#define VALU_DERIVS (1 << 1)
#define KNOT_DERIVS (1 << 2)
#define TRIPLES     (1 << 3)

DECLOP (OP_spline)
{
    bool knot_num_specified = (nargs == 5) ? true : false;
    Symbol &Result (exec->sym (args[0]));
    Symbol &Spline (exec->sym (args[1]));
    Symbol &Value  (exec->sym (args[2]));
    Symbol &Knots  (exec->sym (args[knot_num_specified ? 4 : 3]));

    int num_knots = -1;

    ASSERT (! Result.typespec().is_closure()  &&
            ! Spline.typespec().is_closure()  &&
            ! Value.typespec().is_closure()   &&
            ! Knots.typespec().is_closure());

    ASSERT(Knots.typespec().is_array());

    if (knot_num_specified) {
       Symbol &Knot_num = (exec->sym (args[3]));
       ASSERT(! Knot_num.typespec().is_closure());
       VaryingRef<int> knot_count ((int *)Knot_num.data(), Knot_num.step());
       num_knots = knot_count[0];
    }
    // determine varying-ness
    bool varying = Value.is_varying() || Knots.is_varying() || Spline.is_varying();
    exec->adjust_varying(Result, varying, Value.data() == Result.data());


    // FIXME: only support uniform spline basis
 
    const int array_length = Knots.typespec().arraylength();


    // We'll use a switch(){...} statement instead of nested if...else..
    // statements -- it makes things much cleaner looking.
    unsigned int mode = (Result.has_derivs() ? RES_DERIVS  : 0) |
                        (Value.has_derivs()  ? VALU_DERIVS : 0) |
#ifdef SKIP_KNOT_DERIVS
                        0                                       |
#else
                        (Knots.has_derivs()  ? KNOT_DERIVS : 0) |
#endif
                        (Result.typespec().is_triple() ? TRIPLES : 0);

    switch (mode)
    {
        case (RES_DERIVS | VALU_DERIVS | KNOT_DERIVS | TRIPLES):
            spline_op_guts_generic< Dual2<Vec3>, Dual2<float>, Dual2<Vec3>, Vec3, true, CopySelf<Dual2<Vec3> > >(Result, Spline, array_length, Value, num_knots, Knots,
                exec, false /*zero derivs?*/);
            break;
        case (RES_DERIVS | VALU_DERIVS | KNOT_DERIVS):
            spline_op_guts_generic< Dual2<float>, Dual2<float>, Dual2<float>, float, true, CopySelf<Dual2<float> >  >(Result, Spline, array_length, Value, num_knots, Knots,
                exec, false /*zero derivs?*/);
            break;
        //
        case (RES_DERIVS | VALU_DERIVS | TRIPLES):
            spline_op_guts_generic< Dual2<Vec3>, Dual2<float>, Vec3, Vec3, false, CopySelf<Dual2<Vec3> > >(Result, Spline, array_length, Value, num_knots, Knots,
                exec, false /*zero derivs?*/);
            break;
        case (RES_DERIVS | VALU_DERIVS):
            spline_op_guts_generic< Dual2<float>, Dual2<float>, float, float, false, CopySelf<Dual2<float> > >(Result, Spline, array_length, Value, num_knots, Knots,
                exec, false /*zero derivs?*/);
            break;
        //
        case (RES_DERIVS | TRIPLES):
            spline_op_guts_generic< Dual2<Vec3>, float, Vec3, Vec3, false, CopySelf<Dual2<Vec3> > >(Result, Spline, array_length, Value, num_knots, Knots,
                exec, false /*zero derivs?*/);
            break;
        case (RES_DERIVS):
            spline_op_guts_generic< Dual2<float>, float, float, float, false, CopySelf<Dual2<float> > >(Result, Spline, array_length, Value, num_knots, Knots,
                exec, false /*zero derivs?*/);
            break;

        // The result has no derivatives so ignore the cases where the values
        // and/or knots have derivatives.  This simplifies to case of whether
        // we're dealing with floats or triples.
        default:
            if (mode & TRIPLES)
                spline_op_guts_generic< Vec3, float, Vec3, Vec3, false, CopySelf<Vec3> >(Result, Spline, array_length, Value, num_knots, Knots,
                    exec, false /*zero derivs?*/);
            else
                spline_op_guts_generic< float, float, float, float, false, CopySelf<float> >(Result, Spline, array_length, Value, num_knots, Knots,
                    exec, false /*zero derivs?*/);
            break;
    }
}

}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
