// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

OSL_NAMESPACE_ENTER

namespace pvt {

// declaring a Spline namespace to avoid polluting the existing
// namespaces with all these templated helper functions.
namespace Spline {

struct SplineBasis {
   int      basis_step;
   float    basis[4][4];
};

// ========================================================
//
// Interpolation bases for splines
//
// The order here is very important for the SplineInterp::create
// constructor below. Any additional modes should be added to
// the end, or SplineInterp::create updated as well.
//
// ========================================================

enum {
    kCatmullRom,
    kBezier,
    kBSpline,
    kHermite,
    kLinear,
    kConstant,
    kNumSplineTypes
};

OSL_CONSTANT_DATA const static SplineBasis gBasisSet[kNumSplineTypes] = {
//
// catmullrom
//
   { 1, { {(-1.0f/2.0f),  ( 3.0f/2.0f), (-3.0f/2.0f), ( 1.0f/2.0f)},
          {( 2.0f/2.0f),  (-5.0f/2.0f), ( 4.0f/2.0f), (-1.0f/2.0f)},
          {(-1.0f/2.0f),  ( 0.0f/2.0f), ( 1.0f/2.0f), ( 0.0f/2.0f)},
          {( 0.0f/2.0f),  ( 2.0f/2.0f), ( 0.0f/2.0f), ( 0.0f/2.0f)}  } },
//
// bezier
//
   { 3, { {-1,  3, -3,  1},
          { 3, -6,  3,  0},
          {-3,  3,  0,  0},
          { 1,  0,  0,  0} } },
//
// bspline
//
   { 1, { {(-1.0f/6.0f), ( 3.0f/6.0f),  (-3.0f/6.0f),  (1.0f/6.0f)},
          {( 3.0f/6.0f), (-6.0f/6.0f),  ( 3.0f/6.0f),  (0.0f/6.0f)},
          {(-3.0f/6.0f), ( 0.0f/6.0f),  ( 3.0f/6.0f),  (0.0f/6.0f)},
          {( 1.0f/6.0f), ( 4.0f/6.0f),  ( 1.0f/6.0f),  (0.0f/6.0f)} } },
//
// hermite
//
   { 2, { { 2,  1, -2,  1},
          {-3, -2,  3, -1},
          { 0,  1,  0,  0},
          { 1,  0,  0,  0} } },
//
// linear
//
   { 1, { {0,  0,  0,  0},
          {0,  0,  0,  0},
          {0, -1,  1,  0},
          {0,  1,  0,  0} } },
//
// constant
//
   { 1, { {0,  0,  0,  0},
          {0,  0,  0,  0},
          {0,  0,  0,  0},
          {0,  0,  0,  0} } }
};

struct SplineInterp {
    const SplineBasis& spline;
    const bool         constant;

    OSL_HOSTDEVICE static SplineInterp create(StringParam basis_name)
    {
        if (basis_name == StringParams::catmullrom)
            return { gBasisSet[kCatmullRom], false };
        if (basis_name == StringParams::bezier)
            return { gBasisSet[kBezier], false };
        if (basis_name == StringParams::bspline)
            return { gBasisSet[kBSpline], false };
        if (basis_name == StringParams::hermite)
            return { gBasisSet[kHermite], false };
        if (basis_name == StringParams::constant)
            return { gBasisSet[kConstant], true };

        // Default to linear
        return { gBasisSet[kLinear], false };
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
    template <class OUTTYPE, class INTYPE, bool is_dual>
    struct extractValueFromArray
    {
        OSL_HOSTDEVICE OUTTYPE operator()(const INTYPE *value, int array_length, int idx);
    };

    template <class OUTTYPE, class INTYPE>
    struct extractValueFromArray<OUTTYPE, INTYPE, true> 
    {
        OSL_HOSTDEVICE OUTTYPE operator()(const INTYPE *value, int array_length, int idx)
        {
            return OUTTYPE( value[idx + 0*array_length], 
                            value[idx + 1*array_length],
                            value[idx + 2*array_length] );
        }
    };

    template <class OUTTYPE, class INTYPE>
    struct extractValueFromArray<OUTTYPE, INTYPE, false> 
    {
        OSL_HOSTDEVICE OUTTYPE operator()(const INTYPE *value, int /*array_length*/, int idx)
        {
            return OUTTYPE( value[idx] );
        }
    };

    // Spline functor for use with the inverse function
    template <class RTYPE, class XTYPE>
    struct SplineFunctor {
        OSL_HOSTDEVICE SplineFunctor (const SplineInterp& spline_, const float *knots_,
                                      int knot_count_, int knot_arraylen_)
            : spline(spline_), knots(knots_), knot_count(knot_count_),
              knot_arraylen(knot_arraylen_) { }

        OSL_HOSTDEVICE RTYPE operator() (XTYPE x) {
            RTYPE v;
            spline.evaluate<RTYPE,XTYPE,float,float,false> (v, x, knots, knot_count, knot_arraylen);
            return v;
        }
    private:
        const SplineInterp& spline;
        const float *knots;
        int knot_count, knot_arraylen;
    };

    template <class RTYPE, class XTYPE, class CTYPE, class KTYPE, bool knot_derivs >
    OSL_HOSTDEVICE void
    evaluate(RTYPE &result, XTYPE &xval, const KTYPE *knots,
             int knot_count, int knot_arraylen) const
    {
        using OIIO::clamp;
        XTYPE x = clamp(xval, XTYPE(0.0), XTYPE(1.0));
        int nsegs = ((knot_count - 4) / spline.basis_step) + 1;
        x = x*(float)nsegs;
        float seg_x = removeDerivatives(x);
        int segnum = (int)seg_x;
        if (segnum < 0)
            segnum = 0;
        if (segnum > (nsegs-1))
           segnum = nsegs-1;

        if (constant) {
            // Special case for "constant" basis
            RTYPE P = removeDerivatives (knots[segnum+1]);
            assignment (result, P);
            return;
        }

        // x is the position along segment 'segnum'
        x = x - float(segnum);
        int s = segnum * spline.basis_step;

        // create a functor so we can cleanly(!) extract
        // the knot elements
        extractValueFromArray<CTYPE, KTYPE, knot_derivs> myExtract;
        CTYPE P[4];
        for (int k = 0; k < 4; k++) {
            P[k] = myExtract(knots, knot_arraylen, s + k);
        }

        CTYPE tk[4];
        for (int k = 0; k < 4; k++) {
            tk[k] = spline.basis[k][0] * P[0] +
                    spline.basis[k][1] * P[1] +
                    spline.basis[k][2] * P[2] + 
                    spline.basis[k][3] * P[3];
        }

        RTYPE tresult;
        // The following is what we want, but this gives me template errors
        // which I'm too lazy to decipher:
        //    tresult = ((tk[0]*x + tk[1])*x + tk[2])*x + tk[3];
        tresult = (tk[0]   * x + tk[1]);
        tresult = (tresult * x + tk[2]);
        tresult = (tresult * x + tk[3]);
        assignment(result, tresult);
    }

    // Evaluate the inverse of a spline, i.e., solve for the x for which
    // spline_evaluate(x) == y.
    template <class YTYPE>
    OSL_HOSTDEVICE void
    inverse (YTYPE &x, YTYPE y, const float *knots,
             int knot_count, int knot_arraylen) const
    {
        // account for out-of-range inputs, just clamp to the values we have
        int lowindex = spline.basis_step == 1 ? 1 : 0;
        int highindex = spline.basis_step == 1 ? knot_count-2 : knot_count-1;
        bool increasing = knots[1] < knots[knot_count-2];
        if (increasing) {
            if (y <= knots[lowindex]) {
                x = YTYPE(0);
                return;
            }
            if (y >= knots[highindex]) {
                x = YTYPE(1);
                return;
            }
        } else {
            if (y >= knots[lowindex]) {
                x = YTYPE(0);
                return;
            }
            if (y <= knots[highindex]) {
                x = YTYPE(1);
                return;
            }
        }


        SplineFunctor<YTYPE,YTYPE> S (*this, knots, knot_count, knot_arraylen);
        // Because of the nature of spline interpolation, monotonic knots
        // can still lead to a non-monotonic curve.  To deal with this,
        // search separately on each spline segment and hope for the best.
        int nsegs = (knot_count - 4) / spline.basis_step + 1;
        float nseginv = 1.0f / nsegs;
        YTYPE r0 = 0.0;
        x = 0;
        for (int s = 0;  s < nsegs;  ++s) {  // Search each interval
            YTYPE r1 = nseginv * (s+1);
            bool brack;
            x = OIIO::invert (S, y, r0, r1, 32, YTYPE(1.0e-6), &brack);
            if (brack)
                return;
            r0 = r1;  // Start of next interval is end of this one
        }
    }
};


}; // namespace Spline
}; // namespace pvt
OSL_NAMESPACE_EXIT

