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

#pragma once

// avoid naming conflict with MSVC macro
#ifdef BTYPE
#undef BTYPE
#endif

using namespace std;
OSL_NAMESPACE_ENTER

namespace pvt {

// declaring a Spline namespace to avoid polluting the existing
// namespaces with all these templated helper functions.
namespace Spline {

static ustring u_constant("constant");


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
    OUTTYPE operator()(const INTYPE *value, int array_length, int idx);
};

template <class OUTTYPE, class INTYPE>
struct extractValueFromArray<OUTTYPE, INTYPE, true> 
{
    OUTTYPE operator()(const INTYPE *value, int array_length, int idx)
    {
        return OUTTYPE( value[idx + 0*array_length], 
                        value[idx + 1*array_length],
                        value[idx + 2*array_length] );
    }
};

template <class OUTTYPE, class INTYPE>
struct extractValueFromArray<OUTTYPE, INTYPE, false> 
{
    OUTTYPE operator()(const INTYPE *value, int array_length, int idx)
    {
        return OUTTYPE( value[idx] );
    }
};

inline Dual2<float> Clamp(Dual2<float> x, Dual2<float> minv, Dual2<float> maxv)
{
    return dual_clamp(x, minv, maxv);
}

inline float Clamp(float x, float minv, float maxv) {
    if (x < minv) return minv;
    else if (x > maxv) return maxv;
    else return x;
};




struct SplineBasis {
   ustring  basis_name;
   int      basis_step;
   Matrix44 basis;
};

const SplineBasis *getSplineBasis(const ustring &basis_name);

template <class RTYPE, class XTYPE, class CTYPE, class KTYPE, bool knot_derivs >
void spline_evaluate(const SplineBasis *spline, 
                     RTYPE &result, 
                     XTYPE &xval, 
                     const KTYPE *knots,
                     int knot_count, int knot_arraylen)
{
    XTYPE x = Clamp(xval, XTYPE(0.0), XTYPE(1.0)); //SM: x should lie between 0 and 1
    int nsegs = ((knot_count - 4) / spline->basis_step) + 1; //SM: Why 4
    x = x*(float)nsegs; //SM: Depends on knot count and basis_step
    float seg_x = removeDerivatives(x);
    int segnum = (int)seg_x; //SM: x without derivatives
    if (segnum < 0)
        segnum = 0;
    if (segnum > (nsegs-1))
       segnum = nsegs-1;

    if (spline->basis_name == u_constant) {
        // Special case for "constant" basis
        RTYPE P = removeDerivatives (knots[segnum+1]); //SM: Where is this defined?
        assignment (result, P);
        return;
    }

    // x is the position along segment 'segnum'
    x = x - float(segnum);
    int s = segnum*spline->basis_step;

  //  printf("knot_count::%d\tnsegs::%d\ts::%d\tsegnum::%d\n", knot_count, nsegs, s, segnum);

    // create a functor so we can cleanly(!) extract
    // the knot elements
    extractValueFromArray<CTYPE, KTYPE, knot_derivs> myExtract;
    CTYPE P[4]; //SM: contains knot elements
    for (int k = 0; k < 4; k++) {
        P[k] = myExtract(knots, knot_arraylen, s + k);
       // printf("Calculating P[%d]\n",k );
    }

    CTYPE tk[4];
    for (int k = 0; k < 4; k++) {
        tk[k] = spline->basis[k][0] * P[0] +
                spline->basis[k][1] * P[1] +
                spline->basis[k][2] * P[2] + 
                spline->basis[k][3] * P[3];
        //std::cout<<"CTYPE tk[4]"<<std::endl;
     //  printf("Calculating tk[%d]\n",k );
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



// Spline functor for use with the inverse function
template <class RTYPE, class XTYPE>
struct SplineFunctor {
    SplineFunctor (const SplineBasis *spline, const float *knots,
                   int knot_count, int knot_arraylen)
        : spline(spline), knots(knots), knot_count(knot_count),
          knot_arraylen(knot_arraylen) { }

    RTYPE operator() (XTYPE x) {
        RTYPE v;
        spline_evaluate<RTYPE,XTYPE,float,float,false> (spline, v, x, knots,
                                                 knot_count, knot_arraylen);
        return v;
    }
private:
    const SplineBasis *spline;
    const float *knots;
    int knot_count, knot_arraylen;
};



// Evaluate the inverse of a spline, i.e., solve for the x for which
// spline_evaluate(x) == y.

template <class YTYPE>
void spline_inverse (const SplineBasis *spline,
                     YTYPE &x, YTYPE y, const float *knots, int knot_count,
                     int knot_arraylen)
{
    // account for out-of-range inputs, just clamp to the values we have
    bool increasing = knots[1] < knots[knot_count-2];
    if (increasing) {
        if (y < knots[1]) {
            x = YTYPE(0);
            return;
        }
        if (y > knots[knot_count-2]) {
            x = YTYPE(1);
            return;
        }
    } else {
        if (y > knots[1]) {
            x = YTYPE(0);
            return;
        }
        if (y < knots[knot_count-2]) {
            x = YTYPE(1);
            return;
        }
    }

    //SM: Eliminate the Functor, and move splineinverse to the fast space. And call
    //our fast::spline_evaluate.

    SplineFunctor<YTYPE,YTYPE> S (spline, knots, knot_count, knot_arraylen); //SM: Pass our spline_val
    // Because of the nature of spline interpolation, monotonic knots
    // can still lead to a non-monotonic curve.  To deal with this,
    // search separately on each spline segment and hope for the best.
    int nsegs = (knot_count - 4) / spline->basis_step + 1;
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
}//splineinverse() ends



}; // namespace Spline
}; // namespace pvt
OSL_NAMESPACE_EXIT

