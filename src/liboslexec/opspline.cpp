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
#include "dual_vec.h"
#include "splineimpl.h"


namespace {

// ========================================================
//
// Interpolation bases for splines
//
// ========================================================

static const int kNumSplineTypes = 5;
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
                                           0,  1,  0,  0) }
};

};  // End anonymous namespace


OSL_NAMESPACE_ENTER

namespace pvt {


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


}; // namespace pvt
OSL_NAMESPACE_EXIT



#define USTR(cstr) (*((ustring *)&cstr))
#define DFLOAT(x) (*(Dual2<Float> *)x)
#define DVEC(x) (*(Dual2<Vec3> *)x)

OSL_SHADEOP void  osl_spline_fff(void *out, const char *spline_, void *x, 
                                 float *knots, int knot_count)
{
   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<float, float, float, float, false>
      (spline, *(float *)out, *(float *)x, knots, knot_count);
}

OSL_SHADEOP void  osl_spline_dfdfdf(void *out, const char *spline_, void *x, 
                                    float *knots, int knot_count)
{
   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Dual2<float>, Dual2<float>, Dual2<float>, float, true>
      (spline, DFLOAT(out), DFLOAT(x), knots, knot_count);
}

OSL_SHADEOP void  osl_spline_dffdf(void *out, const char *spline_, void *x, 
                                   float *knots, int knot_count)
{
   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Dual2<float>, float, Dual2<float>, float, true>
      (spline, DFLOAT(out), *(float *)x, knots, knot_count);
}

OSL_SHADEOP void  osl_spline_dfdff(void *out, const char *spline_, void *x, 
                                   float *knots, int knot_count)
{
   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Dual2<float>, Dual2<float>, float, float, false>
      (spline, DFLOAT(out), DFLOAT(x), knots, knot_count);
}

OSL_SHADEOP void  osl_spline_vfv(void *out, const char *spline_, void *x, 
                                 Vec3 *knots, int knot_count)
{
   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Vec3, float, Vec3, Vec3, false>
      (spline, *(Vec3 *)out, *(float *)x, knots, knot_count);
}

OSL_SHADEOP void  osl_spline_dvdfv(void *out, const char *spline_, void *x, 
                                   Vec3 *knots, int knot_count)
{
   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Vec3, float, Vec3, Vec3, false>
      (spline, *(Vec3 *)out, *(float *)x, knots, knot_count);
}

OSL_SHADEOP void  osl_spline_dvfdv(void *out, const char *spline_, void *x, 
                                    Vec3 *knots, int knot_count)
{
   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Dual2<Vec3>, float, Dual2<Vec3>, Vec3, true>
      (spline, DVEC(out), *(float *)x, knots, knot_count);
}

OSL_SHADEOP void  osl_spline_dvdfdv(void *out, const char *spline_, void *x, 
                                    Vec3 *knots, int knot_count)
{
   const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
   Spline::spline_evaluate<Dual2<Vec3>, Dual2<float>, Dual2<Vec3>, Vec3, true>
      (spline, DVEC(out), DFLOAT(x), knots, knot_count);
}



OSL_SHADEOP void osl_splineinverse_fff(void *out, const char *spline_, void *x, 
                                       float *knots, int knot_count)
{
    // Version with no derivs
    const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
    Spline::spline_inverse<float> (spline, *(float *)out, *(float *)x, knots, knot_count);
}

OSL_SHADEOP void osl_splineinverse_dfdff(void *out, const char *spline_, void *x, 
                                         float *knots, int knot_count)
{
    // x has derivs, so return derivs as well
    const Spline::SplineBasis *spline = Spline::getSplineBasis(USTR(spline_));
    Spline::spline_inverse<Dual2<float> > (spline, DFLOAT(out), DFLOAT(x), knots, knot_count);
}

OSL_SHADEOP void osl_splineinverse_dfdfdf(void *out, const char *spline_, void *x, 
                                          float *knots, int knot_count)
{
    // Ignore knot derivatives
    osl_splineinverse_dfdff (out, spline_, x, knots, knot_count);
}

OSL_SHADEOP void osl_splineinverse_dffdf(void *out, const char *spline_, void *x, 
                                         float *knots, int knot_count)
{
    // Ignore knot derivs
    float outtmp = 0;
    osl_splineinverse_fff (&outtmp, spline_, x, knots, knot_count);
    DFLOAT(out) = outtmp;
}


