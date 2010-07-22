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
#include "splineimpl.h"

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
   { ustring("hermite"),     2, Matrix44(  1,   1, -3,  1,
                                          -1,  -2,  4, -1,
                                          -1,   1,  0,  0,
                                           1,   0,  0,  0) },
   { ustring("linear"),      1, Matrix44(  0,  0,  0,  0,
                                           0,  0,  0,  0,
                                           0, -1,  1,  0,
                                           0,  1,  0,  0) }
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
//
template <class ATYPE, class BTYPE, class CTYPE, class DTYPE, bool knot_derivs >
inline void 
spline_op_guts_generic(Symbol &Result, Symbol Spline, int array_length, Symbol &Value, int num_knots, Symbol &Knots,
      ShadingExecution *exec, bool zero_derivs=true)
{
    VaryingRef<ATYPE> result ((ATYPE *)Result.data(), Result.step());
    VaryingRef<BTYPE> xval   ((BTYPE *)Value.data(),  Value.step());
    VaryingRef<DTYPE> knots  ((DTYPE *)Knots.data(),  Knots.step());
    VaryingRef<ustring> spline_name ((ustring *)Spline.data(), Spline.step());

    bool varying_spline = Spline.is_varying();
    const Spline::SplineBasis *spline_basis = NULL;
    int knot_count = (num_knots > 0) ? 
                     std::min(num_knots, Knots.typespec().arraylength()) : 
                     Knots.typespec().arraylength();

    if (!varying_spline)
        spline_basis = Spline::getSplineBasis(spline_name[0]);

    SHADE_LOOP_BEGIN
        if (varying_spline)
            spline_basis = Spline::getSplineBasis(spline_name[i]);

        Spline::spline_evaluate<ATYPE, BTYPE, CTYPE, DTYPE, knot_derivs>(spline_basis, result[i], xval[i], &knots[i], knot_count);
    SHADE_LOOP_END
}



};  // End anonymous namespace

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
            spline_op_guts_generic< Dual2<Vec3>, Dual2<float>, Dual2<Vec3>, Vec3, true >(Result, Spline, array_length, Value, num_knots, Knots,
                exec, false /*zero derivs?*/);
            break;
        case (RES_DERIVS | VALU_DERIVS | KNOT_DERIVS):
            spline_op_guts_generic< Dual2<float>, Dual2<float>, Dual2<float>, float, true >(Result, Spline, array_length, Value, num_knots, Knots,
                exec, false /*zero derivs?*/);
            break;
        //
        case (RES_DERIVS | VALU_DERIVS | TRIPLES):
            spline_op_guts_generic< Dual2<Vec3>, Dual2<float>, Vec3, Vec3, false >(Result, Spline, array_length, Value, num_knots, Knots,
                exec, false /*zero derivs?*/);
            break;
        case (RES_DERIVS | VALU_DERIVS):
            spline_op_guts_generic< Dual2<float>, Dual2<float>, float, float, false >(Result, Spline, array_length, Value, num_knots, Knots,
                exec, false /*zero derivs?*/);
            break;
        //
        case (RES_DERIVS | TRIPLES):
            spline_op_guts_generic< Dual2<Vec3>, float, Vec3, Vec3, false >(Result, Spline, array_length, Value, num_knots, Knots,
                exec, false /*zero derivs?*/);
            break;
        case (RES_DERIVS):
            spline_op_guts_generic< Dual2<float>, float, float, float, false >(Result, Spline, array_length, Value, num_knots, Knots,
                exec, false /*zero derivs?*/);
            break;

        // The result has no derivatives so ignore the cases where the values
        // and/or knots have derivatives.  This simplifies to case of whether
        // we're dealing with floats or triples.
        default:
            if (mode & TRIPLES)
                spline_op_guts_generic< Vec3, float, Vec3, Vec3, false >(Result, Spline, array_length, Value, num_knots, Knots,
                    exec, false /*zero derivs?*/);
            else
                spline_op_guts_generic< float, float, float, float, false >(Result, Spline, array_length, Value, num_knots, Knots,
                    exec, false /*zero derivs?*/);
            break;
    }
}

}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
