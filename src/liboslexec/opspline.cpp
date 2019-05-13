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


#include "oslexec_pvt.h"
#include <OSL/dual_vec.h>
#include <OSL/Imathx.h>
#include <OSL/device_string.h>

#include <OpenImageIO/fmath.h>
#include "splineimpl.h"

OSL_NAMESPACE_ENTER

namespace pvt {


OSL_SHADEOP OSL_HOSTDEVICE void osl_spline_fff(void *out, const char *spline_, void *x,
                                 void* knots, int knot_count, int knot_arraylen)
{
  Spline::SplineInterp::create(HDSTR(spline_)).evaluate<float, float, float, float, false>
      (*(float *)out, *(float *)x, (float *)knots, knot_count, knot_arraylen);
}

OSL_SHADEOP OSL_HOSTDEVICE void osl_spline_dfdfdf(void *out, const char *spline_, void *x,
                                    void* knots, int knot_count, int knot_arraylen)
{
  Spline::SplineInterp::create(HDSTR(spline_)).evaluate<Dual2<float>, Dual2<float>, Dual2<float>, float, true>
      (DFLOAT(out), DFLOAT(x), (float *) knots, knot_count, knot_arraylen);
}

OSL_SHADEOP OSL_HOSTDEVICE void osl_spline_dffdf(void *out, const char *spline_, void *x,
                                   void* knots, int knot_count, int knot_arraylen)
{
  Spline::SplineInterp::create(HDSTR(spline_)).evaluate<Dual2<float>, float, Dual2<float>, float, true>
      (DFLOAT(out), *(float *)x, (float *) knots, knot_count, knot_arraylen);
}

OSL_SHADEOP OSL_HOSTDEVICE void osl_spline_dfdff(void *out, const char *spline_, void *x,
                                   void* knots, int knot_count, int knot_arraylen)
{
  Spline::SplineInterp::create(HDSTR(spline_)).evaluate<Dual2<float>, Dual2<float>, float, float, false>
      (DFLOAT(out), DFLOAT(x), (float *) knots, knot_count, knot_arraylen);
}

OSL_SHADEOP OSL_HOSTDEVICE void osl_spline_vfv(void *out, const char *spline_, void *x,
                                 void *knots, int knot_count, int knot_arraylen)
{
  Spline::SplineInterp::create(HDSTR(spline_)).evaluate<Vec3, float, Vec3, Vec3, false>
      (*(Vec3 *)out, *(float *)x, (Vec3 *) knots, knot_count, knot_arraylen);
}

OSL_SHADEOP OSL_HOSTDEVICE void osl_spline_dvdfv(void *out, const char *spline_, void *x,
                                   void *knots, int knot_count, int knot_arraylen)
{
  Spline::SplineInterp::create(HDSTR(spline_)).evaluate<Dual2<Vec3>, Dual2<float>, Vec3, Vec3, false>
      (DVEC(out), DFLOAT(x), (Vec3 *) knots, knot_count, knot_arraylen);
}

OSL_SHADEOP OSL_HOSTDEVICE void osl_spline_dvfdv(void *out, const char *spline_, void *x,
                                    void *knots, int knot_count, int knot_arraylen)
{
  Spline::SplineInterp::create(HDSTR(spline_)).evaluate<Dual2<Vec3>, float, Dual2<Vec3>, Vec3, true>
      (DVEC(out), *(float *)x, (Vec3 *) knots, knot_count, knot_arraylen);
}

OSL_SHADEOP OSL_HOSTDEVICE void osl_spline_dvdfdv(void *out, const char *spline_, void *x,
                                    void *knots, int knot_count, int knot_arraylen)
{
  Spline::SplineInterp::create(HDSTR(spline_)).evaluate<Dual2<Vec3>, Dual2<float>, Dual2<Vec3>, Vec3, true>
      (DVEC(out), DFLOAT(x), (Vec3 *) knots, knot_count, knot_arraylen);
}

OSL_SHADEOP OSL_HOSTDEVICE void osl_splineinverse_fff(void *out, const char *spline_, void *x,
                                       void* knots, int knot_count, int knot_arraylen)
{
    // Version with no derivs
  Spline::SplineInterp::create(HDSTR(spline_)).inverse<float>
      (*(float *)out, *(float *)x, (float *) knots, knot_count, knot_arraylen);
}

OSL_SHADEOP OSL_HOSTDEVICE void osl_splineinverse_dfdff(void *out, const char *spline_, void *x,
                                         void* knots, int knot_count, int knot_arraylen)
{
    // x has derivs, so return derivs as well
  Spline::SplineInterp::create(HDSTR(spline_)).inverse<Dual2<float> >
      (DFLOAT(out), DFLOAT(x), (float *) knots, knot_count, knot_arraylen);
}

OSL_SHADEOP OSL_HOSTDEVICE void osl_splineinverse_dfdfdf(void *out, const char *spline_, void *x,
                                          void* knots, int knot_count, int knot_arraylen)
{
    // Ignore knot derivatives
    osl_splineinverse_dfdff (out, spline_, x, (float *) knots, knot_count, knot_arraylen);
}

OSL_SHADEOP OSL_HOSTDEVICE void osl_splineinverse_dffdf(void *out, const char *spline_, void *x,
                                         void* knots, int knot_count, int knot_arraylen)
{
    // Ignore knot derivs
    float outtmp = 0;
    osl_splineinverse_fff (&outtmp, spline_, x, (float *) knots, knot_count, knot_arraylen);
    DFLOAT(out) = outtmp;
}



} // namespace pvt
OSL_NAMESPACE_EXIT
