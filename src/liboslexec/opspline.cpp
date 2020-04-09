// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

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
#include <OSL/Imathx/Imathx.h>
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
