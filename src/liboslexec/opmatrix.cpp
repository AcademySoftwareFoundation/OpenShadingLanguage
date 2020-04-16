// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage


/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of matrix operations.
///
/////////////////////////////////////////////////////////////////////////

#include "oslexec_pvt.h"
#include <OSL/dual.h>
#include <OSL/dual_vec.h>
#include <OSL/device_string.h>

#include <OpenImageIO/fmath.h>
#include <OpenImageIO/simd.h>

#include <iostream>
#include <cmath>



OSL_NAMESPACE_ENTER
namespace pvt {



// Matrix ops

OSL_SHADEOP OSL_HOSTDEVICE void
osl_mul_mmm (void *r, void *a, void *b)
{
    MAT(r) = MAT(a) * MAT(b);
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_mul_mmf (void *r, void *a, float b)
{
    MAT(r) = MAT(a) * b;
}


OSL_SHADEOP OSL_HOSTDEVICE void
osl_div_mmm (void *r, void *a, void *b)
{
    MAT(r) = MAT(a) * MAT(b).inverse();
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_div_mmf (void *r, void *a, float b)
{
    MAT(r) = MAT(a) * (1.0f/b);
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_div_mfm (void *r, float a, void *b)
{
    MAT(r) = a * MAT(b).inverse();
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_div_m_ff (void *r, float a, float b)
{
    float f = (b == 0) ? 0.0f : (a / b);
    MAT(r) = Matrix44 (f,0,0,0, 0,f,0,0, 0,0,f,0, 0,0,0,f);
}



OSL_SHADEOP OSL_HOSTDEVICE void
osl_transpose_mm (void *r, void *m)
{
    //MAT(r) = MAT(m).transposed();
	MAT(r) = inlinedTransposed(MAT(m));
}


// point = M * point
OSL_SHADEOP OSL_HOSTDEVICE void osl_transform_vmv(void *result, void* M_, void* v_)
{
   const Vec3 &v = VEC(v_);
   const Matrix44 &M = MAT(M_);
   robust_multVecMatrix (M, v, VEC(result));
}

OSL_SHADEOP OSL_HOSTDEVICE void osl_transform_dvmdv(void *result, void* M_, void* v_)
{
   const Dual2<Vec3> &v = DVEC(v_);
   const Matrix44    &M = MAT(M_);
   robust_multVecMatrix (M, v, DVEC(result));
}

// vector = M * vector
OSL_SHADEOP OSL_HOSTDEVICE void osl_transformv_vmv(void *result, void* M_, void* v_)
{
   const Vec3 &v = VEC(v_);
   const Matrix44 &M = MAT(M_);
   //M.multDirMatrix (v, VEC(result));
   multDirMatrix (M, v, VEC(result));
}

OSL_SHADEOP OSL_HOSTDEVICE void osl_transformv_dvmdv(void *result, void* M_, void* v_)
{
   const Dual2<Vec3> &v = DVEC(v_);
   const Matrix44    &M = MAT(M_);
   multDirMatrix (M, v, DVEC(result));
}


// normal = M * normal
OSL_SHADEOP OSL_HOSTDEVICE void osl_transformn_vmv(void *result, void* M_, void* v_)
{
   const Vec3 &v = VEC(v_);
   const Matrix44 &M = MAT(M_);
   //M.inverse().transposed().multDirMatrix (v, VEC(result));
   multDirMatrix(inlinedTransposed(M.inverse()), v, VEC(result));
}

OSL_SHADEOP OSL_HOSTDEVICE void osl_transformn_dvmdv(void *result, void* M_, void* v_)
{
   const Dual2<Vec3> &v = DVEC(v_);
   const Matrix44    &M = MAT(M_);
   //multDirMatrix (M.inverse().transposed(), v, DVEC(result));
   multDirMatrix (inlinedTransposed(M.inverse()), v, DVEC(result));
}

#ifndef __CUDACC__
OSL_SHADEOP int
osl_get_matrix (void *sg_, void *r, const char *from)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    ShadingContext *ctx = (ShadingContext *)sg->context;
    if (USTR(from) == Strings::common ||
            USTR(from) == ctx->shadingsys().commonspace_synonym()) {
        MAT(r).makeIdentity ();
        return true;
    }
    if (USTR(from) == Strings::shader) {
        ctx->renderer()->get_matrix (sg, MAT(r), sg->shader2common, sg->time);
        return true;
    }
    if (USTR(from) == Strings::object) {
        ctx->renderer()->get_matrix (sg, MAT(r), sg->object2common, sg->time);
        return true;
    }
    int ok = ctx->renderer()->get_matrix (sg, MAT(r), USTR(from), sg->time);
    if (! ok) {
        MAT(r).makeIdentity();
        ShadingContext *ctx = (ShadingContext *)((ShaderGlobals *)sg)->context;
        if (ctx->shadingsys().unknown_coordsys_error())
            ctx->errorf("Unknown transformation \"%s\"", from);
    }
    return ok;
}



OSL_SHADEOP int
osl_get_inverse_matrix (void *sg_, void *r, const char *to)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    ShadingContext *ctx = (ShadingContext *)sg->context;
    if (USTR(to) == Strings::common ||
            USTR(to) == ctx->shadingsys().commonspace_synonym()) {
        MAT(r).makeIdentity ();
        return true;
    }
    if (USTR(to) == Strings::shader) {
        ctx->renderer()->get_inverse_matrix (sg, MAT(r), sg->shader2common, sg->time);
        return true;
    }
    if (USTR(to) == Strings::object) {
        ctx->renderer()->get_inverse_matrix (sg, MAT(r), sg->object2common, sg->time);
        return true;
    }
    int ok = ctx->renderer()->get_inverse_matrix (sg, MAT(r), USTR(to), sg->time);
    if (! ok) {
        MAT(r).makeIdentity ();
        ShadingContext *ctx = (ShadingContext *)((ShaderGlobals *)sg)->context;
        if (ctx->shadingsys().unknown_coordsys_error())
            ctx->errorf("Unknown transformation \"%s\"", to);
    }
    return ok;
}
#else
// Implemented by the renderer
#define OSL_SHADEOP_EXPORT extern "C" OSL_DLL_EXPORT
OSL_SHADEOP_EXPORT OSL_HOSTDEVICE int osl_get_matrix (void *sg_, void *r, const char *from);
OSL_SHADEOP_EXPORT OSL_HOSTDEVICE int osl_get_inverse_matrix (void *sg_, void *r, const char *to);
#undef OSL_SHADEOP_EXPORT
#endif // __CUDACC__



OSL_SHADEOP OSL_HOSTDEVICE int
osl_prepend_matrix_from (void *sg, void *r, const char *from)
{
    Matrix44 m;
    bool ok = osl_get_matrix ((ShaderGlobals *)sg, &m, from);
    if (ok)
        MAT(r) = m * MAT(r);
#ifndef __CUDACC__
    // TODO: How do we manage this in OptiX?
    else {
        ShadingContext *ctx = (ShadingContext *)((ShaderGlobals *)sg)->context;
        if (ctx->shadingsys().unknown_coordsys_error())
            ctx->errorf("Unknown transformation \"%s\"", from);
    }
#endif
    return ok;
}



OSL_SHADEOP OSL_HOSTDEVICE int
osl_get_from_to_matrix (void *sg, void *r, const char *from, const char *to)
{
    Matrix44 Mfrom, Mto;
    int ok = osl_get_matrix ((ShaderGlobals *)sg, &Mfrom, from);
    ok &= osl_get_inverse_matrix ((ShaderGlobals *)sg, &Mto, to);
    MAT(r) = Mfrom * Mto;
    return ok;
}



OSL_SHADEOP OSL_HOSTDEVICE int
osl_transform_triple (void *sg_, void *Pin, int Pin_derivs,
                      void *Pout, int Pout_derivs,
                      void *from, void *to, int vectype)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    Matrix44 M;
    int ok;
    Pin_derivs &= Pout_derivs;   // ignore derivs if output doesn't need it
    if (HDSTR(from) == StringParams::common)
        ok = osl_get_inverse_matrix (sg, &M, (const char *)to);
    else if (HDSTR(to) == StringParams::common)
        ok = osl_get_matrix (sg, &M, (const char *)from);
    else
        ok = osl_get_from_to_matrix (sg, &M, (const char *)from,
                                     (const char *)to);
    if (ok) {
        if (vectype == TypeDesc::POINT) {
            if (Pin_derivs)
                osl_transform_dvmdv(Pout, &M, Pin);
            else
                osl_transform_vmv(Pout, &M, Pin);
        } else if (vectype == TypeDesc::VECTOR) {
            if (Pin_derivs)
                osl_transformv_dvmdv(Pout, &M, Pin);
            else
                osl_transformv_vmv(Pout, &M, Pin);
        } else if (vectype == TypeDesc::NORMAL) {
            if (Pin_derivs)
                osl_transformn_dvmdv(Pout, &M, Pin);
            else
                osl_transformn_vmv(Pout, &M, Pin);
        }
#ifndef __CUDACC__
        else OSL_DASSERT(0 && "Unknown transform type");
#else
        // TBR: Is the ok?
        else ok = false;
#endif
    } else {
        *(Vec3 *)Pout = *(Vec3 *)Pin;
        if (Pin_derivs) {
            ((Vec3 *)Pout)[1] = ((Vec3 *)Pin)[1];
            ((Vec3 *)Pout)[2] = ((Vec3 *)Pin)[2];
        }
    }
    if (Pout_derivs && !Pin_derivs) {
        ((Vec3 *)Pout)[1].setValue (0.0f, 0.0f, 0.0f);
        ((Vec3 *)Pout)[2].setValue (0.0f, 0.0f, 0.0f);
    }
    return ok;
}



OSL_SHADEOP OSL_HOSTDEVICE int
osl_transform_triple_nonlinear (void *sg_, void *Pin, int Pin_derivs,
                                void *Pout, int Pout_derivs,
                                void *from, void *to,
                                int vectype)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
#ifndef __CUDACC__
    RendererServices *rend = sg->renderer;
    if (rend->transform_points (sg, USTR(from), USTR(to), sg->time,
                                (const Vec3 *)Pin, (Vec3 *)Pout, 1,
                                (TypeDesc::VECSEMANTICS)vectype)) {
        // Renderer had a direct way to transform the points between the
        // two spaces.
        if (Pout_derivs) {
            if (Pin_derivs) {
                rend->transform_points (sg, USTR(from), USTR(to), sg->time,
                                        (const Vec3 *)Pin+1,
                                        (Vec3 *)Pout+1, 2, TypeDesc::VECTOR);
            } else {
                ((Vec3 *)Pout)[1].setValue (0.0f, 0.0f, 0.0f);
                ((Vec3 *)Pout)[2].setValue (0.0f, 0.0f, 0.0f);
            }
        }
        return true;
    }
#endif // __CUDACC__

    // Renderer couldn't or wouldn't transform directly
    // Except in OptiX we're the renderer will directly implement
    // the transform in osl_transform_triple.
    return osl_transform_triple (sg, Pin, Pin_derivs, Pout, Pout_derivs,
                                 from, to, vectype);
}


// Calculate the determinant of a 2x2 matrix.
template <typename F>
OSL_HOSTDEVICE inline F det2x2(F a, F b, F c, F d)
{
    return a * d - b * c;
}

// calculate the determinant of a 3x3 matrix in the form:
//     | a1,  b1,  c1 |
//     | a2,  b2,  c2 |
//     | a3,  b3,  c3 |
template <typename F>
OSL_HOSTDEVICE inline F det3x3(F a1, F a2, F a3, F b1, F b2, F b3, F c1, F c2, F c3)
{
    return a1 * det2x2( b2, b3, c2, c3 )
         - b1 * det2x2( a2, a3, c2, c3 )
         + c1 * det2x2( a2, a3, b2, b3 );
}

// calculate the determinant of a 4x4 matrix.
template <typename F>
OSL_HOSTDEVICE inline F det4x4(const Imath::Matrix44<F> &m)
{
    // assign to individual variable names to aid selecting correct elements
    F a1 = m.x[0][0], b1 = m.x[0][1], c1 = m.x[0][2], d1 = m.x[0][3];
    F a2 = m.x[1][0], b2 = m.x[1][1], c2 = m.x[1][2], d2 = m.x[1][3];
    F a3 = m.x[2][0], b3 = m.x[2][1], c3 = m.x[2][2], d3 = m.x[2][3];
    F a4 = m.x[3][0], b4 = m.x[3][1], c4 = m.x[3][2], d4 = m.x[3][3];
    return a1 * det3x3( b2, b3, b4, c2, c3, c4, d2, d3, d4)
         - b1 * det3x3( a2, a3, a4, c2, c3, c4, d2, d3, d4)
         + c1 * det3x3( a2, a3, a4, b2, b3, b4, d2, d3, d4)
         - d1 * det3x3( a2, a3, a4, b2, b3, b4, c2, c3, c4);
}

OSL_SHADEOP OSL_HOSTDEVICE float
osl_determinant_fm (void *m)
{
    return det4x4 (MAT(m));
}




} // namespace pvt
OSL_NAMESPACE_EXIT
