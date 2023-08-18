// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of matrix operations.
///
/////////////////////////////////////////////////////////////////////////

#include "oslexec_pvt.h"
#include <OSL/device_string.h>
#include <OSL/dual.h>
#include <OSL/dual_vec.h>
#include <OSL/fmt_util.h>

#include <OpenImageIO/fmath.h>
#include <OpenImageIO/simd.h>

#include <cmath>
#include <iostream>



OSL_NAMESPACE_ENTER
namespace pvt {



// Matrix ops

OSL_SHADEOP OSL_HOSTDEVICE void
osl_mul_mmm(void* r, void* a, void* b)
{
    MAT(r) = MAT(a) * MAT(b);
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_mul_mmf(void* r, void* a, float b)
{
    MAT(r) = MAT(a) * b;
}


OSL_SHADEOP OSL_HOSTDEVICE void
osl_div_mmm(void* r, void* a, void* b)
{
    MAT(r) = MAT(a) * MAT(b).inverse();
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_div_mmf(void* r, void* a, float b)
{
    MAT(r) = MAT(a) * (1.0f / b);
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_div_mfm(void* r, float a, void* b)
{
    MAT(r) = a * MAT(b).inverse();
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_div_m_ff(void* r, float a, float b)
{
    float f = (b == 0) ? 0.0f : (a / b);
    MAT(r)  = Matrix44(f, 0, 0, 0, 0, f, 0, 0, 0, 0, f, 0, 0, 0, 0, f);
}



OSL_SHADEOP OSL_HOSTDEVICE void
osl_transpose_mm(void* r, void* m)
{
    //MAT(r) = MAT(m).transposed();
    MAT(r) = inlinedTransposed(MAT(m));
}


// point = M * point
OSL_SHADEOP OSL_HOSTDEVICE void
osl_transform_vmv(void* result, void* M_, void* v_)
{
    const Vec3& v     = VEC(v_);
    const Matrix44& M = MAT(M_);
    robust_multVecMatrix(M, v, VEC(result));
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_transform_dvmdv(void* result, void* M_, void* v_)
{
    const Dual2<Vec3>& v = DVEC(v_);
    const Matrix44& M    = MAT(M_);
    robust_multVecMatrix(M, v, DVEC(result));
}

// vector = M * vector
OSL_SHADEOP OSL_HOSTDEVICE void
osl_transformv_vmv(void* result, void* M_, void* v_)
{
    const Vec3& v     = VEC(v_);
    const Matrix44& M = MAT(M_);
    //M.multDirMatrix (v, VEC(result));
    multDirMatrix(M, v, VEC(result));
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_transformv_dvmdv(void* result, void* M_, void* v_)
{
    const Dual2<Vec3>& v = DVEC(v_);
    const Matrix44& M    = MAT(M_);
    multDirMatrix(M, v, DVEC(result));
}


// normal = M * normal
OSL_SHADEOP OSL_HOSTDEVICE void
osl_transformn_vmv(void* result, void* M_, void* v_)
{
    const Vec3& v     = VEC(v_);
    const Matrix44& M = MAT(M_);
    //M.inverse().transposed().multDirMatrix (v, VEC(result));
    multDirMatrix(inlinedTransposed(M.inverse()), v, VEC(result));
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_transformn_dvmdv(void* result, void* M_, void* v_)
{
    const Dual2<Vec3>& v = DVEC(v_);
    const Matrix44& M    = MAT(M_);
    //multDirMatrix (M.inverse().transposed(), v, DVEC(result));
    multDirMatrix(inlinedTransposed(M.inverse()), v, DVEC(result));
}

#ifndef __CUDACC__
OSL_SHADEOP int
osl_get_matrix(OpaqueExecContextPtr oec, void* r, const char* from)
{
    if (HDSTR(from) == STRING_PARAMS(common)
        || HDSTR(from) == get_commonspace_synonym(oec)) {
        MAT(r).makeIdentity();
        return true;
    }
    if (HDSTR(from) == STRING_PARAMS(shader)) {
        rs_get_matrix_xform_time(oec, MAT(r), get_shader2common(oec),
                                 get_time(oec));
        return true;
    }
    if (HDSTR(from) == STRING_PARAMS(object)) {
        rs_get_matrix_xform_time(oec, MAT(r), get_object2common(oec),
                                 get_time(oec));
        return true;
    }
    int ok = rs_get_matrix_space_time(oec, MAT(r), HDSTR(from), get_time(oec));
    if (!ok) {
        MAT(r).makeIdentity();
        if (get_unknown_coordsys_error(oec)) {
            OSL::errorfmt(oec, "Unknown transformation \"{}\"", HDSTR(from));
        }
    }
    return ok;
}



OSL_SHADEOP int
osl_get_inverse_matrix(OpaqueExecContextPtr oec, void* r, const char* to)
{
    if (HDSTR(to) == STRING_PARAMS(common)
        || HDSTR(to) == get_commonspace_synonym(oec)) {
        MAT(r).makeIdentity();
        return true;
    }
    if (HDSTR(to) == STRING_PARAMS(shader)) {
        rs_get_inverse_matrix_xform_time(oec, MAT(r), get_shader2common(oec),
                                         get_time(oec));
        return true;
    }
    if (HDSTR(to) == STRING_PARAMS(object)) {
        rs_get_inverse_matrix_xform_time(oec, MAT(r), get_object2common(oec),
                                         get_time(oec));
        return true;
    }
    int ok = rs_get_inverse_matrix_space_time(oec, MAT(r), HDSTR(to),
                                              get_time(oec));
    if (!ok) {
        MAT(r).makeIdentity();
        if (get_unknown_coordsys_error(oec)) {
            OSL::errorfmt(oec, "Unknown transformation \"{}\"", HDSTR(to));
        }
    }
    return ok;
}
#else
// Implemented by the renderer
#    define OSL_SHADEOP_EXPORT extern "C" OSL_DLL_EXPORT
OSL_SHADEOP_EXPORT OSL_HOSTDEVICE int
osl_get_matrix(OpaqueExecContextPtr oec, void* r, const char* from);
OSL_SHADEOP_EXPORT OSL_HOSTDEVICE int
osl_get_inverse_matrix(OpaqueExecContextPtr oec, void* r, const char* to);
#    undef OSL_SHADEOP_EXPORT
#endif  // __CUDACC__



OSL_SHADEOP OSL_HOSTDEVICE int
osl_prepend_matrix_from(OpaqueExecContextPtr oec, void* r, const char* from)
{
    Matrix44 m;
    bool ok = osl_get_matrix(oec, &m, from);
    if (ok)
        MAT(r) = m * MAT(r);
#ifndef __CUDACC__
    // TODO: How do we manage this in OptiX?
    else {
        if (get_unknown_coordsys_error(oec)) {
            OSL::errorfmt(oec, "Unknown transformation \"{}\"", HDSTR(from));
        }
    }
#endif
    return ok;
}



OSL_SHADEOP OSL_HOSTDEVICE int
osl_get_from_to_matrix(OpaqueExecContextPtr oec, void* r, const char* from,
                       const char* to)
{
    Matrix44 Mfrom, Mto;
    int ok = osl_get_matrix(oec, &Mfrom, from);
    ok &= osl_get_inverse_matrix(oec, &Mto, to);
    MAT(r) = Mfrom * Mto;
    return ok;
}



OSL_SHADEOP OSL_HOSTDEVICE int
osl_transform_triple(OpaqueExecContextPtr oec, void* Pin, int Pin_derivs,
                     void* Pout, int Pout_derivs, void* from, void* to,
                     int vectype)
{
    Matrix44 M;
    int ok;
    Pin_derivs &= Pout_derivs;  // ignore derivs if output doesn't need it
    if (HDSTR(from) == STRING_PARAMS(common))
        ok = osl_get_inverse_matrix(oec, &M, (const char*)to);
    else if (HDSTR(to) == STRING_PARAMS(common))
        ok = osl_get_matrix(oec, &M, (const char*)from);
    else
        ok = osl_get_from_to_matrix(oec, &M, (const char*)from,
                                    (const char*)to);
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
        else
            OSL_DASSERT(0 && "Unknown transform type");
#else
        // TBR: Is the ok?
        else
            ok = false;
#endif
    } else {
        *(Vec3*)Pout = *(Vec3*)Pin;
        if (Pin_derivs) {
            ((Vec3*)Pout)[1] = ((Vec3*)Pin)[1];
            ((Vec3*)Pout)[2] = ((Vec3*)Pin)[2];
        }
    }
    if (Pout_derivs && !Pin_derivs) {
        ((Vec3*)Pout)[1].setValue(0.0f, 0.0f, 0.0f);
        ((Vec3*)Pout)[2].setValue(0.0f, 0.0f, 0.0f);
    }
    return ok;
}



OSL_SHADEOP OSL_HOSTDEVICE int
osl_transform_triple_nonlinear(OpaqueExecContextPtr oec, void* Pin,
                               int Pin_derivs, void* Pout, int Pout_derivs,
                               void* from, void* to, int vectype)
{
#ifndef __CUDACC__

    if (rs_transform_points(oec, HDSTR(from), HDSTR(to), get_time(oec),
                            (const Vec3*)Pin, (Vec3*)Pout, 1,
                            (TypeDesc::VECSEMANTICS)vectype)) {
        // Renderer had a direct way to transform the points between the
        // two spaces.
        if (Pout_derivs) {
            if (Pin_derivs) {
                rs_transform_points(oec, HDSTR(from), HDSTR(to), get_time(oec),
                                    (const Vec3*)Pin + 1, (Vec3*)Pout + 1, 2,
                                    TypeDesc::VECTOR);
            } else {
                ((Vec3*)Pout)[1].setValue(0.0f, 0.0f, 0.0f);
                ((Vec3*)Pout)[2].setValue(0.0f, 0.0f, 0.0f);
            }
        }
        return true;
    }
#endif  // __CUDACC__

    // Renderer couldn't or wouldn't transform directly
    // Except in OptiX we're the renderer will directly implement
    // the transform in osl_transform_triple.
    return osl_transform_triple(oec, Pin, Pin_derivs, Pout, Pout_derivs, from,
                                to, vectype);
}



OSL_SHADEOP OSL_HOSTDEVICE float
osl_determinant_fm(void* m)
{
    return det4x4(MAT(m));
}



}  // namespace pvt
OSL_NAMESPACE_EXIT
