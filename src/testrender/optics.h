// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>


OSL_NAMESPACE_ENTER

#ifndef OSL_HOSTDEVICE
#  ifdef __CUDACC__
#    define OSL_HOSTDEVICE __host__ __device__
#  else
#    define OSL_HOSTDEVICE
#  endif
#endif

inline OSL_HOSTDEVICE float
fresnel_dielectric(float cosi, float eta)
{
    // special case: ignore fresnel
    if (eta == 0)
        return 1;

    // compute fresnel reflectance without explicitly computing the refracted direction
    if (cosi < 0.0f)
        eta = 1.0f / eta;
    float c = fabsf(cosi);
    float g = eta * eta - 1 + c * c;
    if (g > 0) {
        g       = sqrtf(g);
        float A = (g - c) / (g + c);
        float B = (c * (g + c) - 1) / (c * (g - c) + 1);
        return 0.5f * A * A * (1 + B * B);
    }
    return 1.0f;  // TIR (no refracted component)
}

inline OSL_HOSTDEVICE float
fresnel_refraction(const Vec3& I, const Vec3& N, float eta, Vec3& T)
{
    // compute refracted direction and fresnel term
    // return value will be 0 if TIR occurs
    // NOTE: I is the incoming ray direction (points toward the surface, normalized)
    //       N is the surface normal (points toward the incoming ray origin, normalized)
    //       T is the outgoing refracted direction (points away from the surface)
    float cosi = -dot(I, N);
    // check which side of the surface we are on
    Vec3 Nn;
    float neta;
    if (cosi > 0) {
        // we are on the outside of the surface, going in
        neta = 1 / eta;
        Nn   = N;
    } else {
        // we are inside the surface,
        cosi = -cosi;
        neta = eta;
        Nn   = -N;
    }
    float arg = 1.0f - (neta * neta * (1.0f - cosi * cosi));
    if (arg >= 0) {
        float dnp = sqrtf(arg);
        float nK  = (neta * cosi) - dnp;
        T         = I * neta + Nn * nK;
        return 1 - fresnel_dielectric(cosi, eta);
    }
    T = make_Vec3(0, 0, 0);
    return 0;
}

OSL_HOSTDEVICE Color3
fresnel_conductor(float cos_theta, Color3 n, Color3 k)
{
    cos_theta       = OIIO::clamp(cos_theta, 0.0f, 1.0f);
    float cosTheta2 = cos_theta * cos_theta;
    float sinTheta2 = 1.0f - cosTheta2;
    Color3 n2       = n * n;
    Color3 k2       = k * k;
    Color3 t0       = n2 - k2 - Color3(sinTheta2);
    Color3 a2plusb2(sqrtf(t0.x * t0.x + 4.0f * n2.x * k2.x),
                    sqrtf(t0.y * t0.y + 4.0f * n2.y * k2.y),
                    sqrtf(t0.z * t0.z + 4.0f * n2.z * k2.z));
    Color3 t1 = a2plusb2 + Color3(cosTheta2);
    Color3 a(sqrtf(std::max(0.5f * (a2plusb2.x + t0.x), 0.0f)),
             sqrtf(std::max(0.5f * (a2plusb2.y + t0.y), 0.0f)),
             sqrtf(std::max(0.5f * (a2plusb2.z + t0.z), 0.0f)));
    Color3 t2 = (2.0f * cos_theta) * a;
    Color3 rs = (t1 - t2) / (t1 + t2);

    Color3 t3 = cosTheta2 * a2plusb2 + Color3(sinTheta2 * sinTheta2);
    Color3 t4 = t2 * sinTheta2;
    Color3 rp = rs * (t3 - t4) / (t3 + t4);

    return 0.5f * (rp + rs);
}

inline OSL_HOSTDEVICE float
fresnel_schlick(float cos_theta, float F0, float F90)
{
    float x  = OIIO::clamp(1.0f - cos_theta, 0.0f, 1.0f);
    float x2 = x * x;
    float x4 = x2 * x2;
    float x5 = x4 * x;
    return OIIO::lerp(F0, F90, x5);
}

inline OSL_HOSTDEVICE Color3
fresnel_generalized_schlick(float cos_theta, Color3 F0, Color3 F90,
                            float exponent)
{
    float x = OIIO::clamp(1.0f - cos_theta, 0.0f, 1.0f);
    float m = OIIO::fast_pow_pos(x, exponent);
    return OIIO::lerp(F0, F90, m);
}

OSL_NAMESPACE_EXIT
