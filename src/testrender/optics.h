#pragma once

#include <OSL/oslconfig.h>


OSL_NAMESPACE_ENTER

inline float fresnel_dielectric(float cosi, float eta) {
    // special case: ignore fresnel
    if (eta == 0)
        return 1;

    // compute fresnel reflectance without explicitly computing the refracted direction
    if (cosi < 0.0f) eta = 1.0f / eta;
    float c = fabsf(cosi);
    float g = eta * eta - 1 + c * c;
    if (g > 0) {
        g = sqrtf(g);
        float A = (g - c) / (g + c);
        float B = (c * (g + c) - 1) / (c * (g - c) + 1);
        return 0.5f * A * A * (1 + B * B);
    }
    return 1.0f; // TIR (no refracted component)
}

inline float fresnel_refraction(const Dual2<Vec3>& I, const Vec3& N, float eta, Dual2<Vec3>& T) {
    // compute refracted direction and fresnel term
    // return value will be 0 if TIR occurs
    // NOTE: I is the incoming ray direction (points toward the surface, normalized)
    //       N is the surface normal (points toward the incoming ray origin, normalized)
    //       T is the outgoing refracted direction (points away from the surface)
    Dual2<float> cosi = -dot(I, N);
    // check which side of the surface we are on
    Vec3 Nn; float neta;
    if (cosi.val() > 0) {
        // we are on the outside of the surface, going in
        neta = 1 / eta;
        Nn = N;
    } else {
        // we are inside the surface,
        cosi = -cosi;
        neta = eta;
        Nn = -N;
    }
    Dual2<float> arg = 1.0f - (neta * neta * (1.0f - cosi * cosi));
    if (arg.val() >= 0) {
       Dual2<float> dnp = sqrt(arg);
       Dual2<float> nK = (neta * cosi) - dnp;
       T = I * neta + Nn * nK;
       return 1 - fresnel_dielectric(cosi.val(), eta);
    }
    T = make_Vec3(0, 0, 0);
    return 0;
}

OSL_NAMESPACE_EXIT
