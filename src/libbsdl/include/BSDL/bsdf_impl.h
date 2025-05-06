// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/bsdf_decl.h>
#include <BSDL/spectrum_impl.h>
#include <BSDL/thinfilm_impl.h>
#include <BSDL/tools.h>

BSDL_ENTER_NAMESPACE

// Update the sum in w (eval / pdf) so it includes ow and opdf from a different
// evaluation. cpdf is the probability of the new contribution (choice pdf).
BSDL_INLINE_METHOD void
Sample::update(Power ow, float opdf, float cpdf)
{
    assert(pdf >= 0);
    assert(opdf >= 0);
    assert(cpdf >= 0);
    assert(cpdf <= 1);

    if (cpdf > PDF_MIN) {
        opdf *= cpdf;
        ow *= 1 / cpdf;
        if (opdf == pdf)
            weight = LERP(0.5f, weight, ow);
        else if (opdf < pdf)
            weight = LERP(1 / (1 + opdf / pdf), ow, weight);
        else
            weight = LERP(1 / (1 + pdf / opdf), weight, ow);
        pdf += opdf;
    }

    assert(pdf >= 0);
}

BSDL_INLINE_METHOD float
Sample::stretch(float x, float min, float length)
{
    assert(x >= min);
    assert(length > 0.0f);
    return std::min((x - min) / length, ALMOSTONE);
}

// Fix the shading normal for the viewing direction Rd. It returns a normal as
// close as possible to N without pointing away from the viewer.
BSDL_INLINE_METHOD Imath::V3f
BsdfGlobals::visible_normal(const Imath::V3f& wo, const Imath::V3f& Ngf,
                            const Imath::V3f& N)
{
    // assert(wo.dot(Ngf) >= 0.0f);
    // Normal visible to the viewer?
    if (wo.dot(N) > 0.0f)
        return N;  // Early exit to avoid wasting cycles

    // Normal of plane containing wo and N
    Imath::V3f V = wo.cross(N);
    if (MAX_ABS_XYZ(V) < 1e-4f) {
        // Degenerate case, we fail to fix the normal, use the original one
        V = wo.cross(Ngf);
        if (MAX_ABS_XYZ(V) < 1e-4f) {
            // Ngf must be too close to wo, pick some V orthogonal to wo,
            // this is borrowed from the frame function
            const float s = copysignf(1.0f, wo.z);
            const float a = -1.0f / (s + wo.z);
            V             = { wo.x * wo.y * a, s + SQR(wo.y) * a, -wo.y };
        } else
            V.normalize();
    } else
        V.normalize();

    // We just want a normal with a positive dot product, we take the one in
    // the limit and turn it a bit towards the viewer.
    return (V.cross(wo) + 1e-4f * wo).normalized();
}

BSDL_INLINE_METHOD Imath::V3f
BsdfGlobals::visible_normal(const Imath::V3f& N) const
{
    return BsdfGlobals::visible_normal(wo, Ngf, N);
}

template<typename T>
BSDL_INLINE_METHOD Power
BsdfGlobals::Filter::eval(const T& lobe, const Imath::V3f& wo,
                          const Imath::V3f& Nf, const Imath::V3f& wi) const
{
    constexpr auto fast_exp = BSDLConfig::Fast::expf;

    Power filter = Power(1, lambda_0);
    if (bump_alpha2 > 0.0f) {
        // This shadowing function is based in the GGX microfacet model
        // I'm commenting out the outgoing shadow part intentionally so we
        // don't change the look too much. That makes it non-symmetrical.
        float light_cos = wi.dot(Nf);
        float t         = copysignf(SQR(light_cos), light_cos);
        if (t > 0.0f /* && s > 0.0f */) {
            float G1 = 2.0f
                       / (1.0f + sqrtf(1.0f + bump_alpha2 * (1.0f - t) / t));
            //float G2 = 2.0f / (1.0f + sqrtf(1.0f + bump_alpha2 * (1.0f - s)/s));
            filter *= G1 /* * G2 */;
        } else
            return Power::ZERO();
    }
    if (sigma_t.max_abs() > 0) {
        // Hacky way of receiving ignore_clearcoat_fix flag, since we don't
        // have access to options here.
        const bool legacy_absorption = sigma_t.min(1) < 0.0f;
        // Only account for the incoming absorption, the one for wo is already
        // applied via filter_o(). Unless ignore_clearcoat_fix is enabled.
        const float dist
            = 1 / fabsf(lobe.frame.Z.dot(wi))
              + (legacy_absorption ? 1 / fabsf(lobe.frame.Z.dot(wo)) : 0);
        filter
            *= Power([&](int i) { return fast_exp(-fabsf(sigma_t[i]) * dist); },
                     lambda_0);
    }
    if (lambda_0 == 0) {
        // TODO: how does this work when spectral?
        Imath::C3f f = thinfilm.get(wo, wi, lobe.roughness());
        filter[0] *= f.x;
        filter[1] *= f.y;
        filter[2] *= f.z;
    }
    return filter;
}

template<typename T>
BSDL_INLINE_METHOD Power
BsdfGlobals::Filter::eval(const T& lobe, const Imath::V3f& wo,
                          const Imath::V3f& Nf, const Imath::V3f& Ngf,
                          const Imath::V3f& wi) const
{
    if (!lobe.transmissive() && Ngf.dot(wi) <= 0)
        return Power::ZERO();
    return eval<T>(lobe, wo, Nf, wi);
}

BSDL_INLINE_METHOD float
BsdfGlobals::regularize_roughness(float roughness) const
{
    const float roughness_product = 1.0f - path_roughness;
    return 1.0f - (1.0f - roughness) * roughness_product;
}

BSDL_INLINE_METHOD Power
BsdfGlobals::wave(const Imath::C3f& c) const
{
    return Power(c, lambda_0);
}

BSDL_INLINE_METHOD BsdfGlobals::Filter
BsdfGlobals::get_sample_filter(const Imath::V3f& N, bool bump_shadow) const
{
    const float cosNNs = N.dot(Nf);
    float bump_alpha2  = 0;
    if (bump_shadow && cosNNs < (1.0f - 1e-4f)) {
        // If the shading normal differs from Nf, compute a plausible
        // roughness for a normal distribution assuming the tangent is at 2
        // standard deviations. Knowing thar GGX tangent variance is about
        // 2 alpha^2 we get this mappinga:
        bump_alpha2 = CLAMP(0.125f * (1 - SQR(cosNNs)) / SQR(cosNNs), 0.0f,
                            1.0f);
    }
    return { bump_alpha2, Power::ZERO(), {}, lambda_0 };
}

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
Lobe<BSDF_ROOT>::Lobe(T* child, const Imath::V3f& Z, float r, float l0, bool tr)
    : BSDF_ROOT(child, r, l0, tr), frame(Z)
{
}

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
Lobe<BSDF_ROOT>::Lobe(T* child, const Imath::V3f& Z, const Imath::V3f& X,
                      float r, float l0, bool tr)
    : BSDF_ROOT(child, r, l0, tr), frame(Z, X)
{
}

BSDL_LEAVE_NAMESPACE
