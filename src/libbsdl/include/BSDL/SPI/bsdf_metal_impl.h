// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/SPI/bsdf_metal_decl.h>

BSDL_ENTER_NAMESPACE

namespace spi {

BSDL_INLINE_METHOD
MetalFresnel::MetalFresnel(Power c, Power edge, float artist_blend,
                           float artist_power)
    : r(c.clamped(0.0f, 0.99f))
    , g(edge.clamped(0.0f, 1.00f))
    , blend(CLAMP(artist_blend, 0.0f, 1.00f))
    , p(CLAMP(artist_power, 0.0f, 1.00f))
{
}

BSDL_INLINE_METHOD Power
MetalFresnel::eval(float cosine) const
{
    return LERP(blend, fresnel_metal(cosine, r, g, 1),
                fresnel_schlick(cosine, r, g, p));
}

BSDL_INLINE_METHOD Power
MetalFresnel::avg() const
{
    // the following is a mathematica fit to the true integral as a function of r and g:
    // max error is ~2.02%
    // avg error is ~0.25%
    Power metal_avg
        = Power(0.087237f, 1)
          + r
                * (Power(0.782654f, 1) + 0.19744f * g
                   + r * (Power(-0.136432f, 1) - 0.2586f * g + r * 0.278708f))
          + g
                * (Power(0.0230685f, 1)
                   + g
                         * (Power(-0.0864902f, 1) + 0.0360605f * r
                            + g * 0.0774594f));
    metal_avg = metal_avg.clamped(0, 1);

    // Integral[2*c*(r + (g-r) * (1-c)^(1/p),{c,0,1}] happens to have a closed form solution
    const float A     = 2 * SQR(p);
    const float B     = 1 + 3 * p;
    Power schlick_avg = (g * A + r * B) * (1.0f / (A + B));

    return LERP(blend, metal_avg, schlick_avg);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
MetalLobe<BSDF_ROOT>::adjust_reflection(float r, float outer_ior)
{
    // Get IOR for this facing angle reflection, and divide by outer_ior
    const float sqrtr = sqrtf(CLAMP(r, 0.0f, 1.0f));
    const float eta
        = std::max(std::min((1 + sqrtr) / (1 - sqrtr), BIG) / outer_ior, 1.0f);
    // Then map it back to a facing angle reflection
    return SQR((eta - 1) / (1 + eta));
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Power
MetalLobe<BSDF_ROOT>::adjust_reflection(float outer_ior, Power r,
                                        float force_eta)
{
    force_eta = CLAMP(force_eta, 0.0f, 1.0f);
    if (outer_ior == 1.0f || force_eta == 1)
        return r;
    Power adjusted([&](int i) { return adjust_reflection(r[i], outer_ior); },
                   1);
    return LERP(force_eta, adjusted, r);
}

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
MetalLobe<BSDF_ROOT>::MetalLobe(T* lobe, const BsdfGlobals& globals,
                                const Data& data)
    : Base(lobe, globals.visible_normal(data.N), data.U,
           globals.regularize_roughness(data.roughness), globals.lambda_0,
           false)
{
    Base::sample_filter = globals.get_sample_filter(Base::frame.Z, true);
    Power color = adjust_reflection(globals.outer_ior, globals.wave(data.color),
                                    data.force_eta);
    spec = GGX(GGXDist(Base::roughness(), CLAMP(data.anisotropy, 0.0f, 1.0f)),
               MetalFresnel(color, globals.wave(data.edge_tint),
                            data.artist_blend, data.artist_power),
               Base::frame.Z.dot(globals.wo), Base::roughness());
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
MetalLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                const Imath::V3f& wi) const
{
    Sample s    = spec.eval(wo, wi);
    s.roughness = Base::roughness();
    return s;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
MetalLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                  const Imath::V3f& rnd) const
{
    Sample s    = spec.sample(wo, rnd.x, rnd.y, rnd.z);
    s.roughness = Base::roughness();
    return s;
}

}  // namespace spi

BSDL_LEAVE_NAMESPACE
