// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <BSDL/MTX/bsdf_conductor_decl.h>

BSDL_ENTER_NAMESPACE

namespace mtx {

BSDL_INLINE_METHOD
ConductorFresnel::ConductorFresnel(Power IOR, Power extinction, float lambda_0)
    : IOR(IOR), extinction(extinction), lambda_0(lambda_0)
{
}

BSDL_INLINE_METHOD Power
ConductorFresnel::eval(float cos_theta) const
{
    cos_theta = CLAMP(cos_theta, 0.0f, 1.0f);
    const Power one(1, lambda_0);
    const Power cosTheta2(cos_theta * cos_theta, lambda_0);
    const Power sinTheta2 = one - cosTheta2;
    const Power& n        = IOR;
    const Power& k        = extinction;
    const Power n2        = n * n;
    const Power k2        = k * k;
    const Power t0        = n2 - k2 - sinTheta2;
    const Power a2plusb2  = sqrt(t0 * t0 + 4 * n2 * k2);
    const Power t1        = a2plusb2 + cosTheta2;
    const Power a         = sqrt(0.5f * (a2plusb2 + t0));
    const Power t2        = (2.0f * cos_theta) * a;
    const Power rs        = (t1 - t2) / (t1 + t2).clamped(FLOAT_MIN, BIG);

    const Power t3 = cosTheta2 * a2plusb2 + sinTheta2 * sinTheta2;
    const Power t4 = t2 * sinTheta2;
    const Power rp = rs * (t3 - t4) / (t3 + t4).clamped(FLOAT_MIN, BIG);

    return 0.5f * (rp + rs).clamped(0, 2);
}

BSDL_INLINE_METHOD Power
ConductorFresnel::F0() const
{
    const Power one(1, lambda_0);
    const Power& n       = IOR;
    const Power& k       = extinction;
    const Power n2       = n * n;
    const Power k2       = k * k;
    const Power t0       = n2 - k2;
    const Power a2plusb2 = sqrt(t0 * t0 + 4 * n2 * k2);
    const Power t1       = a2plusb2 + one;
    const Power a        = sqrt(0.5f * (a2plusb2 + t0));
    const Power t2       = 2.0f * a;
    const Power rs       = (t1 - t2) / (t1 + t2).clamped(FLOAT_MIN, BIG);

    return rs.clamped(0, 1);
}

BSDL_INLINE_METHOD Power
ConductorFresnel::avg() const
{
    return Power(
        [&](int i) {
            // Very simple fit for cosine weighted average fresnel. Not very
            // accurate but enough for albedo based sampling decisions.
            constexpr float a = -0.32775145f, b = 0.18346033f, c = 0.61146583f,
                            d = -0.07785134f;
            const float x = IOR[i], y = extinction[i];
            const float p = a + b * x + c * y + d * x * y;
            return p / (1 + p);
        },
        lambda_0);
}

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
ConductorLobe<BSDF_ROOT>::ConductorLobe(T* lobe, const BsdfGlobals& globals,
                                        const Data& data)
    : Base(lobe, globals.visible_normal(data.N), data.U, 0.0f, globals.lambda_0,
           false)
{
    Base::sample_filter = globals.get_sample_filter(Base::frame.Z, true);
    // MaterialX expects the raw x/y roughness as input, but for albedo tables it
    // is better to use the roughness/anisotropy parametrization so we can
    // ignore roughness
    const float rx    = CLAMP(data.roughness_x, EPSILON, 2.0f);
    const float ry    = CLAMP(data.roughness_y, EPSILON, 2.0f);
    const float ax    = std::max(rx, ry);
    const float ay    = std::min(rx, ry);
    const float b     = ay / ax;
    const float aniso = (1 - b) / (1 + b);
    // Also assume we square the roughness for linearity
    const float roughness = globals.regularize_roughness(
        sqrtf(ax / (1 + aniso)));
    const float cosNO = Base::frame.Z.dot(globals.wo);
    // Flip aniso if roughness_x < roughness_y
    dist    = GGXDist(roughness, aniso, (rx < ry));
    fresnel = ConductorFresnel(globals.wave(data.IOR),
                               globals.wave(data.extinction), globals.lambda_0);
    TabulatedEnergyCurve<spi::MiniMicrofacetGGX> curve(roughness, 0.0f);
    E_ms = curve.Emiss_eval(cosNO);
    Base::set_roughness(roughness);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
ConductorLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                    const Imath::V3f& wi) const
{
    Sample s    = eval_turquin_microms_reflection(dist, fresnel, E_ms, wo, wi);
    s.roughness = Base::roughness();
    return s;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
ConductorLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                      const Imath::V3f& rnd) const
{
    Sample s = sample_turquin_microms_reflection(dist, fresnel, E_ms, wo, rnd);
    s.roughness = Base::roughness();
    return s;
}

}  // namespace mtx

BSDL_LEAVE_NAMESPACE
