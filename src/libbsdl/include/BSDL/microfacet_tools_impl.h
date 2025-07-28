// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <BSDL/config.h>
#include <BSDL/microfacet_tools_decl.h>
#include <BSDL/tools.h>

#ifndef BAKE_BSDL_TABLES
#    include <BSDL/SPI/microfacet_tools_luts.h>
#endif

#include <Imath/ImathVec.h>

BSDL_ENTER_NAMESPACE

BSDL_INLINE_METHOD float
GGXDist::D(const Imath::V3f& Hr) const
{
    // Have these two multiplied by sinThetaM2 for convenience
    const float cosPhi2st2 = SQR(Hr.x / ax);
    const float sinPhi2st2 = SQR(Hr.y / ay);
    const float cosThetaM2 = SQR(Hr.z);
    const float sinThetaM2 = cosPhi2st2 + sinPhi2st2;
    return 1.0f / (float(M_PI) * ax * ay * SQR(cosThetaM2 + sinThetaM2));
}

BSDL_INLINE_METHOD float
GGXDist::G1(Imath::V3f w) const
{
    assert(w.z > 0);
    w = { w.x * ax, w.y * ay, w.z };
    return 2.0f * w.z / (w.z + w.length());
}

BSDL_INLINE_METHOD float
GGXDist::G2_G1(Imath::V3f wi, Imath::V3f wo) const
{
    assert(wi.z > 0);
    assert(wo.z > 0);
    wi             = { wi.x * ax, wi.y * ay, wi.z };
    wo             = { wo.x * ax, wo.y * ay, wo.z };
    const float nl = wi.length();
    const float nv = wo.length();
    return wi.z * (wo.z + nv) / (wo.z * nl + wi.z * nv);
}

BSDL_INLINE_METHOD Imath::V3f
GGXDist::sample(const Imath::V3f& wo, float randu, float randv) const
{
    // Much simpler technique from Bruce Walter's tech report on Ellipsoidal NDF's (trac#4776)
    const Imath::V3f V = Imath::V3f(ax * wo.x, ay * wo.y, wo.z).normalized();
    // NOTE: the orientation can "flip" as we approach normal incidence, but there doesn't seem to be a good way to solve this
    const Imath::V3f T1 = V.z < 0.9999f ? Imath::V3f(V.y, -V.x, 0).normalized()
                                        : Imath::V3f(1, 0, 0);
    const Imath::V3f T2 = T1.cross(V);

    // Sample point in circle (concentric mapping)
    Imath::V2f p = square_to_unit_disc({ randu, randv });
    // Offset
    const float s   = 0.5f * (1 + V.z);
    const float p2o = s * p.y + (1 - s) * sqrtf(1 - p.x * p.x);
    // Calculate third component (unit hemisphere)
    const float p3 = sqrtf(std::max(1.0f - SQR(p.x) - SQR(p2o), 0.0f));

    const Imath::V3f N = p.x * T1 + p2o * T2 + p3 * V;
    return Imath::V3f(ax * N.x, ay * N.y, std::max(N.z, 0.0f)).normalized();
}

BSDL_INLINE_METHOD Imath::V3f
GGXDist::sample_for_refl(const Imath::V3f& wo, float randu, float randv) const
{
    // From "Bounded VNDF Sampling for Smith–GGX Reflections" Eto - Tokuyoshi.
    // This code is taken from listing 1 almost verbatim.
    Imath::V3f i_std = Imath::V3f(wo.x * ax, wo.y * ay, wo.z).normalized();
    // Sample a spherical cap
    const float phi = 2.0f * PI * randu;
    const float a   = CLAMP(std::min(ax, ay), 0.0f, 1.0f);  // Eq. 6
    const float s   = 1 + sqrtf(SQR(wo.x) + SQR(wo.y));  // Omit sgn for a <=1
    const float a2  = SQR(a);
    const float s2  = SQR(s);
    const float k   = (1 - a2) * s2 / (s2 + a2 * SQR(wo.z));  // Eq. 5
    const float b   = k * i_std.z;
    const float z   = (1 - randv) * (1 + b) - b;
    const float sinTheta = sqrtf(CLAMP(1 - SQR(z), 0.0f, 1.0f));
    constexpr auto cos   = BSDLConfig::Fast::cosf;
    constexpr auto sin   = BSDLConfig::Fast::sinf;
    Imath::V3f o_std     = { sinTheta * cos(phi), sinTheta * sin(phi), z };
    // Compute the microfacet normal m
    Imath::V3f m_std = i_std + o_std;
    Imath::V3f m = Imath::V3f(m_std.x * ax, m_std.y * ay, m_std.z).normalized();
    return m;
}

BSDL_INLINE_METHOD float
GGXDist::D_refl_D(const Imath::V3f& wo, const Imath::V3f& m) const
{
    // From "Bounded VNDF Sampling for Smith–GGX Reflections" Eto - Tokuyoshi.
    // This code is taken from listing 2 almost verbatim.
    const Imath::V2f ai = { wo.x * ax, wo.y * ay };
    const float len2    = ai.dot(ai);
    const float t       = sqrtf(len2 + SQR(wo.z));
    const float a       = CLAMP(std::min(ax, ay), 0.0f, 1.0f);  // Eq. 6
    const float s  = 1 + sqrtf(SQR(wo.x) + SQR(wo.y));  // Omit sgn for a <=1
    const float a2 = SQR(a);
    const float s2 = SQR(s);
    const float k  = (1 - a2) * s2 / (s2 + a2 * SQR(wo.z));  // Eq. 5
    return 2 * wo.z /* * D(m) */ / (k * wo.z + t);           // Eq. 8
}

template<typename BSDF>
BSDL_INLINE_METHOD float
TabulatedEnergyCurve<BSDF>::interpolate_emiss(int i) const
{
    const float* storedE = BSDF::get_energy().data;
    // interpolate a custom energy compensation curve for the chosen roughness
    float rf = roughness * (BSDF::Nr - 1);
    int ra   = static_cast<int>(rf);
    int rb   = std::min(ra + 1, BSDF::Nr - 1);
    rf -= ra;
    assert(ra >= 0 && ra < BSDF::Nr);
    assert(rb >= 0 && rb < BSDF::Nr);
    assert(rf >= 0 && rf <= 1);

    if (BSDF::Nf == 1) {
        return LERP(rf, storedE[ra * BSDF::Nc + i], storedE[rb * BSDF::Nc + i]);
    } else {
        // bilinear interpolation for the chosen roughness and fresnel
        assert(fresnel_index >= 0);
        assert(fresnel_index <= 1);
        float ff = fresnel_index * (BSDF::Nf - 1);
        int fa   = static_cast<int>(ff);
        int fb   = std::min(fa + 1, BSDF::Nf - 1);
        ff -= fa;

        assert(fa >= 0 && fa < BSDF::Nf);
        assert(fb >= 0 && fb < BSDF::Nf);
        assert(ff >= 0 && ff <= 1);

        return LERP(ff,
                    LERP(rf, storedE[(fa * BSDF::Nr + ra) * BSDF::Nc + i],
                         storedE[(fa * BSDF::Nr + rb) * BSDF::Nc + i]),
                    LERP(rf, storedE[(fb * BSDF::Nr + ra) * BSDF::Nc + i],
                         storedE[(fb * BSDF::Nr + rb) * BSDF::Nc + i]));
    }
}

template<typename BSDF>
BSDL_INLINE_METHOD float
TabulatedEnergyCurve<BSDF>::get_Emiss_avg() const
{
    // integrate 2*c*Emiss(c)
    // note that the table is not uniformly spaced
    // the first entry is a triangle, not a trapezoid, account for it seperately
    float cos0 = BSDF::get_cosine(0);
    float F0   = cos0 * interpolate_emiss(0);  // skip factor of 2 here
    float Emiss_avg
        = F0 * cos0;  // beacuse it cancels out with 0.5 of trapeze area formula

    for (int i = 1; i < BSDF::Nc; i++) {
        const float Emiss = interpolate_emiss(i);
        float cos1        = BSDF::get_cosine(i);
        float F1          = cos1 * Emiss;
        Emiss_avg += (F0 + F1) * (cos1 - cos0);
        cos0 = cos1;
        F0   = F1;
        assert(Emiss >= 0);
        assert(Emiss <= 1);
    }

    assert(Emiss_avg >= 0);
    assert(Emiss_avg <= 1);
    return Emiss_avg;
}

template<typename BSDF>
BSDL_INLINE_METHOD float
TabulatedEnergyCurve<BSDF>::Emiss_eval(float c) const
{
    assert(c >= 0);

    // The get_cosine call may not return uniformly spaced cosines,
    // so we need to account for this in our evaluation

    float cos0 = BSDF::get_cosine(0);
    if (c <= cos0) {
        // lerp to 0 for first bin
        float E = interpolate_emiss(0);
        assert(E >= 0);
        assert(E <= 1);
        return E;
    }
    for (int i = 1; i < BSDF::Nc; i++) {
        const float cos1 = BSDF::get_cosine(i);
        if (c < cos1) {
            float q = (c - cos0) / (cos1 - cos0);
            assert(q >= 0);
            assert(q <= 1);
            float E = LERP(q, interpolate_emiss(i - 1), interpolate_emiss(i));
            assert(E >= 0);
            assert(E <= 1);
            return E;
        }
        cos0 = cos1;
    }
    // just a smidge over 1, just clamp
    return interpolate_emiss(BSDF::Nc - 1);
}

template<typename Dist>
BSDL_INLINE_METHOD Sample
MiniMicrofacet<Dist>::sample(Imath::V3f wo, float randu, float randv,
                             float randw) const
{
    // compute just enough to get the weight
    const Imath::V3f m = d.sample(wo, randu, randv);
    if (m.dot(wo) > 0) {
        const Imath::V3f wi = reflect(wo, m);
        if (wi.z > 0) {
            const float weight = d.G2_G1(wi, wo);
            // NOTE: we just care about the weight in this context, don't bother computing the PDF
            return { wi, Power(weight, 1), 0, d.roughness() };
        }
    }
    return {};
}

template<typename Fresnel>
BSDL_INLINE_METHOD
MicrofacetMS<Fresnel>::MicrofacetMS(float cosNO, float roughness_index,
                                    float fresnel_index)
    : d(roughness_index, 0), f(Fresnel::from_table_index(fresnel_index))
{
    TabulatedEnergyCurve<spi::MiniMicrofacetGGX> curve(
        roughness_index, fresnel_index);  // Energy compensation helper
    Eo     = curve.Emiss_eval(cosNO);
    Eo_avg = curve.get_Emiss_avg();
}

template<typename Fresnel>
BSDL_INLINE_METHOD
MicrofacetMS<Fresnel>::MicrofacetMS(const GGXDist& dist, const Fresnel& fresnel,
                                    float cosNO, float roughness)
    : d(dist), f(fresnel)
{
    TabulatedEnergyCurve<spi::MiniMicrofacetGGX> curve(
        roughness,
        0.0f);  // Energy compensation helper
    Eo     = curve.Emiss_eval(cosNO);
    Eo_avg = curve.get_Emiss_avg();
}

template<typename Fresnel>
BSDL_INLINE_METHOD Sample
MicrofacetMS<Fresnel>::eval(Imath::V3f wo, Imath::V3f wi) const
{
    const float cosNO = wo.z;
    const float cosNI = wi.z;
    if (cosNI <= 0 || cosNO <= 0)
        return {};

    // evaluate multiple scattering lobe:
    Sample s = { wi, computeFmiss(), Eo * cosNI * ONEOVERPI };

    // get half vector
    Imath::V3f m = (wo + wi).normalized();
    float cosMO  = m.dot(wo);
    if (cosMO > 0) {
        // eq. 20: (F*G*D)/(4*in*on)
        // eq. 33: first we calculate D(m) with m=Hr:
        const float D = d.D(m);
        // eq. 34: now calculate G
        const float G1 = d.G1(wo);
        // eq. 24 (over the PDF below)
        const float out = d.G2_G1(wi, wo);
        // convert into pdf of the sampled direction
        // eq. 38 - from the visible micronormal distribution in "Importance
        // Sampling Microfacet-Based BSDFs using the Distribution of Visible
        // Normals", Heitz and d'Eon (equation 2)
        float s_pdf = (G1 * D) / (4.0f * cosNO);
        // fresnel term between outgoing direction and microfacet
        const Power F = f.eval(cosMO);

        // merge specular into
        s.update(F * out, s_pdf, 1 - Eo);
    }
    return s;
}

template<typename Fresnel>
BSDL_INLINE_METHOD Sample
MicrofacetMS<Fresnel>::sample(Imath::V3f wo, float randu, float randv,
                              float randw) const
{
    const float cosNO = wo.z;
    if (cosNO <= 0)
        return {};

    // probability of choosing the energy compensation lobe
    const float ec_prob = Eo;
    Imath::V3f wi;
    if (randu < ec_prob) {
        // sample diffuse energy compensation lobe
        randu = Sample::stretch(randu, 0.0f, ec_prob);
        wi    = sample_cos_hemisphere(randu, randv);
    } else {
        // sample microfacet (half vector)
        // generate outgoing direction
        wi = reflect(wo,
                     d.sample(wo, Sample::stretch(randu, ec_prob, 1 - ec_prob),
                              randv));
    }
    // evaluate brdf on outgoing direction
    return eval(wo, wi);
}

template<typename Fresnel>
BSDL_INLINE_METHOD Power
MicrofacetMS<Fresnel>::computeFmiss() const
{
    // The following expression was derived after discussions with Stephen Hill. This is the biggest difference
    // compared to our earlier implementation.
    // The idea is to model the extra bounces using the average fresnel and average energy of a single bounce, and
    // being careful to only include the energy _beyond_ the first bounce (ie: not double-counting the single-scattering)
    const float Emiss_avg = Eo_avg;
    const Power Favg      = f.avg();
    assert(Favg.min(1) >= 0);
    assert(Favg.max() <= 1);
    Power Fmiss(
        [&](int i) {
            return SQR(Favg[i]) * (1 - Emiss_avg) / (1 - Favg[i] * Emiss_avg);
        },
        1);
    assert(Fmiss.min(1) >= 0);
    assert(Fmiss.max() <= 1);
    return Fmiss;
}

// Turquin style microfacet with multiple scattering
template<typename Fresnel>
BSDL_INLINE Sample
eval_turquin_microms_reflection(const GGXDist& dist, const Fresnel& fresnel,
                                float E_ms, const Imath::V3f& wo,
                                const Imath::V3f& wi)
{
    const float cosNO = wo.z;
    const float cosNI = wi.z;
    if (cosNI <= 0 || cosNO <= 0)
        return {};

    // get half vector
    Imath::V3f m    = (wo + wi).normalized();
    float cosMO     = m.dot(wo);
    const float D   = dist.D(m);
    float D_refl_D  = dist.D_refl_D(wo, m);
    float D_refl    = D_refl_D * D;
    const float G1  = dist.G1(wo);
    float pdf       = D_refl / (4 * cosNO);
    const float out = dist.G2_G1(wi, wo) * G1 / D_refl_D;
    // fresnel term between outgoing direction and microfacet
    const Power F = fresnel.eval(cosMO);
    // From "Practical multiple scattering compensation for microfacet models" - Emmanuel Turquin
    // equation 16. Should we use F0 for msf scaling? Doesn't make a big difference.
    const Power F_ms = F;
    const float msf  = E_ms / std::max(0.01f, 1 - E_ms);
    const Power one(1, 1);
    const Power O = out * F * (one + F_ms * msf);

    return { wi, O, pdf, -1 /* roughness set by caller */ };
}

template<typename Fresnel>
BSDL_INLINE Sample
sample_turquin_microms_reflection(const GGXDist& dist, const Fresnel& fresnel,
                                  float E_ms, const Imath::V3f& wo,
                                  const Imath::V3f& rnd)
{
    const float cosNO = wo.z;
    if (cosNO <= 0)
        return {};

    Imath::V3f m  = dist.sample_for_refl(wo, rnd.x, rnd.y);
    Imath::V3f wi = reflect(wo, m);
    if (wi.z <= 0)
        return {};

    // evaluate brdf on outgoing direction
    return eval_turquin_microms_reflection(dist, fresnel, E_ms, wo, wi);
}

// SPI style microfacet with multiple scattering, with a diffuse lobe
template<typename Dist, typename Fresnel>
BSDL_INLINE Sample
eval_spi_microms_reflection(const Dist& dist, const Fresnel& fresnel,
                            float E_ms, float E_ms_avg, const Imath::V3f& wo,
                            const Imath::V3f& wi)
{
    const float cosNO = wo.z;
    const float cosNI = wi.z;
    if (cosNI <= 0 || cosNO <= 0)
        return {};

    // get half vector
    Imath::V3f m    = (wo + wi).normalized();
    float cosMO     = m.dot(wo);
    const float D   = dist.D(m);
    const float G1  = dist.G1(wo);
    const float out = dist.G2_G1(wi, wo);
    float s_pdf     = (G1 * D) / (4.0f * cosNO);
    // fresnel term between outgoing direction and microfacet
    const Power F = fresnel.eval(cosMO);
    const Power O = out * F;

    const Power one = Power(1, 1);
    // Like Turquin we are using F_avg = F instead of an average fresnel, the
    // render integrator has an average effect for high roughness.
    //const Power F0 = fresnel.F0();
    const Power F_avg = F;
    // Evaluate multiple scattering lobe, roughness set by caller. The
    // 1 / (1 - F_avg * E_ms_avg) term comes from the series summation.
    const Power O_ms = SQR(F_avg) * (1 - E_ms_avg) / (one - F_avg * E_ms_avg);

    Sample ec = { wi, O_ms, E_ms * cosNI * ONEOVERPI, -1 };
    ec.update(O, s_pdf, 1 - E_ms);

    return ec;
}

template<typename Dist, typename Fresnel>
BSDL_INLINE Sample
sample_spi_microms_reflection(const Dist& dist, const Fresnel& fresnel,
                              float E_ms, float E_ms_avg, const Imath::V3f& wo,
                              const Imath::V3f& rnd)
{
    const float cosNO = wo.z;
    if (cosNO <= 0)
        return {};

    Imath::V3f wi;
    if (rnd.z < E_ms) {
        // sample diffuse energy compensation lobe
        wi = sample_cos_hemisphere(rnd.x, rnd.y);
    } else {
        // sample microfacet (half vector)
        // generate outgoing direction
        wi = reflect(wo, dist.sample(wo, rnd.x, rnd.y));
        if (wi.z <= 0)
            return {};
    }
    // evaluate brdf on outgoing direction
    return eval_spi_microms_reflection(dist, fresnel, E_ms, E_ms_avg, wo, wi);
}

BSDL_LEAVE_NAMESPACE
