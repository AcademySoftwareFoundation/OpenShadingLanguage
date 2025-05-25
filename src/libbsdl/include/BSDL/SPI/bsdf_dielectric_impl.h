// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/SPI/bsdf_dielectric_decl.h>
#include <BSDL/microfacet_tools_decl.h>
#ifndef BAKE_BSDL_TABLES
#    include <BSDL/SPI/bsdf_dielectric_back_luts.h>
#    include <BSDL/SPI/bsdf_dielectric_front_luts.h>
#endif

BSDL_ENTER_NAMESPACE

namespace spi {

BSDL_INLINE_METHOD
DielectricFresnel::DielectricFresnel(float _eta, bool backside)
{
    if (backside)
        _eta = 1 / _eta;
    eta = _eta >= 1 ? CLAMP(_eta, IOR_MIN, IOR_MAX)
                    : CLAMP(_eta, 1 / IOR_MAX, 1 / IOR_MIN);
}

BSDL_INLINE_METHOD float
DielectricFresnel::eval(const float c) const
{
    assert(c >= 0);  // slightly above 1.0 is ok
    float g = (eta - 1.0f) * (eta + 1.0f) + c * c;
    if (g > 0) {
        g       = sqrtf(g);
        float A = (g - c) / (g + c);
        float B = (c * (g + c) - 1) / (c * (g - c) + 1);
        return 0.5f * A * A * (1 + B * B);
    }
    return 1.0f;  // TIR (no refracted component)
}

BSDL_INLINE_METHOD float
DielectricFresnel::avg() const
{
    return avg_fresnel_dielectric(eta);
}

BSDL_INLINE_METHOD DielectricFresnel
DielectricFresnel::from_table_index(float tx, int side)
{
    const float eta = LERP(SQR(tx), IOR_MIN, IOR_MAX);
    return DielectricFresnel(eta, side);
}

// Note index of eta equals index of 1 / eta, so this function works for
// either side (both tables)
BSDL_INLINE_METHOD float
DielectricFresnel::table_index() const
{
    // turn the IOR value into something suitable for integrating
    // this is the reverse of the method above
    const float feta = eta;
    const float seta = CLAMP(feta < 1 ? 1 / feta : feta, IOR_MIN, IOR_MAX);
    const float x    = (seta - IOR_MIN) * (1 / (IOR_MAX - IOR_MIN));
    assert(x >= 0);
    assert(x <= 1);
    return sqrtf(x);
}

template<typename Dist, int side>
BSDL_INLINE_METHOD
Dielectric<Dist, side>::Dielectric(float, float roughness_index,
                                   float fresnel_index)
    : d(roughness_index, 0)
    , f(DielectricFresnel::from_table_index(fresnel_index, side))
    , prob_clamp(0)
{
}

// Compute a sampling probability based on fresnel. Returning f would be ideal
// if were not going to find >1 energy after a bounce.
template<typename Dist, int side>
BSDL_INLINE_METHOD float
Dielectric<Dist, side>::fresnel_prob(float f) const
{
    const float safe_prob = 0.2f;
    return LERP(prob_clamp, f, CLAMP(f, safe_prob, 1 - safe_prob));
}

template<typename Dist, int side>
BSDL_INLINE_METHOD Sample
Dielectric<Dist, side>::eval(Imath::V3f wo, Imath::V3f wi, bool doreflect,
                             bool dorefract) const
{
    const float cosNO = wo.z;
    const float cosNI = wi.z;
    const bool both   = doreflect && dorefract;
    assert(cosNO >= 0);
    if (cosNI > 0) {
        const Imath::V3f m = (wo + wi).normalized();
        const float cosMO  = m.dot(wo);
        if (cosMO <= 0)
            return {};
        const float D   = d.D(m);
        const float G1  = d.G1(wo);
        const float F   = f.eval(cosMO);
        const float P   = both ? fresnel_prob(F) : 1;
        const float out = d.G2_G1(wi, wo) * (F / P);
        const float pdf = (G1 * D) / (4.0f * cosNO) * (both ? P : 1);
        return { wi, Power(out, 1), pdf, 0 };
    } else if (cosNI < 0) {
        // flip to same side as N
        const Imath::V3f Ht = (f.eta * wi + wo).normalized()
                              * ((f.eta > 1) ? -1 : 1);
        // compute fresnel term
        const float cosHO = Ht.dot(wo);
        const float cosHI = Ht.dot(wi);
        if (cosHO <= 0 || cosHI >= 0)
            return {};
        const float Ft = 1.0f - f.eval(cosHO);
        const float Pt = both ? fresnel_prob(Ft) : 1;
        if (Ht.z <= 0 || cosHO <= 0 || cosHI >= 0 || Ft <= 0)
            return {};
        const float D   = d.D(Ht);
        const float G1  = d.G1(wo);
        const float out = d.G2_G1({ wi.x, wi.y, -wi.z }, wo) * (Ft / Pt);

        float pdf = (-cosHI * cosHO * SQR(f.eta) * (G1 * D))
                    / (wo.z * SQR(cosHI * f.eta + cosHO)) * (both ? Pt : 1);
        return { wi, Power(out, 1), pdf, 0 };

    } else
        return {};
}

template<typename Dist, int side>
BSDL_INLINE_METHOD Sample
Dielectric<Dist, side>::sample(Imath::V3f wo, float randu, float randv,
                               float randw, bool doreflect,
                               bool dorefract) const
{
    const Imath::V3f m = d.sample(wo, randu, randv);
    const float cosMO  = wo.dot(m);
    const bool both    = doreflect && dorefract;
    if (cosMO <= 0)
        return {};
    const float F = both ? fresnel_prob(f.eval(cosMO)) : (dorefract ? 0 : 1);
    bool choose_reflect = randw < F;
    const Imath::V3f wi = choose_reflect ? reflect(wo, m)
                                         : refract(wo, m, f.eta);
    if ((choose_reflect && wi.z <= 0) || (!choose_reflect && wi.z >= 0))
        return {};
    return eval(wo, wi, doreflect, dorefract);
}

BSDL_INLINE_METHOD
DielectricFront::DielectricFront(float, float roughness_index,
                                 float fresnel_index)
    : Dielectric<GGXDist, 0>(0, roughness_index, fresnel_index)
{
}

BSDL_INLINE_METHOD
DielectricFront::DielectricFront(const GGXDist& dist,
                                 const DielectricFresnel& fresnel,
                                 float prob_clamp)
    : Dielectric<GGXDist, 0>(dist, fresnel, prob_clamp)
{
}

BSDL_INLINE_METHOD
DielectricBack::DielectricBack(float, float roughness_index,
                               float fresnel_index)
    : Dielectric<GGXDist, 1>(0, roughness_index, fresnel_index)
{
}

BSDL_INLINE_METHOD
DielectricBack::DielectricBack(const bsdl::GGXDist& dist,
                               const DielectricFresnel& fresnel,
                               float prob_clamp)
    : Dielectric<GGXDist, 1>(dist, fresnel, prob_clamp)
{
}

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
DielectricLobe<BSDF_ROOT>::DielectricLobe(T* lobe, const BsdfGlobals& globals,
                                          const Data& data)
    : Base(lobe, globals.visible_normal(data.N), data.T,
           globals.regularize_roughness(data.roughness), globals.lambda_0, true)
    , spec(GGXDist(globals.regularize_roughness(data.roughness),
                   CLAMP(data.anisotropy, 0.0f, 1.0f)),
           DielectricFresnel(LERP(CLAMP(data.force_eta, 0.0f, 1.0f),
                                  globals.relative_eta(data.IOR), data.IOR),
                             globals.backfacing),
           data.prob_clamp)
    , refl_tint(globals.wave(data.refl_tint))
    , refr_tint(globals.wave(data.refr_tint))
    , backside(spec.eta() < 1.0f)
{
    // Compiler should optimize all these calls to regularize_roughness
    const float roughness = globals.regularize_roughness(data.roughness);
    TabulatedEnergyCurve<DielectricFront> diff_curve_front(
        roughness, spec.fresnel().table_index());
    TabulatedEnergyCurve<DielectricBack> diff_curve_back(
        roughness, spec.fresnel().table_index());

    const float ratio_F = avg_fresnel_dielectric(spec.eta());
    const float ratio_B = avg_fresnel_dielectric(1 / spec.eta());

    const float cosNO = globals.wo.dot(Base::frame.Z);
    assert(cosNO >= 0);
    const float davg_front = backside ? diff_curve_back.get_Emiss_avg()
                                      : diff_curve_front.get_Emiss_avg();
    const float davg_back  = backside ? diff_curve_front.get_Emiss_avg()
                                      : diff_curve_back.get_Emiss_avg();
    Eo                     = backside ? diff_curve_back.Emiss_eval(cosNO)
                                      : diff_curve_front.Emiss_eval(cosNO);
    // now compute RT_ratio such that the reciprocity constraint is obeyed:
    //     (1 - ratio_F) / m_cdf_Q[0] === (1 - ratio_B) / m_cdf_Q[1] * eta^2
    // since the equality does not hold in general, we adjust by x on the left
    // and by (1-x) on the right until they do hold and recompute a new ratio_F

    // FIXME: figure out where the eta^2 factor should go after we straighten
    //        out the reverse PDF calculation
    const float L = (1 - ratio_F) / davg_back;
    const float R = (1 - ratio_B) / davg_front * SQR(spec.eta());
    const float x = R > 1e12f ? 1 : R / (L + R);
    RT_ratio      = 1 - x * (1 - ratio_F);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
DielectricLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo, const Imath::V3f& wi,
                                     bool doreflect, bool dorefract) const
{
    const float cosNI = wi.z;
    const bool both   = doreflect && dorefract;

    if ((cosNI > 0 && !doreflect) || (cosNI < 0 && !dorefract) || cosNI == 0)
        return {};
    Sample s       = { wi };
    const float PE = Eo * (both ? 1 : (dorefract ? 1 - RT_ratio : RT_ratio));
    Sample ss      = spec.eval(wo, wi, doreflect, dorefract);
    s.update(ss.weight, ss.pdf, 1 - PE);
    s = eval_ec_lobe(s);
    s.weight *= get_tint(cosNI);
    s.roughness = Base::roughness();
    return s;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
DielectricLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                       const Imath::V3f& _rnd, bool doreflect,
                                       bool dorefract) const
{
    const bool both = doreflect && dorefract;

    const float PE = Eo * (both ? 1 : (dorefract ? 1 - RT_ratio : RT_ratio));
    Sample s       = {};
    s.roughness    = Base::roughness();

    Imath::V3f rnd = _rnd;
    if (rnd.x < 1 - PE) {
        // sample specular lobe
        rnd.x   = Sample::stretch(rnd.x, 0.0f, 1 - PE);
        auto ss = spec.sample(wo, rnd.x, rnd.y, rnd.z, doreflect, dorefract);
        if (MAX_ABS_XYZ(ss.wi) < EPSILON)
            return {};
        s.wi = ss.wi;
        s.update(ss.weight, ss.pdf, 1 - PE);
        s = eval_ec_lobe(s);
        if (s.wi.z < 0)
            // From "Efficient Rendering of Layered Materials using an
            // Atomic Decomposition with Statistical Operators" (Belcour)
            // equation 10. Adjust roughness with the equivalent of a
            // reflection.
            s.roughness = std::min(
                1.0f, Base::roughness()
                          * SQR(0.5f * (1 + (wo.z / (s.wi.z * spec.eta())))));
    } else {
        // sample diffuse lobe
        rnd.x     = Sample::stretch(rnd.x, 1 - PE, PE);
        bool back = !(!dorefract || (both && rnd.z < RT_ratio));
        s.wi      = sample_ec_lobe(rnd.x, rnd.y, back);
        s         = eval_ec_lobe(s);
        auto ss   = spec.eval(wo, s.wi, doreflect, dorefract);
        s.update(ss.weight, ss.pdf, 1 - PE);
    }
    const float cosNI = s.wi.z;
    s.weight *= get_tint(cosNI);
    return s;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Power
DielectricLobe<BSDF_ROOT>::get_tint(float cosNI) const
{
    return cosNI > 0 ? refl_tint : refr_tint;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
DielectricLobe<BSDF_ROOT>::eval_ec_lobe(Sample s) const
{
    const float dpdf = (Eo * fabsf(s.wi.z)) * ONEOVERPI
                       * (s.wi.z > 0 ? RT_ratio : 1 - RT_ratio);
    s.update(Power::UNIT(), dpdf, 1);
    return s;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Imath::V3f
DielectricLobe<BSDF_ROOT>::sample_ec_lobe(float randu, float randv,
                                          bool back) const
{
    const Imath::V3f wi = sample_cos_hemisphere(randu, randv);
    return back ? Imath::V3f { wi.x, wi.y, -wi.z } : wi;
}

}  // namespace spi

BSDL_LEAVE_NAMESPACE
