// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/SPI/bsdf_thinlayer_decl.h>
#ifndef BAKE_BSDL_TABLES
#    include <BSDL/SPI/bsdf_thinlayer_luts.h>
#endif
#include <BSDL/bsdf_impl.h>
#include <BSDL/microfacet_tools_impl.h>

BSDL_ENTER_NAMESPACE

namespace spi {

BSDL_INLINE_METHOD
ThinFresnel::ThinFresnel(float eta) : eta(CLAMP(eta, IOR_MIN, IOR_MAX)) {}

BSDL_INLINE_METHOD float
ThinFresnel::eval(const float c) const
{
    assert(c >= 0);   // slightly above 1.0 is ok
    assert(eta > 1);  // avoid singularity at eta==1
    // optimized for c in [0,1] and eta in (1,inf)
    const float g = sqrtf(eta * eta - 1 + c * c);
    const float A = (g - c) / (g + c);
    const float B = (c * (g + c) - 1) / (c * (g - c) + 1);
    return 0.5f * A * A * (1 + B * B);
}

BSDL_INLINE_METHOD float
ThinFresnel::eval_inv(const float c) const
{
    return fresnel_dielectric(c, 1 / eta);
}

BSDL_INLINE_METHOD float
ThinFresnel::avgf() const
{
    // see Avg Fresnel -- but we know that eta >= 1 here
    return (eta - 1) / (4.08567f + 1.00071f * eta);
}

BSDL_INLINE_METHOD Power
ThinFresnel::avg() const
{
    return Power(avgf(), 1);
}

BSDL_INLINE_METHOD float
ThinFresnel::avg_invf() const
{
    return avg_fresnel_dielectric(1 / eta);
}

BSDL_INLINE_METHOD ThinFresnel
ThinFresnel::from_table_index(float tx)
{
    const float eta = LERP(SQR(tx), IOR_MIN, IOR_MAX);
    return ThinFresnel(eta);
}

// Note index of eta equals index of 1 / eta, so this function works for
// either side (both tables)
BSDL_INLINE_METHOD float
ThinFresnel::table_index() const
{
    // turn the IOR value into something suitable for integrating
    // this is the reverse of the method above
    const float seta = CLAMP(eta < 1 ? 1 / eta : eta, IOR_MIN, IOR_MAX);
    const float x    = (seta - IOR_MIN) * (1 / (IOR_MAX - IOR_MIN));
    assert(x >= 0);
    assert(x <= 1);
    return sqrtf(x);
}

template<typename Dist>
BSDL_INLINE_METHOD float
ThinMicrofacet<Dist>::sum_refl_series(float Rout, float Tin, float Rin, float A)
{
    // The amount of reflected energy is
    //
    //  Rout + Tin (1 - Rin) Rin A^2 + Tin Tout Rin^3 A^4 ...
    //         Tin (1 - Rin) Rin^(2n-1) A^(2n) =
    //  Rout + Tin (1 - Rin) Rin A^2 / (1 - Rin^2 A^2)
    //
    // which without absorption adds up to 1 with the refracted component in
    // the next function.
    const float b = SQR(Rin * A);  // Basis of the geometric series
    return Rout
           + (1 - b < EPSILON ? (A < 1 ? 0 : Tin * 0.5f)
                              : Tin * (1 - Rin) * SQR(A) * Rin / (1 - b));
}

template<typename Dist>
BSDL_INLINE_METHOD float
ThinMicrofacet<Dist>::sum_refr_series(float Rout, float Tin, float Rin, float A)
{
    // The amount of energy exiting the thin layer on the other side is
    //
    //  Tin A (1 - Rin) (1 + Rin^2 A^2 + Rin^4 A^4 + ... Rin^(2n) A^(2n)) =
    //  Tin A (1 - Rin) / (1 - Rin^2 A^2)
    //
    // which we evaluate for each RGB channel guarding the singularity, when
    // (1 - Rin^2 A^2) approaches 0 the limit is Tin * 0.5 if A is 1.0 or
    // 0.0 if A < 1.0. Note A and Rin are always < 1.0.
    const float b = SQR(Rin * A);  // Basis of the geometric series
    return 1 - b < EPSILON ? (A < 1 ? 0 : Tin * 0.5f)
                           : Tin * (1 - Rin) * A / (1 - b);
}
template<typename Dist>
BSDL_INLINE_METHOD
ThinMicrofacet<Dist>::ThinMicrofacet(float, float roughness_index,
                                     float fresnel_index)
    : d(roughness_index, 0)
    , sigma_t(0.0f, 1)
    , f(ThinFresnel::from_table_index(fresnel_index))
    , thickness(0)
    , roughness(roughness_index)
    , prob_clamp(0)
{
}

template<typename Dist>
BSDL_INLINE_METHOD
ThinMicrofacet<Dist>::ThinMicrofacet(float roughness, float aniso, float eta,
                                     float thickness, float prob_clamp,
                                     Power sigma_t)
    : d(roughness, aniso)
    , sigma_t(sigma_t)
    , f(eta)
    , thickness(thickness)
    , roughness(roughness)
    , prob_clamp(prob_clamp)
{
}

template<typename Dist>
BSDL_INLINE_METHOD Sample
ThinMicrofacet<Dist>::eval(const Imath::V3f& wo, const Imath::V3f& wi,
                           bool doreflect, bool dorefract) const
{
    const bool both      = doreflect && dorefract;
    const bool isrefl    = wi.z > 0;
    const Imath::V3f wif = { wi.x, wi.y,
                             fabsf(wi.z) };  // Flipped reflection trick;
    // The micronormal we actually use for scattering, which may or may not
    // have its slopes scaled
    const Imath::V3f mt          = (wo + wif).normalized();
    const float refr_slope_scale = refraction_slope_scale(wo.z);
    // The micronormal that we pretend to use, but a lie with refraction
    const Imath::V3f m = isrefl ? mt : scale_slopes(mt, 1 / refr_slope_scale);
    if (wi.z == 0 || wo.z == 0 || m.dot(wo) <= 0 || mt.dot(wo) <= 0)
        return {};
    Power refl_atten, refr_atten;
    attenuation(wo, m, &refl_atten, &refr_atten);
    // Call the common eval
    return eval(wo, m, wi, both, refr_slope_scale, refl_atten, refr_atten);
}

template<typename Dist>
BSDL_INLINE_METHOD Sample
ThinMicrofacet<Dist>::sample(Imath::V3f wo, float randu, float randv,
                             float randw, bool doreflect, bool dorefract) const
{
    const bool both    = doreflect && dorefract;
    const Imath::V3f m = d.sample(wo, randu, randv);
    if (wo.dot(m) <= 0)
        return {};
    Power refl_atten, refr_atten;
    attenuation(wo, m, &refl_atten, &refr_atten);
    // We compute a pseudo-fresnel factor from the attenuations, it will serve
    // as a probability for choosing the lobe
    const float R = refl_atten.max(), T = refr_atten.max();
    const float F = R / std::max(R + T, std::numeric_limits<float>::min());
    const float P = both ? fresnel_prob(F) : (dorefract ? 0 : 1);
    bool isrefl   = randw < P;
    const float refr_slope_scale = refraction_slope_scale(wo.z);
    // If we are doing refraction fake a micronormal by scaling the slopes
    // to account for the reduced blurring in refraction
    const Imath::V3f mt = isrefl ? m : scale_slopes(m, refr_slope_scale);
    if (wo.dot(mt) <= 0)
        return {};
    const Imath::V3f wif = reflect(wo, mt);
    // Flipped reflection trick for refraction
    const Imath::V3f wi = isrefl ? wif : Imath::V3f(wif.x, wif.y, -wif.z);
    if ((isrefl && wi.z <= 0) || (!isrefl && wi.z >= 0))
        return {};
    // Call the common eval
    return eval(wo, m, wi, both, refr_slope_scale, refl_atten, refr_atten);
}

// Borrowed from dielectric
template<typename Dist>
BSDL_INLINE_METHOD float
ThinMicrofacet<Dist>::fresnel_prob(float f) const
{
    const float safe_prob = 0.2f;
    return LERP(prob_clamp, f, CLAMP(f, safe_prob, 1 - safe_prob));
}

template<typename Dist>
BSDL_INLINE_METHOD Sample
ThinMicrofacet<Dist>::eval(const Imath::V3f& wo, const Imath::V3f& m,
                           const Imath::V3f& wi, const bool both,
                           const float refr_slope_scale, Power refl_atten,
                           Power refr_atten) const
{
    const bool isrefl = wi.z > 0;
    const float R = refl_atten.max(), T = refr_atten.max();
    const float F = R / std::max(R + T, std::numeric_limits<float>::min());
    const float P = both ? fresnel_prob(isrefl ? F : 1 - F) : 1;
    if (P < PDF_MIN)
        return {};
    // Flipped reflection trick in case of refraction
    const Imath::V3f wif = { wi.x, wi.y, fabsf(wi.z) };
    const float cosNO    = wo.z;
    const float cosNM    = m.z;
    const float sinNM    = sqrtf(1 - std::min(SQR(cosNM), 1.0f));
    assert(cosNO >= 0);
    // If we did the slope scaling there is a jacobian to account for
    // cos^3 to go to slope space, the scale^2 and then 1 / cos^3 of the new
    // angle. It boils down to this:
    const float tmp = sqrtf(SQR(refr_slope_scale * sinNM) + SQR(cosNM));
    const float J   = isrefl ? 1 : tmp * tmp * tmp / SQR(refr_slope_scale);
    const float D   = d.D(m);
    const float G1  = d.G1(wo);
    const float out = d.G2_G1(wif, wo) / P;
    const float pdf = (G1 * D * J * P) / (4.0f * cosNO);
    const Power w   = (isrefl ? refl_atten : refr_atten) * out;
    // P is computed so the weights never go over one, but due to rounding
    // errors and mostly fresnel_prob(), weight can go over one.
    assert(w.max() < 2.5f);
    return { wi, w, pdf, 0 };
}

// This function computes the slope scale needed for a micronormal if we
// are going to fake the refraction with a flipped reflection
template<typename Dist>
BSDL_INLINE_METHOD float
ThinMicrofacet<Dist>::refraction_slope_scale(float cosNO) const
{
    cosNO = std::min(cosNO, 1.0f);
    assert(cosNO >= 0);
    const float inveta = 1 / eta();
    const float sinNI  = inveta * sqrtf(1 - SQR(cosNO));
    if (sinNI > 1.0f)
        return 1;
    const float cosNI = -sqrtf(1 - SQR(sinNI));
    // From "Efficient Rendering of Layered Materials using an
    // Atomic Decomposition with Statistical Operators" (Belcour)
    // equation 10. This is the jacobian from half vector slope to
    // refracted slope.
    const float refr_jacobian_entry = (1 + inveta * (cosNO / cosNI));
    const float refr_jacobian_exit  = (1 + eta() * (cosNI / cosNO));
    // And the trivial reflection one
    const float refl_jacobian = 2;
    // Without loss of generality, we assume roughness is 1.0 here.
    // Take into account there is entry and exit events, so we have a
    // convolution of two GGX. We know its pseudo variance is 2 alpha^2
    // but in this case using just alpha^2 works better. Add the two variances:
    const float refr_variance = SQR(refr_jacobian_entry)
                                + SQR(refr_jacobian_exit);
    // which we will divide by the reflection variance, the one actually used
    const float refl_variance = SQR(refl_jacobian);
    // Never scale the slope so it equates to a roughness higher than one
    const float max_scale = 1 / roughness;
    // And finally map from variance to slope the ratio of the two. There
    // is a consistent visual effect that scales up the effective variance
    // by the inverse of cosNO. I attribute it to the stretching of
    // the projected surface but I still ignore the full explanation.
    return std::min(sqrtf((refr_variance / refl_variance) / cosNO), max_scale);
}

template<typename Dist>
BSDL_INLINE_METHOD Imath::V3f
ThinMicrofacet<Dist>::scale_slopes(const Imath::V3f& m, const float s) const
{
    return Imath::V3f(m.x * s, m.y * s, m.z).normalized();
}

template<typename Dist>
BSDL_INLINE_METHOD void
ThinMicrofacet<Dist>::attenuation(Imath::V3f wo, Imath::V3f m, Power* refl,
                                  Power* refr) const
{
    const Imath::V3f wr = refract(wo, m, eta());
    const float cosNO   = std::min(wo.dot(m), 1.0f);
    const float cosNR   = CLAMP(-wr.z, 0.0f, 1.0f);
    assert(cosNO >= 0);
    // Traveled distance inside the thin layer depends on the
    // refracted angle. As roughness grows, the inverse cosine
    // tends to the avg
    const float d = thickness
                    * LERP(roughness, 1 / std::max(cosNR, 1e-6f), AVG_INV_COS);
    // Fresnel coeficients
    const float Rout = f.eval(cosNO);
    const float Tin  = 1.0f - Rout;
    // And as roughness grows, the internal bounces tend to the avg too
    const float Rin = LERP(roughness, f.eval_inv(cosNR), f.avg_invf());
    // Attenuation for one segment or bounce inside the layer
    Power A = Power::UNIT();

    constexpr auto fast_exp = BSDLConfig::Fast::expf;
    if (d > 0) {
        A = Power(
            [&](int i) {
                return sigma_t[i] > 0 ? fast_exp(-d * sigma_t[i]) : 1;
            },
            1);
    }

    *refl = Power([&](int i) { return sum_refl_series(Rout, Tin, Rin, A[i]); },
                  1);
    *refr = Power([&](int i) { return sum_refr_series(Rout, Tin, Rin, A[i]); },
                  1);
    assert((*refl + *refr).max() < 1.0001f);
}

BSDL_INLINE_METHOD
Thinlayer::Thinlayer(float, float roughness_index, float fresnel_index)
    : ThinMicrofacet<bsdl::GGXDist>(0, roughness_index, fresnel_index)
{
}

BSDL_INLINE_METHOD
Thinlayer::Thinlayer(float roughness, float aniso, float eta, float thickness,
                     float prob_clamp, Power sigma_t)
    : ThinMicrofacet<GGXDist>(roughness, aniso, eta, thickness, prob_clamp,
                              sigma_t)
{
}

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
ThinLayerLobe<BSDF_ROOT>::ThinLayerLobe(T* lobe, const BsdfGlobals& globals,
                                        const Data& data)
    :

    Base(lobe, globals.visible_normal(data.N), data.T,
         globals.regularize_roughness(CLAMP(data.roughness, 0.0f, 1.0f)),
         globals.lambda_0, true)
    , spec(Base::roughness(), data.anisotropy, data.IOR, data.thickness,
           data.prob_clamp, globals.wave(data.sigma_t))
    ,
    // We also scale refl_tint to avoid secondary rays bouncing inside, which is
    // bad when we have transparent shadows and produces energy explosion
    refl_tint(globals.wave(data.refl_tint).clamped(0, 1))
    , refr_tint(globals.wave(data.refr_tint).clamped(0, 1))
{
    TabulatedEnergyCurve<Thinlayer> diff_curve(Base::roughness(),
                                               spec.fresnel().table_index());

    float cosNO = CLAMP(globals.wo.dot(Base::frame.Z), 0.0f, 1.0f);
    Eo          = diff_curve.Emiss_eval(cosNO);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
ThinLayerLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo, const Imath::V3f& wi,
                                    bool doreflect, bool dorefract) const
{
    const bool both       = doreflect && dorefract;
    const float cosNI     = wi.z;
    const auto diff_trans = get_diff_trans();
    const float R = diff_trans.first.max(), T = diff_trans.second.max();
    const float Tprob = T / std::max(R + T, std::numeric_limits<float>::min());

    assert(wo.z >= 0);
    if ((cosNI > 0 && !doreflect) || (cosNI < 0 && !dorefract) || cosNI == 0)
        return {};
    Sample s       = { wi };
    const float PE = Eo * (both ? 1 : (dorefract ? Tprob : 1 - Tprob));
    Sample ss      = spec.eval(wo, wi, doreflect, dorefract);
    s.update(ss.weight, ss.pdf, 1 - PE);
    eval_ec_lobe(&s, wi, diff_trans.first, diff_trans.second, Tprob);
    s.weight *= get_tint(cosNI);
    s.roughness = sum_max(Base::roughness(), cosNI < 0 ? Base::roughness() : 0,
                          1.0f);
    return s;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
ThinLayerLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                      const Imath::V3f& _rnd, bool doreflect,
                                      bool dorefract) const
{
    const bool both       = doreflect && dorefract;
    const auto diff_trans = get_diff_trans();
    const float R = diff_trans.first.max(), T = diff_trans.second.max();
    const float Tprob = T / std::max(R + T, std::numeric_limits<float>::min());
    const float PE    = Eo * (both ? 1 : (dorefract ? Tprob : 1 - Tprob));
    Sample s          = {};

    Imath::V3f rnd = _rnd;
    if (rnd.x < 1 - PE) {
        // sample specular lobe
        rnd.x   = Sample::stretch(rnd.x, 0.0f, 1 - PE);
        auto ss = spec.sample(wo, rnd.x, rnd.y, rnd.z, doreflect, dorefract);
        if (MAX_ABS_XYZ(ss.wi) == 0)
            return {};
        s.wi = ss.wi;
        s.update(ss.weight, ss.pdf, 1 - PE);
        eval_ec_lobe(&s, s.wi, diff_trans.first, diff_trans.second, Tprob);
    } else {
        // sample diffuse lobe
        rnd.x           = Sample::stretch(rnd.x, 1 - PE, PE);
        const bool back = !(!dorefract || (both && rnd.z >= Tprob));
        s.wi            = sample_lobe(rnd.x, rnd.y, back);
        eval_ec_lobe(&s, s.wi, diff_trans.first, diff_trans.second, Tprob);
        const auto ss = spec.eval(wo, s.wi, doreflect, dorefract);
        s.update(ss.weight, ss.pdf, 1 - PE);
    }
    const float cosNI = s.wi.z;
    s.weight *= get_tint(cosNI);
    s.roughness = sum_max(Base::roughness(), cosNI < 0 ? Base::roughness() : 0,
                          1.0f);
    return s;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD std::pair<Power, Power>
ThinLayerLobe<BSDF_ROOT>::get_diff_trans() const
{
    constexpr auto fast_exp = BSDLConfig::Fast::expf;
    // Start with average fresnel
    const float Tf = 1 - spec.fresnel().avgf(),
                Tb = 1 - spec.fresnel().avg_invf();
    // And then correct to satisfy the refraction reciprocity
    const float x    = Tb * SQR(spec.eta()) / (Tf + Tb * SQR(spec.eta()));
    const float Tout = Tb * (1 - x), Tin = Tf * x,
                // And therefore reflection is ...
        Rout = 1 - Tin, Rin = 1 - Tout;

    // Diffuse and Translucent lobes for energy compensation
    const float avgd = spec.thickness * Thinlayer::AVG_INV_COS;

    Power A([&](int i) { return fast_exp(-avgd * spec.sigma_t[i]); }, 1);
    // Sum up series for the tints
    Power diff_tint(
        [&](int i) { return Thinlayer::sum_refl_series(Rout, Tin, Rin, A[i]); },
        1);
    Power trans_tint(
        [&](int i) { return Thinlayer::sum_refr_series(Rout, Tin, Rin, A[i]); },
        1);
    return { diff_tint, trans_tint };
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Power
ThinLayerLobe<BSDF_ROOT>::get_tint(float cosNI) const
{
    return cosNI > 0 ? refl_tint : refr_tint;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD void
ThinLayerLobe<BSDF_ROOT>::eval_ec_lobe(Sample* s, Imath::V3f wi_l,
                                       const Power diff_tint,
                                       const Power trans_tint,
                                       float Tprob) const
{
    const bool back       = wi_l.z <= 0;
    const float side_prob = back ? Tprob : 1 - Tprob;
    if (side_prob == 0)
        return;
    const float dpdf = Eo * fabsf(wi_l.z) * ONEOVERPI;
    const Power w    = back ? trans_tint : diff_tint;
    s->update(w, dpdf, side_prob);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Imath::V3f
ThinLayerLobe<BSDF_ROOT>::sample_lobe(float randu, float randv, bool back) const
{
    Imath::V3f wi = sample_cos_hemisphere(randu, randv);
    return back ? Imath::V3f(wi.x, wi.y, -wi.z) : wi;
}

}  // namespace spi

BSDL_LEAVE_NAMESPACE
