// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/SPI/bsdf_diffuse_decl.h>

BSDL_ENTER_NAMESPACE

namespace spi {

template<typename BSDF_ROOT, bool TR>
template<typename T>
BSDL_INLINE_METHOD
DiffuseLobeGen<BSDF_ROOT, TR>::DiffuseLobeGen(
    T* lobe, const BsdfGlobals& globals,
    const DiffuseLobeGen<BSDF_ROOT, TR>::Data& data)
    : Base(lobe, globals.visible_normal(data.N), 1.0f, globals.lambda_0, TR)
{
    Base::sample_filter = globals.get_sample_filter(Base::frame.Z, !TR);
}

template<typename BSDF_ROOT, bool TR>
BSDL_INLINE_METHOD Sample
DiffuseLobeGen<BSDF_ROOT, TR>::eval_impl(const Imath::V3f& wo,
                                         const Imath::V3f& wi) const
{
    float IdotN = wi.z;
    // For normal diffuse both wi and wo have to be above N
    if ((!TR && IdotN <= 0) ||
        // otherwise just in different sides
        (TR && IdotN >= 0))
        return {};
    return { wi, Power::UNIT(), fabsf(IdotN) * ONEOVERPI, 1.0f };
}

template<typename BSDF_ROOT, bool TR>
BSDL_INLINE_METHOD Sample
DiffuseLobeGen<BSDF_ROOT, TR>::sample_impl(const Imath::V3f& wo,
                                           const Imath::V3f& rnd) const
{
    Imath::V3f wi = sample_cos_hemisphere(rnd.x, rnd.y);
    if (TR)
        wi.z = -wi.z;
    Sample s = eval_impl(wo, wi);
    return s;
}

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
BasicDiffuseLobe<BSDF_ROOT>::BasicDiffuseLobe(
    T* lobe, const BsdfGlobals& globals,
    const typename BasicDiffuseLobe<BSDF_ROOT>::Data& data)
    : Base(lobe, globals.visible_normal(data.N),
           0.84f,  // Legacy roughness
           globals.lambda_0, data.translucent > 0)
    , diff_rough(CLAMP(data.diffuse_roughness, 0.0f, 1.0f))
    , diff_trans(CLAMP(data.translucent, 0.0f, 1.0f))
    , diff_color(globals.wave(CLAMP(data.color, 0.0f, 1.0f)))
{
    Base::sample_filter = globals.get_sample_filter(Base::frame.Z,
                                                    data.translucent == 0);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
BasicDiffuseLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                       const Imath::V3f& wi) const
{
    assert(wo.z >= 0);

    const bool trans = diff_trans > 0;
    const bool front = wi.z > 0;

    Sample sample = { wi, Power::ZERO(), 0, Base::roughness() };
    if (front || trans) {
        const float cosNO = wo.z;
        const float cosNI = fabsf(wi.z);

        const float diff_scale = front ? 1 - diff_trans : diff_trans;
        // oren-nayar adjustment
        const float s     = wo.dot(wi) - cosNO * cosNI;
        const float stinv = s > 0 ? s / std::max(cosNO, cosNI) : s;
        const float ON    = (1 - 0.235f * diff_rough)
                         * MAX(1 + diff_rough * stinv, 0.0f);
        sample.weight = diff_color * ON;
        sample.pdf    = diff_scale * cosNI * ONEOVERPI;
    }
    return sample;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
BasicDiffuseLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                         const Imath::V3f& rnd) const
{
    const float cosNO = wo.z;
    if (cosNO <= 0)
        return {};

    const bool trans = rnd.x < diff_trans;
    const float x    = trans ? Sample::stretch(rnd.x, 0, diff_trans)
                             : Sample::stretch(rnd.x, diff_trans, 1 - diff_trans);
    Imath::V3f wi    = sample_cos_hemisphere(x, rnd.y);
    wi.z             = trans ? -wi.z : wi.z;  // Flip if transmissive
    // evaluate brdf on outgoing direction
    return eval_impl(wo, wi);
}

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
ChandrasekharLobe<BSDF_ROOT>::ChandrasekharLobe(T* lobe,
                                                const BsdfGlobals& globals,
                                                const Data& data)
    : Base(lobe, globals.visible_normal(data.N), 1.0f, globals.lambda_0, false)
    , m_a(globals.wave(data.albedo).clamped(0, 1))
{
    Base::sample_filter = globals.get_sample_filter(Base::frame.Z, true);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
ChandrasekharLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const
{
    /* When translucent we mirror wi to the other side of the normal and perform
         * a regular oren-nayar BSDF. */
    const float NL = wi.z;
    const float NV = wo.z;
    if (NL > 0 && NV > 0) {
        const Power HL  = H(NL);
        const Power HV  = H(NV);
        const float pdf = NL * ONEOVERPI;

        // A HITCHHIKER’S GUIDE TO MULTIPLE SCATTERING v0.1.3
        // 7.3.4 Emerging Distribution (BRDF) [Chandrasekhar 1960]
        // Equation 272
        const Power b = m_a * HL * HV * (0.25f / (NL + NV));

        return { wi, b, pdf, 1.0f };
    }
    return {};
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
ChandrasekharLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& rnd) const
{
    Imath::V3f wi = sample_cos_hemisphere(rnd.x, rnd.y);
    return eval_impl(wo, wi);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
ChandrasekharLobe<BSDF_ROOT>::H(float a, float u) const
{
    // this is an approximation accurate to within 1% for all albedos and inclinations
    // A HITCHHIKER’S GUIDE TO MULTIPLE SCATTERING v0.1.3
    // 8.5.3 H-function Approximations, Approximation 2 [Hapke 2012]
    const float y = sqrtf(1 - a);
    const float n = (1 - y) / (1 + y);
    return 1
           / (1
              - (1 - y) * u
                    * (n + (1 - n * 0.5f - n * u) * fast_log(1 / u + 1)));
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Power
ChandrasekharLobe<BSDF_ROOT>::H(float u) const
{
    return Power([&](int i) { return H(m_a[i], u); }, 1);
}

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
DiffuseTLobe<BSDF_ROOT>::DiffuseTLobe(T* lobe, const BsdfGlobals& globals,
                                      const Data& data)
    : Base(lobe, globals.visible_normal(data.N), 1.0f, globals.lambda_0, false)
    , m_c(globals.wave(data.albedo).clamped(0, 1))
{
    Base::sample_filter = globals.get_sample_filter(Base::frame.Z, true);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
DiffuseTLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                   const Imath::V3f& wi) const
{
    const float ui = wi.z;
    const float uo = wo.z;
    if (ui > 0 && uo > 0) {
        const float pdf = ui * ONEOVERPI;

        Power sqrt1minusc([&](int i) { return sqrtf(1.0f - m_c[i]); }, 1);

        const Power Hi = HFunctionGamma2(ui, sqrt1minusc);
        const Power Ho = HFunctionGamma2(uo, sqrt1minusc);

        const Power Hterm = Hi * Ho * (1.0f / (ui + uo));

        const float math1   = ui * ui + 3.0f * ui * uo + uo * uo;
        const float uiplus2 = (1.0f + ui) * (1.0f + ui);
        const float uoplus2 = (1.0f + uo) * (1.0f + uo);
        const float math2   = 1.0f / (uiplus2 * uoplus2);
        const float math3   = (ui * ui + 2.0f * uo + 3.0f * ui * uo)
                            * (uo * uo + 2.0f * ui + 3.0f * ui * uo);
        const float math4 = (ui * ui * ui + uo * uo * uo
                             + ui * uo
                                   * (2.0f * (1.0f + ui * ui + uo * uo)
                                      + 3.0f * (ui + uo) + 6.0f * ui * uo))
                            * ui * uo / (ui + uo);

        Power b
            = 0.25f * m_c * Hterm * Hterm
              * (Power(math1 / (ui + uo), 1)
                 - 0.5f * math2
                       * ((Power::UNIT() - sqrt1minusc) * math3 + m_c * math4));

        return { wi, b, pdf, 1.0f };
    }
    return {};
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
DiffuseTLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                     const Imath::V3f& rnd) const
{
    Imath::V3f wi = sample_cos_hemisphere(rnd.x, rnd.y);
    return eval_impl(wo, wi);
}

// Solve for single-scattering albedo c given diffuse reflectance kD under uniform hemisphere illumination
template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Power
DiffuseTLobe<BSDF_ROOT>::albedoInvert(const Power in_R)
{
    const Power white(1, 1);
    const Power n = in_R * 4;
    const Power d = (white + in_R) * (white + in_R);
    return n / d;
}

// H-function for Gamma-2 flights in 3D with isotropic scattering
template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
DiffuseTLobe<BSDF_ROOT>::HFunctionGamma2(const float u, const float sqrt1minusc)
{
    return (1.0f + u) / (1.0f + sqrt1minusc * u);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Power
DiffuseTLobe<BSDF_ROOT>::HFunctionGamma2(const float u, const Power sqrt1minusc)
{
    return Power([&](int i) { return HFunctionGamma2(u, sqrt1minusc[i]); }, 1);
}

}  // namespace spi

BSDL_LEAVE_NAMESPACE
