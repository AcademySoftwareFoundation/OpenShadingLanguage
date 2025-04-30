// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/SPI/bsdf_sheenltc_decl.h>
#include <BSDL/SPI/bsdf_sheenltc_param.h>
#ifndef BAKE_BSDL_TABLES
#    include <BSDL/SPI/bsdf_sheenltc_luts.h>
#endif

BSDL_ENTER_NAMESPACE

namespace spi {

BSDL_INLINE_METHOD Sample
SheenLTC::eval(Imath::V3f wo, Imath::V3f wi) const
{
    assert(wo.z >= 0);
    assert(wi.z >= 0);

    // Rotate coordinate frame to align with incident direction wo.
    float phiStd     = calculate_phi(wo);
    Imath::V3f wiStd = rotate(wi, { 0, 0, 1 }, -phiStd);

    // Evaluate the LTC distribution in aligned coordinates.
    Imath::V3f ltcCoeffs = fetchCoeffs(wo);
    float pdf            = evalLTC(wiStd, ltcCoeffs);
    float R              = 1.333814f * ltcCoeffs.z;  // reflectance
    Power col            = Power(R, 1);

    return { wi, col, pdf, roughness };
}

BSDL_INLINE_METHOD Sample
SheenLTC::sample(Imath::V3f wo, float randu, float randv, float randw) const
{
    // Sample from the LTC distribution in aligned coordinates.
    Imath::V3f wiStd = sampleLTC(fetchCoeffs(wo), randu, randv);

    // Rotate coordinate frame based on incident direction wo.
    float phiStd  = calculate_phi(wo);
    Imath::V3f wi = rotate(wiStd, { 0, 0, 1 }, +phiStd);

    if (!same_hemisphere(wo, wi))
        return {};

    return eval(wo, wi);
}

BSDL_INLINE_METHOD float
SheenLTC::calculate_phi(const Imath::V3f& v) const
{
    float p = BSDLConfig::Fast::atan2f(v.y, v.x);
    if (p < 0) {
        p += 2 * PI;
    }
    return p;
}

BSDL_INLINE_METHOD bool
SheenLTC::same_hemisphere(const Imath::V3f& wo, const Imath::V3f& wi) const
{
    return wo.z * wi.z > 0;
}

BSDL_INLINE_METHOD void
SheenLTC::albedo_range(float& min_albedo, float& max_albedo) const
{
    const V32_array param = param_ptr();
    min_albedo = max_albedo = 0;
    for (int i = 0; i < ltcRes; ++i)
        for (int j = 0; j < ltcRes; ++j) {
            Imath::V3f v = param[i][j];
            float R      = v.z;  // reflectance
            if (i == 0 && j == 0) {
                min_albedo = max_albedo = R;
            } else {
                min_albedo = std::min(R, min_albedo);
                max_albedo = std::max(R, max_albedo);
            }
        }
}

BSDL_INLINE_METHOD void
SheenLTC::compute_scale(float& scale) const
{
    float min_albedo = 0, max_albedo = 0;
    albedo_range(min_albedo, max_albedo);

    if (max_albedo == 0) {
        scale = 0;
    } else {
        scale = 1.0f / max_albedo;
    }
}

// The following functions, and the data arrays ltcParamTableVolume and ltcParamTableApprox,
//     are translated from the code repository https://github.com/tizian/ltc-sheen, and the
//     paper "Practical Multiple-Scattering Sheen Using Linearly Transformed Cosines", by Tizian
//     Zeltner, Brent Burley, and Matt Jen-Yuan Chiang.

// Fetch the LTC coefficients by bilinearly interpolating entries in a 32x32
// lookup table. */
BSDL_INLINE_METHOD Imath::V3f
SheenLTC::fetchCoeffs(const Imath::V3f& wo) const
{
    // Compute table indices and interpolation factors.
    float row = CLAMP(roughness, 0.0f, ALMOSTONE) * (ltcRes - 1);
    float col = CLAMP(wo.z, 0.0f, ALMOSTONE) * (ltcRes - 1);
    float r   = std::floor(row);
    float c   = std::floor(col);
    float rf  = row - r;
    float cf  = col - c;
    int ri    = (int)r;
    int ci    = (int)c;

    const V32_array param = param_ptr();
    // Bilinear interpolation
    Imath::V3f coeffs;
    const Imath::V3f v1 = param[ri][ci];
    const Imath::V3f v2 = param[ri][ci + 1];
    const Imath::V3f v3 = param[ri + 1][ci];
    const Imath::V3f v4 = param[ri + 1][ci + 1];
    coeffs              = LERP(rf, LERP(cf, v1, v2), LERP(cf, v3, v4));
    return coeffs;
}

// Evaluate the LTC distribution in its local coordinate system.
BSDL_INLINE_METHOD float
SheenLTC::evalLTC(const Imath::V3f& wi, const Imath::V3f& ltcCoeffs) const
{
    // The (inverse) transform matrix `M^{-1}` is given by:
    //              [[aInv 0    bInv]
    //     M^{-1} =  [0    aInv 0   ]
    //               [0    0    1   ]]
    // with `aInv = ltcCoeffs[0]`, `bInv = ltcCoeffs[1]` fetched from the
    // table. The transformed direction `wiOriginal` is therefore:
    //                                [[aInv * wi.x + bInv * wi.z]
    //     wiOriginal = M^{-1} * wi =  [aInv * wi.y              ]
    //                                 [wi.z                     ]]
    // which is subsequently normalized. The determinant of the matrix is
    //     |M^{-1}| = aInv * aInv
    // which is used to compute the Jacobian determinant of the complete
    // mapping including the normalization.
    // See the original paper [Heitz et al. 2016] for details about the LTC
    // itself.
    float aInv = ltcCoeffs.x, bInv = ltcCoeffs.y;
    Imath::V3f wiOriginal = { aInv * wi.x + bInv * wi.z, aInv * wi.y, wi.z };
    const float length    = wiOriginal.length();
    wiOriginal *= 1.0f / length;
    float det      = aInv * aInv;
    float jacobian = det / (length * length * length);

    return wiOriginal.z * ONEOVERPI * jacobian;
}

// Sample from the LTC distribution in its local coordinate system.
BSDL_INLINE_METHOD Imath::V3f
SheenLTC::sampleLTC(const Imath::V3f& ltcCoeffs, float randu, float randv) const
{
    // The (inverse) transform matrix `M^{-1}` is given by:
    //              [[aInv 0    bInv]
    //     M^{-1} =  [0    aInv 0   ]
    //               [0    0    1   ]]
    // with `aInv = ltcCoeffs[0]`, `bInv = ltcCoeffs[1]` fetched from the
    // table. The non-inverted matrix `M` is therefore:
    //         [[1/aInv 0      -bInv/aInv]
    //     M =  [0      1/aInv  0        ]
    //          [0      0       1        ]]
    // and the transformed direction wi is:
    //                           [[wiOriginal.x/aInv - wiOriginal.z*bInv/aInv]
    //     wi = M * wiOriginal =  [wiOriginal.y/aInv                         ]
    //                            [wiOriginal.z                              ]]
    // which is subsequently normalized.
    // See the original paper [Heitz et al. 2016] for details about the LTC
    // itself.
    Imath::V3f wiOriginal = sample_uniform_hemisphere(randu, randv);

    float aInv = ltcCoeffs.x, bInv = ltcCoeffs.y;
    Imath::V3f wi = { wiOriginal.x / aInv - wiOriginal.z * bInv / aInv,
                      wiOriginal.y / aInv, wiOriginal.z };
    return wi.normalized();
}

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
SheenLTCLobe<BSDF_ROOT>::SheenLTCLobe(T* lobe, const BsdfGlobals& globals,
                                      const Data& data)
    : Base(lobe, globals.visible_normal(data.N), data.roughness,
           globals.lambda_0, false)
    , sheenLTC(CLAMP(globals.regularize_roughness(data.roughness), 0.0f, 1.0f))
    , tint(globals.wave(data.tint))
    , back(data.doublesided ? false : globals.backfacing)
{
    Base::sample_filter = globals.get_sample_filter(Base::frame.Z, true);
    TabulatedEnergyCurve<SheenLTC> curve(sheenLTC.get_roughness(), 0);
    // Get energy compensation taking tint into account
    Eo = back ? 1
              : 1
                    - std::min(
                        (1 - curve.Emiss_eval(Base::frame.Z.dot(globals.wo)))
                            * tint.max(),
                        1.0f);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
SheenLTCLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                   const Imath::V3f& wi) const
{
    const float cosNO = wo.z;
    const float cosNI = wi.z;
    const bool isrefl = cosNI > 0 && cosNO >= 0;
    const bool doself = isrefl && !back;

    Sample s = {};
    if (doself) {
        s = sheenLTC.eval(wo, wi);  // Return a grayscale sheen.
        s.weight *= tint;
    }
    return s;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
SheenLTCLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                     const Imath::V3f& sample) const
{
    const bool doself = !back;

    if (!doself)
        return {};

    Sample s = sheenLTC.sample(wo, sample.x, sample.y, sample.z);
    s.weight *= tint;
    return s;
}

}  // namespace spi

BSDL_LEAVE_NAMESPACE
