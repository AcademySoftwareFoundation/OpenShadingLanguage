// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once
#include <BSDL/jakobhanika_impl.h>
#include <BSDL/spectrum_decl.h>
#include <BSDL/spectrum_luts.h>

BSDL_ENTER_NAMESPACE

template<typename T>
BSDL_INLINE_METHOD T
Spectrum::lookup(float lambda, const T array[LAMBDA_RES])
{
    const float lambda_wrap = fmodf(std::max(0.0f, lambda - LAMBDA_MIN),
                                    LAMBDA_RANGE);
    const int i             = roundf(lambda_wrap / LAMBDA_STEP);
    return array[i];
    const float lambda_fidx = fmodf(std::max(0.0f, lambda - LAMBDA_MIN),
                                    LAMBDA_RANGE)
                              / LAMBDA_STEP;
    const int lo    = std::min(int(lambda_fidx), LAMBDA_RES - 2);
    const int hi    = lo + 1;
    const float mix = lambda_fidx - lo;
    return LERP(mix, array[lo], array[hi]);
}

template<int I>
BSDL_INLINE_METHOD Imath::C3f
Spectrum::spec_to_xyz(Power wave, float lambda_0)
{
    constexpr float inv_norm
        = 1
          / integrate_illuminant(I == 60 ? get_luts_ctxr().D60_illuminant
                                         : get_luts_ctxr().D65_illuminant,
                                 get_luts_ctxr().xyz_response);
    const Data& data        = get_luts();
    const float* illuminant = I == 60 ? data.D60_illuminant
                                      : data.D65_illuminant;

    Imath::C3f total(0);
    BSDL_UNROLL()
    for (int i = 0; i != Power::N; ++i) {
        const float lambda     = lambda_0 + i * Power::HERO_STEP;
        const Imath::C3f power = lookup(lambda, data.xyz_response)
                                 * wave.data[i];
        total += power * (LAMBDA_RANGE * inv_norm / Power::N)
                 * lookup(lambda, illuminant);
    }
    return total;
}

// Modify basic IOR to get the dispersion IOR.
BSDL_INLINE_METHOD
float
Spectrum::get_dispersion_ior(const float dispersion, const float basic_ior,
                             const float wavelength)
{
    // Fraunhofer D, F and C spectral lines, in micrometers squared
    const float nD2 = SQR(589.3f * 1e-3f);
    const float nF2 = SQR(486.1f * 1e-3f);
    const float nC2 = SQR(656.3f * 1e-3f);

    // convert Abbe number to Cauchy coefficients
    const float abbe    = 1 / dispersion;
    const float cauchyC = ((basic_ior - 1.0f) / abbe)
                          * ((nC2 * nF2) / (nC2 - nF2));
    const float cauchyB = basic_ior - cauchyC * (1.0f / nD2);

    // Cauchy's equation with two terms
    // wavelength converted from nanometers to micrometers
    return cauchyB + cauchyC / SQR(wavelength * 1e-3f);
}

BSDL_INLINE_METHOD
Power::Power(const Imath::C3f rgb, float lambda_0)
{
    ColorSpace cs = Spectrum::get_color_space(lambda_0);
    *this         = cs.upsample(rgb, lambda_0);
}

BSDL_INLINE_METHOD Imath::C3f
Power::toRGB(float lambda_0) const
{
    ColorSpace cs = Spectrum::get_color_space(lambda_0);
    return cs.downsample(*this, lambda_0);
}

BSDL_INLINE_METHOD Power
Power::resample(float from, float to) const
{
    if (from == to)
        return *this;
    Imath::C3f rgb = toRGB(from);
    return Power(rgb, to);
}

BSDL_INLINE_METHOD Power
sRGBColorSpace::upsample_impl(const Imath::C3f rgb, float lambda_0) const
{
    assert(lambda_0 != 0);
    const auto& basis = get_luts().rgb_basis_sRGB;
    Power w;
    BSDL_UNROLL()
    for (int i = 0; i != Power::N; ++i) {
        const float lambda     = lambda_0 + i * Power::HERO_STEP;
        const Imath::C3f power = Spectrum::lookup(lambda, basis);
        w.data[i] = power.x * rgb.x + power.y * rgb.y + power.z * rgb.z;
    }
    return w;
}

BSDL_INLINE_METHOD Imath::C3f
sRGBColorSpace::downsample_impl(const Power wave, float lambda_0) const
{
    assert(lambda_0 != 0);
    Imath::C3f total = Spectrum::spec_to_xyz<65>(wave, lambda_0);
    Imath::V3f cv    = { total.x, total.y, total.z };
    // This is BT709 aka sRGB
    Imath::V3f XYZ_to_RGB[3] = { { 3.2405f, -1.5371f, -0.4985f },
                                 { -0.9693f, 1.8760f, 0.0416f },
                                 { 0.0556f, -0.2040f, 1.0572f } };

    return { std::max(0.0f, XYZ_to_RGB[0].dot(cv)),
             std::max(0.0f, XYZ_to_RGB[1].dot(cv)),
             std::max(0.0f, XYZ_to_RGB[2].dot(cv)) };
}

BSDL_INLINE_METHOD Power
ACEScgColorSpace::upsample_impl(const Imath::C3f _rgb, float lambda_0) const
{
    assert(lambda_0 != 0);
    const float maxrgb = std::max(_rgb.x, std::max(_rgb.y, _rgb.z));
    const float minrgb = std::min(_rgb.x, std::min(_rgb.y, _rgb.z));
    assert(minrgb >= 0);
    if (maxrgb - minrgb < EPSILON)
        return Power(maxrgb, lambda_0);
    // Over this intensity, very saturated colors start losing saturation
    // with the table for ACEScg. We scale down and boost the spectrum.
    constexpr float safe = 0.7f;
    const float scale    = std::max(maxrgb, safe);
    Imath::C3f rgb       = _rgb * (safe / scale);
    auto curve           = JakobHanikaUpsampler(BSDLConfig::get_jakobhanika_lut(
                                          BSDLConfig::ColorSpaceTag::ACEScg))
                     .lookup(rgb.x, rgb.y, rgb.z);
    Power w;
    BSDL_UNROLL()
    for (int i = 0; i != Power::N; ++i) {
        const float lambda = Spectrum::wrap(lambda_0 + i * Power::HERO_STEP);
        w.data[i]          = curve(lambda) * scale * (1 / safe);
    }
    return w;
}

BSDL_INLINE_METHOD Imath::C3f
ACEScgColorSpace::downsample_impl(const Power wave, float lambda_0) const
{
    assert(lambda_0 != 0);
    constexpr float JH_range_correction = 1;  //400.0f / 470.0f;
    Imath::C3f total = Spectrum::spec_to_xyz<60>(wave, lambda_0)
                       * JH_range_correction;
    Imath::V3f cv = { total.x, total.y, total.z };
    // This is ACEScg
    Imath::V3f XYZ_to_RGB[3] = { { 1.641023f, -0.324803f, -0.236425f },
                                 { -0.663663f, 1.615332f, 0.016756f },
                                 { 0.011722f, -0.008284f, 0.988395f } };
    return { std::max(0.0f, XYZ_to_RGB[0].dot(cv)),
             std::max(0.0f, XYZ_to_RGB[1].dot(cv)),
             std::max(0.0f, XYZ_to_RGB[2].dot(cv)) };
}

BSDL_INLINE_METHOD Power
BypassColorSpace::upsample_impl(const Imath::C3f rgb, float lambda_0) const
{
    assert(lambda_0 == 0);
    Power w;
    w.data[0] = rgb.x;
    w.data[1] = rgb.y;
    w.data[2] = rgb.z;
    BSDL_UNROLL()
    for (int i = 3; i != Power::N; ++i)
        w.data[i] = 0;
    return w;
}

BSDL_INLINE_METHOD Imath::C3f
BypassColorSpace::downsample_impl(const Power wave, float lambda_0) const
{
    assert(lambda_0 == 0);
    return { wave.data[0], wave.data[1], wave.data[2] };
}

BSDL_INLINE_METHOD Power
ColorSpace::upsample(const Imath::C3f rgb, float lambda_0) const
{
    return dispatch([&](auto& cs) { return cs.upsample_impl(rgb, lambda_0); });
}

BSDL_INLINE_METHOD Imath::C3f
ColorSpace::downsample(const Power wave, float lambda_0) const
{
    return dispatch(
        [&](auto& cs) { return cs.downsample_impl(wave, lambda_0); });
}

BSDL_LEAVE_NAMESPACE
