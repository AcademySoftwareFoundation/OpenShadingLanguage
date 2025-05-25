// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/thinfilm_decl.h>

BSDL_ENTER_NAMESPACE

// We're computing the energy of interference pattern between the directly reflected wave
// and the phase-shifted refracted-reflected wave:
//
// w/pi\int_0^{2*pi/w}((r*sin(wi)+(1-r)*sin(w+s))^2)
//
// where
// w = frequency
// s = phase-shift
// r = amount of light reflected from the upper interface
//
// We're currently making a some assumptions - all light refracted through the upper interface
// of the thin film is reflected by the lower interface. This is wrong, but visually it
// just amounts to a saturation control.
BSDL_INLINE_METHOD float
ThinFilm::interferenceEnergy(float r, float s)
{
    float c = BSDLConfig::Fast::cosf(s);
    return 1 + 2 * r * (r + c - r * c - 1);
}

// compute the phase shift between reflected and refracted-reflected wave
// for an arbitrary viewing angle.
//
BSDL_INLINE_METHOD float
ThinFilm::phase(float n, float d, float sinTheta)
{
    return d * std::sqrt((n - sinTheta) * (n + sinTheta));
}

//  We need this to compute the reflectance off the outer thin-film interface
BSDL_INLINE_METHOD float
ThinFilm::schlick(float cosTheta, float r)
{
    float c = 1 - cosTheta;
    return r + (1 - r) * SQR(SQR(c)) * c;
}

BSDL_INLINE_METHOD Imath::C3f
ThinFilm::thinFilmSpectrum(float cosTheta) const
{
    cosTheta = CLAMP(cosTheta, 0.0f, 1.0f);
    // Compute the amount of light reflected from the top interface, depending
    // on viewing angle. We're setting the normal incidence reflectivity to
    // 0.5 - it is arbitrary, but suitable, as we need something between
    // 0 and 1 to get an interference pattern.
    float refl = schlick(cosTheta, 0.5);

    // Make sure max thickness is at least as large as min thickness
    const float max_thickness = std::max(this->max_thickness, min_thickness);
    // We take the absolute value here since we support negative view_dependence (for
    // flipping the rainbow ramp)
    const float view_dependence_mix = std::fabs(view_dependence);

    // The thin film range is specified by min and max thickness, assuming an ior = 1 (for convenience).
    // If view dependent, we compute the corresponding ior and thickness needed for a
    // similar spectrum range for a varying view dependence
    const float ratio = CLAMP(min_thickness / max_thickness, 0.f, 1 - EPSILON);
    const float n     = 1 / std::sqrt(1 - ratio * ratio);  // 1 here is the ior

    const float d_view   = max_thickness / n;
    const float d_normal = LERP(thickness, min_thickness, max_thickness);

    // We support flipping the direction of the ramp when view-dependent, if view
    // dependence is negative.
    if (view_dependence < 0)
        cosTheta = std::sqrt(1 - cosTheta * cosTheta);

    // we're feeding in cosTheta, and not sinTheta as the phase function suggests - it is just
    // to make the direction of the view dependent ramp go from normal to grazing angle from left to right
    const float sigma_view = phase(n, d_view, cosTheta);
    const float sigma_norm = d_normal;
    const float sigma = 2 * LERP(view_dependence_mix, sigma_norm, sigma_view);

    Imath::C3f RGBout = { 0, 0, 0 };
    Imath::C3f XYZout = { 0, 0, 0 };

    auto XYZ_to_RGB = [](const Imath::C3f& xyzCol) -> Imath::C3f {
        Imath::V3f xyzVec = { xyzCol.x, xyzCol.y, xyzCol.z };

        // E - equal energy illuminant
        constexpr Imath::V3f redVec = { 2.3706743f, -0.9000405f, -0.4706338f };
        constexpr Imath::V3f grnVec = { -0.5138850f, 1.4253036f, 0.0885814f };
        constexpr Imath::V3f bluVec = { 0.0052982f, -0.0146949f, 1.0093968f };

        return { std::max(0.f, redVec.dot(xyzVec)),
                 std::max(0.f, grnVec.dot(xyzVec)),
                 std::max(0.f, bluVec.dot(xyzVec)) };
    };

    if (enhanced == 1) {
        // Normalization by Sandip Shukla.
        auto XYZ_normalize = [](const Imath::C3f& xyzCol) {
            return xyzCol
                   * (1
                      / std::max(1e-4f,
                                 sqrtf(xyzCol.x * xyzCol.x + xyzCol.y * xyzCol.y
                                       + xyzCol.z * xyzCol.z)));
        };
        // This is incorrect, but usually produces brighter colors and looks better.
        // First suggested by Sandip Shukla.
        for (int i = 0; i < Spectrum::LAMBDA_RES; ++i) {
            float wavelength = Spectrum::LAMBDA_MIN + i * Spectrum::LAMBDA_STEP;
            float spectrum   = interferenceEnergy(refl, sigma / wavelength);
            XYZout += spectrum * Spectrum::get_luts().xyz_response[i];
        }
        RGBout = XYZ_to_RGB(XYZ_normalize(XYZout));
    } else if (enhanced == 2) {
        // Normalization by Sandip Shukla.
        Imath::C3f XYZtotal = { EPSILON, EPSILON, EPSILON };
        for (int i = 0; i < Spectrum::LAMBDA_RES; ++i) {
            float wavelength = Spectrum::LAMBDA_MIN + i * Spectrum::LAMBDA_STEP;
            float spectrum   = interferenceEnergy(refl, sigma / wavelength);
            Imath::C3f CIEcolor = Spectrum::get_luts().xyz_response[i];
            XYZout += spectrum * CIEcolor;
            XYZtotal += CIEcolor;
        }
        // normalize spectrum colors
        XYZout.x /= XYZtotal.z;
        XYZout.y /= XYZtotal.y;
        XYZout.z /= XYZtotal.z;
        RGBout = XYZ_to_RGB(XYZout);
    } else {
        auto old_XYZ_normalize = [](const Imath::C3f& xyzCol) {
            return xyzCol
                   * (1 / std::max(1e-4f, xyzCol.x + xyzCol.y + xyzCol.z));
        };
        for (int i = 0; i < Spectrum::LAMBDA_RES; ++i) {
            float wavelength = Spectrum::LAMBDA_MIN + i * Spectrum::LAMBDA_STEP;
            float spectrum   = interferenceEnergy(refl, sigma / wavelength);
            XYZout += spectrum * Spectrum::get_luts().xyz_response[i];
        }
        RGBout = XYZ_to_RGB(old_XYZ_normalize(XYZout));
    }

    return RGBout;
}

BSDL_INLINE_METHOD Imath::C3f
ThinFilm::get(const Imath::V3f& wo, const Imath::V3f& wi, float roughness) const
{
    if (saturation == 0.0f)
        return { 1, 1, 1 };
    const float GLOSSY_HI  = 0.3f;
    const float DIFFUSE_LO = 0.6f;
    // Apply only to glossy exclusively.
    const float saturation = CLAMP(this->saturation, 0.0f, 1.0f);
    const float actual_saturation
        = saturation * (1 - SMOOTHSTEP(GLOSSY_HI, DIFFUSE_LO, roughness));
    if (actual_saturation == 0.0f)
        return { 1, 1, 1 };
    Imath::V3f m = (wi + wo).normalized();
    Imath::C3f c = thinFilmSpectrum(wo.dot(m));
    return LERP(saturation, Imath::C3f(1, 1, 1), c);
}

BSDL_LEAVE_NAMESPACE
