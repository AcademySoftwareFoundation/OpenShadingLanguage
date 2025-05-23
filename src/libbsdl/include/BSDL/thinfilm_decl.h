// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

BSDL_ENTER_NAMESPACE

struct ThinFilm {
    float saturation = 0;
    float min_thickness;
    float max_thickness;
    float view_dependence;
    float thickness;
    int enhanced;

    static BSDL_INLINE_METHOD float interferenceEnergy(float r, float s);
    static BSDL_INLINE_METHOD float phase(float n, float d, float sinTheta);
    static BSDL_INLINE_METHOD float schlick(float cosTheta, float r);
    BSDL_INLINE_METHOD Imath::C3f thinFilmSpectrum(float cosTheta) const;

    BSDL_INLINE_METHOD Imath::C3f
    get(const Imath::V3f& wo, const Imath::V3f& wi, float roughness) const;
};

BSDL_LEAVE_NAMESPACE
