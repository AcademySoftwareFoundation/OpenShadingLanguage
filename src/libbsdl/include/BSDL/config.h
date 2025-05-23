// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#ifndef BSDL_NS
#    define BSDL_NS bsdl
#endif

#define BSDL_ENTER_NAMESPACE namespace BSDL_NS {
#define BSDL_LEAVE_NAMESPACE }

#ifndef BSDL_INLINE_METHOD
#    define BSDL_INLINE_METHOD inline
#endif
#ifndef BSDL_INLINE
#    define BSDL_INLINE inline
#endif
#ifndef BSDL_DECL
#    define BSDL_DECL
#endif
#ifndef BSDL_UNROLL
#    define BSDL_UNROLL() _Pragma("unroll")
#endif

#include <cassert>
#include <cmath>

#ifndef M_PI
#    define M_PI 3.1415926535897932
#endif

BSDL_ENTER_NAMESPACE

BSDL_DECL constexpr float EPSILON   = 1e-4f;
BSDL_DECL constexpr float PI        = float(M_PI);
BSDL_DECL constexpr float ONEOVERPI = 1 / PI;
BSDL_DECL constexpr float ALMOSTONE
    = 0.999999940395355224609375f;  // Max float  < 1.0f
BSDL_DECL constexpr float FLOAT_MIN
    = 1.17549435e-38f;  // Minimum float normal value
BSDL_DECL constexpr float SQRT2   = 1.4142135623730951f;
BSDL_DECL constexpr float BIG     = 1e12f;
BSDL_DECL constexpr float PDF_MIN = 1e-6f;

struct BSDLDefaultConfig {
    struct Fast {
        static BSDL_INLINE_METHOD float cosf(float x) { return std::cos(x); }
        static BSDL_INLINE_METHOD float sinf(float x) { return std::sin(x); }
        static BSDL_INLINE_METHOD void sincosf(float x, float* s, float* c)
        {
            *s = std::sin(x);
            *c = std::cos(x);
        }
        static BSDL_INLINE_METHOD float sinpif(float x)
        {
            return std::sin(x * float(M_PI));
        }
        static BSDL_INLINE_METHOD float cospif(float x)
        {
            return std::cos(x * float(M_PI));
        }
        static BSDL_INLINE_METHOD float asinf(float x) { return std::asin(x); }
        static BSDL_INLINE_METHOD float acosf(float x) { return std::acos(x); }
        static BSDL_INLINE_METHOD float atan2f(float x, float y)
        {
            return std::atan2(x, y);
        }
        static BSDL_INLINE_METHOD float expf(float x) { return std::exp(x); }
        static BSDL_INLINE_METHOD float exp2f(float x) { return std::exp2(x); }
        static BSDL_INLINE_METHOD float logf(float x) { return std::log(x); }
        static BSDL_INLINE_METHOD float log2f(float x) { return std::log2(x); }
        static BSDL_INLINE_METHOD float log1pf(float x)
        {
            return std::log1p(x);
        }
        static BSDL_INLINE_METHOD float powf(float x, float y)
        {
            return x < FLOAT_MIN ? 0 : std::pow(x, y);
        }
        static BSDL_INLINE_METHOD float erff(float x) { return std::erf(x); }
        // No default implementation
        //  static BSDL_INLINE_METHOD float ierff(float x) { return std::erfinv(x); }
    };

    static constexpr int HERO_WAVELENGTH_CHANNELS = 4;

    enum class ColorSpaceTag { sRGB, ACEScg };

    static BSDL_INLINE_METHOD ColorSpaceTag current_color_space()
    {
        return ColorSpaceTag::ACEScg;
    }

    struct JakobHanikaLut {
        struct Coeff {
            static constexpr int N    = 3;
            static constexpr int NPAD = 4;  // waste 1 to get SSE
            float c[NPAD];
        };
        static constexpr int RGB_RES = 64;

        float scale[RGB_RES];
        Coeff coeff[3][RGB_RES][RGB_RES][RGB_RES];
    };

    static BSDL_INLINE_METHOD const JakobHanikaLut*
    get_jakobhanika_lut(ColorSpaceTag cs)
    {
        switch (cs) {
        case ColorSpaceTag::ACEScg: return &JH_ACEScg_lut;
        case ColorSpaceTag::sRGB:
            return nullptr;  // not handled by Jakob-Hanika
        default: return nullptr;
        }
    }

    static const JakobHanikaLut JH_ACEScg_lut;
};

BSDL_LEAVE_NAMESPACE
