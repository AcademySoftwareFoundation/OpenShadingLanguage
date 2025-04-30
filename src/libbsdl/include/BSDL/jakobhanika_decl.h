// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once
#include <BSDL/config.h>

BSDL_ENTER_NAMESPACE

struct JakobHanikaUpsampler {
    struct SigmoidPolynomial {
        BSDL_INLINE_METHOD SigmoidPolynomial(float a, float b, float c)
            : a(a), b(b), c(c)
        {
        }
        static BSDL_INLINE_METHOD float sigmoid(float x)
        {
            return !std::isfinite(x) ? (x > 0 ? 1 : 0)
                                     : 0.5f + x / (2 * std::sqrt(1 + x * x));
        }
        BSDL_INLINE_METHOD float operator()(float x) const
        {
            return sigmoid((a * x + b) * x + c);
        }

        float a, b, c;
    };

    BSDL_INLINE_METHOD
    JakobHanikaUpsampler(const BSDLConfig::JakobHanikaLut* lut) : lut(lut) {}

    BSDL_INLINE_METHOD SigmoidPolynomial lookup(float c_r, float c_g,
                                                float c_b) const;

    const BSDLConfig::JakobHanikaLut* lut;
};

BSDL_LEAVE_NAMESPACE
