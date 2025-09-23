// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once
#include <BSDL/jakobhanika_decl.h>
#include <BSDL/tools.h>
#include <Imath/ImathColor.h>

BSDL_ENTER_NAMESPACE

template<typename Predicate>
BSDL_INLINE int
find_interval(int sz, const Predicate& pred)
{
    int size = sz - 2, first = 1;
    while (size > 0) {
        // Evaluate predicate at midpoint and update _first_ and _size_
        int half = size >> 1, middle = first + half;
        bool predResult = pred(middle);
        first           = predResult ? middle + 1 : first;
        size            = predResult ? size - (half + 1) : half;
    }
    return CLAMP(first - 1, 0, sz - 2);
}

// Operators so we can LERP them
BSDL_INLINE BSDLConfig::JakobHanikaLut::Coeff
operator*(const BSDLConfig::JakobHanikaLut::Coeff c, float f)
{
    BSDLConfig::JakobHanikaLut::Coeff res;
    for (int i = 0; i != BSDLConfig::JakobHanikaLut::Coeff::NPAD; ++i)
        res.c[i] = c.c[i] * f;
    return res;
}
BSDL_INLINE BSDLConfig::JakobHanikaLut::Coeff
operator*(float f, const BSDLConfig::JakobHanikaLut::Coeff c)
{
    return c * f;
}
BSDL_INLINE BSDLConfig::JakobHanikaLut::Coeff
operator+(const BSDLConfig::JakobHanikaLut::Coeff a,
          const BSDLConfig::JakobHanikaLut::Coeff b)
{
    BSDLConfig::JakobHanikaLut::Coeff res;
    for (int i = 0; i != BSDLConfig::JakobHanikaLut::Coeff::NPAD; ++i)
        res.c[i] = a.c[i] + b.c[i];
    return res;
}

BSDL_INLINE_METHOD JakobHanikaUpsampler::SigmoidPolynomial
JakobHanikaUpsampler::lookup(float c_r, float c_g, float c_b) const
{
    constexpr int RGB_RES = BSDLConfig::JakobHanikaLut::RGB_RES;

    assert(0 <= c_r && c_r <= 1.0f && 0 <= c_g && c_g <= 1.0f && 0 <= c_b
           && c_b <= 1.0f);

    // Find maximum component and compute remapped component values
    const int maxc = (c_r > c_g) ? ((c_r > c_b) ? 0 : 2)
                                 : ((c_g > c_b) ? 1 : 2);

    const float z = maxc == 0 ? c_r : (maxc == 1 ? c_g : c_b);
    const float x = (maxc == 0 ? c_g : (maxc == 1 ? c_b : c_r)) * (RGB_RES - 1)
                    / z;
    const float y = (maxc == 0 ? c_b : (maxc == 1 ? c_r : c_g)) * (RGB_RES - 1)
                    / z;

    // Compute integer indices and offsets for coefficient interpolation
    const int xi   = std::min((int)x, RGB_RES - 2),
              yi   = std::min((int)y, RGB_RES - 2),
              zi   = find_interval(RGB_RES,
                                   [&](int i) { return lut->scale[i] < z; });
    const float dx = x - xi, dy = y - yi,
                dz = (z - lut->scale[zi])
                     / (lut->scale[zi + 1] - lut->scale[zi]);

    // Trilinearly interpolate sigmoid polynomial coefficients
    auto co = [&](int dx, int dy, int dz) {
        return lut->coeff[maxc][zi + dz][yi + dy][xi + dx];
    };

    const BSDLConfig::JakobHanikaLut::Coeff c
        = LERP(dz,
               LERP(dy, LERP(dx, co(0, 0, 0), co(1, 0, 0)),
                    LERP(dx, co(0, 1, 0), co(1, 1, 0))),
               LERP(dy, LERP(dx, co(0, 0, 1), co(1, 0, 1)),
                    LERP(dx, co(0, 1, 1), co(1, 1, 1))));

    return SigmoidPolynomial(c.c[0], c.c[1], c.c[2]);
}

BSDL_LEAVE_NAMESPACE
