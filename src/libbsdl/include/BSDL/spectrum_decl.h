// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/config.h>
#include <BSDL/jakobhanika_decl.h>
#include <BSDL/static_virtual.h>
#include <Imath/ImathColor.h>

BSDL_ENTER_NAMESPACE

struct Power;
struct sRGBColorSpace;
struct ACEScgColorSpace;
struct BypassColorSpace;

using AbstractColorSpace
    = StaticVirtual<sRGBColorSpace, ACEScgColorSpace, BypassColorSpace>;

struct ColorSpace : public AbstractColorSpace {
    template<typename CS>
    BSDL_INLINE_METHOD ColorSpace(const CS* cs) : AbstractColorSpace(cs)
    {
    }

    BSDL_INLINE_METHOD Power upsample(const Imath::C3f rgb,
                                      float lambda_0) const;
    BSDL_INLINE_METHOD Imath::C3f downsample(const Power wave,
                                             float lambda_0) const;
};

struct sRGBColorSpace : public ColorSpace {
    BSDL_INLINE_METHOD sRGBColorSpace() : ColorSpace(this) {}
    BSDL_INLINE_METHOD Power upsample_impl(const Imath::C3f rgb,
                                           float lambda_0) const;
    BSDL_INLINE_METHOD Imath::C3f downsample_impl(const Power wave,
                                                  float lambda_0) const;

    static constexpr int BASIS_RES = 81;

private:
    struct Data {
        // Basis for RGB -> spectrum conversion. It is a color table because
        // they are three curves. Conversion is the linear combination of:
        //
        //   spectrum = R * curve1 + G * curve2 + B * curve3
        //
        // The sRGB basis roundtrips very well with its D65 illuminant. From
        // Spectral Primary Decomposition for Rendering with sRGB Reflectance
        //   Agatha Mallett, Cem Yuksel
        // https://graphics.geometrian.com/research/spectral-primaries.html
        Imath::C3f rgb_basis_sRGB[BASIS_RES];
    };

    static BSDL_INLINE_METHOD const Data& get_luts();
};

struct ACEScgColorSpace : public ColorSpace {
    BSDL_INLINE_METHOD ACEScgColorSpace() : ColorSpace(this) {}
    BSDL_INLINE_METHOD Power upsample_impl(const Imath::C3f rgb,
                                           float lambda_0) const;
    BSDL_INLINE_METHOD Imath::C3f downsample_impl(const Power wave,
                                                  float lambda_0) const;
};

struct BypassColorSpace : public ColorSpace {
    BSDL_INLINE_METHOD BypassColorSpace() : ColorSpace(this) {}
    BSDL_INLINE_METHOD Power upsample_impl(const Imath::C3f rgb,
                                           float lambda_0) const;
    BSDL_INLINE_METHOD Imath::C3f downsample_impl(const Power wave,
                                                  float lambda_0) const;
};

// This struct is a proxy for lookup tables and spectral rendering constants
struct Spectrum {
    // Wavelength range for our sampling and tables, in nanometers.
    static constexpr int LAMBDA_MIN   = 380;
    static constexpr int LAMBDA_MAX   = 780;
    static constexpr int LAMBDA_RANGE = LAMBDA_MAX - LAMBDA_MIN;
    // Tables will have entries spaced at this step size, in nanometers.
    static constexpr int LAMBDA_STEP = 5;
    // And therefore, thips is the table size, with an additional end point.
    static constexpr int LAMBDA_RES = LAMBDA_RANGE / LAMBDA_STEP + 1;

    struct Data {
        // CIE 1931 observer curves for going from spectrum to XYZ
        Imath::C3f xyz_response[LAMBDA_RES];
        // White point illuminants
        float D65_illuminant[LAMBDA_RES];
        float D60_illuminant[LAMBDA_RES];
    };

    static BSDL_INLINE_METHOD float wrap(float lambda)
    {
        return fmodf(lambda - LAMBDA_MIN, LAMBDA_RANGE) + LAMBDA_MIN;
    }

    static BSDL_INLINE_METHOD constexpr Data get_luts_ctxr();
    static BSDL_INLINE_METHOD const Data& get_luts();

    BSDL_INLINE_METHOD static constexpr float
    integrate_illuminant(const float* I, const Imath::C3f* xyz)
    {
        float s = 0;
        for (auto i = 0; i != LAMBDA_RES - 1; ++i) {
            const float a = I[i] * xyz[i].y;
            const float b = I[i + 1] * xyz[i + 1].y;
            s += (a + b) * 0.5f;
        }
        return s * LAMBDA_RANGE / (LAMBDA_RES - 1);
    }

    template<int N>
    static BSDL_INLINE_METHOD Imath::C3f spec_to_xyz(Power wave,
                                                     float lambda_0);

    static BSDL_INLINE_METHOD ColorSpace get_color_space(float lambda_0)
    {
        if (lambda_0 == 0)
            return BypassColorSpace();
        else {
            switch (BSDLConfig::current_color_space()) {
            case BSDLConfig::ColorSpaceTag::sRGB: return sRGBColorSpace();
            case BSDLConfig::ColorSpaceTag::ACEScg: return ACEScgColorSpace();
            default: return sRGBColorSpace();
            }
        }
    }

    static BSDL_INLINE_METHOD float get_dispersion_ior(const float dispersion,
                                                       const float basic_ior,
                                                       const float wavelength);

    template<typename T>
    static BSDL_INLINE_METHOD T lookup(float lambda, const T array[LAMBDA_RES]);
};

static_assert(sRGBColorSpace::BASIS_RES == Spectrum::LAMBDA_RES);

// Hero wavelength spectral representation. For every camera ray, a hero wavelength
// is chosen randomly. To reduce color noise, additional wavelengths, equaly spaced
// along the spectrum, are tracked. We use hero + 3, so 4 channels. And Power replaces
// the typical 3 channel RGB color representation. Otherwise is the same operations,
// + - * / that look the same as with old school RGB colors.
//
// Note this type also handles RGB by setting lambda_0 to 0.0, we just zero out
// the extra channels. This way the render can work in both RGB or spectral mode.
struct Power {
    // We track 4 floats that fit in a single simd register for CPUs. Always pass
    // by value for performance, even GPU.
    static constexpr unsigned N = BSDLConfig::HERO_WAVELENGTH_CHANNELS;
    // Wavelength spacing of the channels.
    static constexpr float HERO_STEP = float(Spectrum::LAMBDA_RANGE) / N;

    // Hero wave length lambda_0 is considered external, tracked by the integrator.
    // We only care about channel intensities. If a function needs lambda_0 it is
    // passed as an argument.
    float data[N];

    // Basic initialization takes either a float or a lambda function, so the typical
    //
    //   color.r = exp(-t * sigma.r);
    //   color.g = exp(-t * sigma.g);
    //   color.b = exp(-t * sigma.b);
    //
    // becomes
    //
    //   wave = Power([&] (int i) { return exp(-t * sigma[i]); }, lambda_0);
    //
    // or just wave = Power(0.17f, lambda_0); Where lambda_0 is only used to handle
    // the RGB mode.
    template<typename F>
    BSDL_INLINE_METHOD constexpr Power(const F& f, float lambda_0)
    {
        if constexpr (std::is_invocable_r<float, F, int>::value) {
            BSDL_UNROLL()
            for (int i = 0; i != N; ++i) {
                data[i] = i < 3 || lambda_0 != 0 ? f(i) : 0;
            }
        } else {
            BSDL_UNROLL()
            for (int i = 0; i != N; ++i)
                data[i] = f;
            if (lambda_0 == 0)
                for (int i = 3; i != N; ++i)
                    data[i] = 0;
        }
    }

    Power()               = default;
    Power(const Power& o) = default;

    // RGB to spectrum upsampling, lambda_0 is the hero wavelength (0 means RGB).
    BSDL_INLINE_METHOD Power(const Imath::C3f rgb, float lambda_0);
    template<typename F>
    BSDL_INLINE_METHOD void update(const F& f, float lambda_0)
    {
        BSDL_UNROLL()
        for (int i = 0; i != N; ++i)
            data[i] = i < 3 || lambda_0 != 0 ? f(i, data[i]) : 0;
    }

    static constexpr Power ZERO()
    {
        return Power(0, 1);  // Same for RGB or spectral
    }
    static constexpr Power UNIT()
    {
        return Power(1, 1);  // Watch out RGB use of this
    }

    // Convert back to RGB
    BSDL_INLINE_METHOD Imath::C3f toRGB(float lambda_0) const;
    // From wavelength from to to, including 0.0 for RGB
    BSDL_INLINE_METHOD Power resample(float from, float to) const;

    BSDL_INLINE_METHOD float& operator[](int i) { return data[i]; }
    BSDL_INLINE_METHOD float operator[](int i) const { return data[i]; }

    BSDL_INLINE_METHOD Power operator*=(const Power o);
    BSDL_INLINE_METHOD Power operator*=(float f);
    BSDL_INLINE_METHOD Power operator+=(const Power o);

    BSDL_INLINE_METHOD float max() const
    {
        float m = data[0];
        BSDL_UNROLL()
        for (int i = 1; i != N; ++i)
            m = std::max(m, data[i]);
        return m;
    }
    BSDL_INLINE_METHOD float max_abs() const
    {
        float m = fabsf(data[0]);
        BSDL_UNROLL()
        for (int i = 1; i != N; ++i)
            m = std::max(m, fabsf(data[i]));
        return m;
    }
    BSDL_INLINE_METHOD float min(float lambda_0) const
    {
        float m = data[0];
        BSDL_UNROLL()
        for (int i = 1; i != N; ++i)
            m = std::min(m, i != 3 || lambda_0 != 0 ? data[i] : m);
        return m;
    }
    BSDL_INLINE_METHOD float sum() const
    {
        float m = 0;
        BSDL_UNROLL()
        for (int i = 0; i != N; ++i)
            m += data[i];
        return m;
    }
    BSDL_INLINE_METHOD float avg(float lambda_0) const
    {
        return sum() / (lambda_0 > 0 ? N : 3);
    }
    BSDL_INLINE_METHOD float luminance(float lambda_0) const
    {
        return lambda_0 > 0
                   ? avg(lambda_0)
                   : data[0] * 0.3086f + data[1] * 0.6094f + data[2] * 0.0824f;
    }
    BSDL_INLINE_METHOD Power clamped(float a, float b) const
    {
        Power r;
        BSDL_UNROLL()
        for (int i = 0; i != N; ++i)
            r[i] = std::max(a, std::min(b, data[i]));
        return r;
    }
    BSDL_INLINE_METHOD Power scale_clamped(float maxv) const
    {
        const float scale = 1 / std::max(maxv, max());
        return Power([&](int i) { return data[i] * scale; }, 1);
    }
    BSDL_INLINE_METHOD bool is_zero(float eps = 0) const
    {
        for (int i = 0; i != N; ++i)
            if (fabsf(data[i]) > eps)
                return false;
        return true;
    }
    BSDL_INLINE_METHOD bool is_corrupted() const
    {
        BSDL_UNROLL()
        for (int i = 0; i != N; ++i)
            if (!std::isfinite(data[i]))
                return true;
        return false;
    }
    BSDL_INLINE_METHOD bool is_illegal() const
    {
        BSDL_UNROLL()
        for (int i = 0; i != N; ++i)
            if (!std::isfinite(data[i]) || data[i] < 0)
                return true;
        return false;
    }
    BSDL_INLINE_METHOD Power cliped_rgb(float lambda_0) const
    {
        Power m = *this;
        for (int i = lambda_0 > 0 ? N : 3; i != N; ++i)
            m.data[i] = 0;
        return m;
    }
};

BSDL_INLINE Power
sqrt(Power x)
{
    x = x.clamped(0.0f, std::numeric_limits<float>::max());
    BSDL_UNROLL()
    for (int i = 0; i != Power::N; ++i)
        x.data[i] = sqrtf(x.data[i]);
    return x;
}

BSDL_INLINE Power
operator-(Power o)
{
    return Power([&](int i) { return -o[i]; }, 1);
}

BSDL_INLINE_METHOD Power
operator*(const Power a, const Power o)
{
    Power n;
    BSDL_UNROLL()
    for (int i = 0; i != Power::N; ++i)
        n.data[i] = a.data[i] * o.data[i];
    return n;
}

BSDL_INLINE_METHOD Power
operator/(const Power a, const Power o)
{
    Power n;
    BSDL_UNROLL()
    for (int i = 0; i != Power::N; ++i)
        n.data[i] = a.data[i] / o.data[i];
    return n;
}

BSDL_INLINE_METHOD Power
operator*(const Power a, float f)
{
    Power n;
    BSDL_UNROLL()
    for (int i = 0; i != Power::N; ++i)
        n.data[i] = a.data[i] * f;
    return n;
}

BSDL_INLINE_METHOD Power
operator+(const Power a, const Power o)
{
    Power n;
    BSDL_UNROLL()
    for (int i = 0; i != Power::N; ++i)
        n.data[i] = a.data[i] + o.data[i];
    return n;
}

BSDL_INLINE_METHOD Power
operator-(const Power a, const Power o)
{
    Power n;
    BSDL_UNROLL()
    for (int i = 0; i != Power::N; ++i)
        n.data[i] = a.data[i] - o.data[i];
    return n;
}

BSDL_INLINE_METHOD Power
Power::operator*=(const Power o)
{
    BSDL_UNROLL()
    for (int i = 0; i != N; ++i)
        data[i] *= o.data[i];
    return *this;
}

BSDL_INLINE_METHOD Power
Power::operator*=(float f)
{
    for (int i = 0; i != N; ++i)
        data[i] *= f;
    return *this;
}

BSDL_INLINE_METHOD Power
Power::operator+=(const Power o)
{
    BSDL_UNROLL()
    for (int i = 0; i != N; ++i)
        data[i] += o.data[i];
    return *this;
}

BSDL_INLINE Power
operator*(float f, const Power o)
{
    return o * f;
}

BSDL_LEAVE_NAMESPACE
