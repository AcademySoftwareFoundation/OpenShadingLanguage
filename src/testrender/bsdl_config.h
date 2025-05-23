#pragma once

#define BSDL_INLINE        static inline OSL_HOSTDEVICE
#define BSDL_INLINE_METHOD inline OSL_HOSTDEVICE
#define BSDL_DECL          OSL_DEVICE
#define BSDL_UNROLL()      // Do nothing

#include <BSDL/config.h>

#include <OpenImageIO/fmath.h>

struct BSDLConfig : public bsdl::BSDLDefaultConfig {
    // testrender won't do spectral render, just 3 channels covers RGB
    static constexpr int HERO_WAVELENGTH_CHANNELS = 3;

    struct Fast {
        static BSDL_INLINE_METHOD float cosf(float x)
        {
            return OIIO::fast_cos(x);
        }
        static BSDL_INLINE_METHOD float sinf(float x)
        {
            return OIIO::fast_sin(x);
        }
        static BSDL_INLINE_METHOD float asinf(float x)
        {
            return OIIO::fast_asin(x);
        }
        static BSDL_INLINE_METHOD float acosf(float x)
        {
            return OIIO::fast_acos(x);
        }
        static BSDL_INLINE_METHOD float atan2f(float y, float x)
        {
            return OIIO::fast_atan2(y, x);
        }
        static BSDL_INLINE_METHOD void sincosf(float x, float* s, float* c)
        {
            return OIIO::fast_sincos(x, s, c);
        }
        static BSDL_INLINE_METHOD float sinpif(float x)
        {
            return OIIO::fast_sinpi(x);
        }
        static BSDL_INLINE_METHOD float cospif(float x)
        {
            return OIIO::fast_cospi(x);
        }
        static BSDL_INLINE_METHOD float expf(float x)
        {
            return OIIO::fast_exp(x);
        }
        static BSDL_INLINE_METHOD float exp2f(float x)
        {
            return OIIO::fast_exp2(x);
        }
        static BSDL_INLINE_METHOD float logf(float x)
        {
            return OIIO::fast_log(x);
        }
        static BSDL_INLINE_METHOD float log2f(float x)
        {
            return OIIO::fast_log2(x);
        }
        static BSDL_INLINE_METHOD float log1pf(float x)
        {
            return OIIO::fast_log1p(x);
        }
        static BSDL_INLINE_METHOD float powf(float x, float y)
        {
            return OIIO::fast_safe_pow(x, y);
        }
    };

    static BSDL_INLINE_METHOD ColorSpaceTag current_color_space()
    {
        return ColorSpaceTag::sRGB;
    }
    static BSDL_INLINE_METHOD const JakobHanikaLut*
    get_jakobhanika_lut(ColorSpaceTag cs)
    {
        return nullptr;
    }
};
