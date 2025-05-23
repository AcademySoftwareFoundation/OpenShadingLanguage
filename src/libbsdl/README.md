# BSDL library

BSDL is the working title (Bidirectional Scattering Distribution Library) for
this header only collection of BSDFs. It is self-contained depending only on
Imath and intended to be used in any renderer. This is a build only dependency.

The basic idea is that you choose a BSDF from it and give it a thin wrapper to
integrate in your renderer. There is an example of this in OSL's testrender. The
key features are:
  * BSDFs provide constructor, eval and sample methods.
  * Everything is inlined (split among _decl.h and _imple.h headers) to be GPU friendly.
  * There is a template based autiomatic switch/case virtual dispatch mechanism to be GPU firendly.
  * It works both in RGB or spectral mode (via hero wavelength).
  * It includes the Sony Pictures Imageworks set of BSDFs used for movie production.

## Pseudo virtual methods

The header static_virtual.h provides the ```StaticVirtual<type1, type2, ...>```
template. If you inherit from it will implement a dispatch() method to execute
"virtual" methods covering ```type1, type2, ...``` using switch case statements.
The idea was borrowed from PBRT but the implementation is different. See the
use in testrender (shading.h/cpp).

## The ```Power``` type

To support both RGB and spectral render with the same code we use the 
```Power``` type, which is just a float array. It has from RGB construction and
conversion. But when 'lambda_0', the hero wavelength is zero, these are no-ops
and pure RGB rendering is assueme. Take into account spectral support in BSDL
is in very early stages.

## Lookup tables

A bunch of BSDFs use albedo lookup tables. Those are generated at build time
very quickly and put in headers. You shouldn't have to do anything.

## Usage

Either ```include(bsdl.cmake)``` and call ```add_bsdl_library(my_name)``` to
create a 'my_name' INTERFACE library that you then link, or 
```add_subdirectory (path_to_libbsdl)``` to get it as 'BSDL'.

For the integration, a config header needs to be defined as you can see in
testrender's 'bsdl_config.h'.
```cpp
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
        static BSDL_INLINE_METHOD float cosf(float x) { return OIIO::fast_cos(x); }
        static BSDL_INLINE_METHOD float sinf(float x) { return OIIO::fast_sin(x); }
        static BSDL_INLINE_METHOD float asinf(float x) { return OIIO::fast_asin(x); }
        static BSDL_INLINE_METHOD float acosf(float x) { return OIIO::fast_acos(x); }
        static BSDL_INLINE_METHOD float atan2f(float y, float x) { return OIIO::fast_atan2(y, x); }
        static BSDL_INLINE_METHOD void sincosf(float x, float* s, float* c) { return OIIO::fast_sincos(x, s, c); }
        static BSDL_INLINE_METHOD float sinpif(float x) { return OIIO::fast_sinpi(x); }
        static BSDL_INLINE_METHOD float cospif(float x) { return OIIO::fast_cospi(x); }
        static BSDL_INLINE_METHOD float expf(float x) { return OIIO::fast_exp(x); }
        static BSDL_INLINE_METHOD float exp2f(float x) { return OIIO::fast_exp2(x); }
        static BSDL_INLINE_METHOD float logf(float x) { return OIIO::fast_log(x); }
        static BSDL_INLINE_METHOD float log2f(float x) { return OIIO::fast_log2(x); }
        static BSDL_INLINE_METHOD float log1pf(float x) { return OIIO::fast_log1p(x); }
        static BSDL_INLINE_METHOD float powf(float x, float y) { return OIIO::fast_safe_pow(x, y); }
    };

    // Don't care for colorspaces/spectral
    static BSDL_INLINE_METHOD ColorSpaceTag current_color_space() { return ColorSpaceTag::sRGB; }
    static BSDL_INLINE_METHOD const JakobHanikaLut* get_jakobhanika_lut(ColorSpaceTag cs) { return nullptr; }
};
```

And this header should be included before you include anything else from BSDL.

If spectral rendering is desired there is ready to use sRGB upsampling via
spectral primaries by Mallett and Yuksel. For wide gamut color spaces like
ACEScg we use Jakob and Hanika approach, but you have to ask for the tables
in .cpp for from cmake:
```
add_bsdl_library(BSDL SPECTRAL_COLOR_SPACES "ACEScg" "ACES2065")
``` 

which will bake tables to two cpp files and return them in 'BSDL_LUTS_CPP'.
Then you need to include that in your sources.