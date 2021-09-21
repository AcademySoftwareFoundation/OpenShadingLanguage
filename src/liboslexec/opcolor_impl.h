// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shared implementation of color operations
/// between opcolor.cpp and wide_opcolor.cpp.
///
/////////////////////////////////////////////////////////////////////////

#include <OSL/oslconfig.h>

#include <OpenImageIO/fmath.h>

#include <OSL/sfmath.h>


OSL_NAMESPACE_ENTER

#ifdef __OSL_WIDE_PVT
    namespace __OSL_WIDE_PVT {
#else
    namespace pvt {
#endif


namespace {

OSL_HOSTDEVICE inline void clamp_zero (Color3 &c)
{
    if (c.x < 0.0f)
        c.x = 0.0f;
    if (c.y < 0.0f)
        c.y = 0.0f;
    if (c.z < 0.0f)
        c.z = 0.0f;
}

// CIE colour matching functions xBar, yBar, and zBar for
//   wavelengths from 380 through 780 nanometers, every 5
//   nanometers.  For a wavelength lambda in this range:
//        cie_colour_match[(lambda - 380) / 5][0] = xBar
//        cie_colour_match[(lambda - 380) / 5][1] = yBar
//        cie_colour_match[(lambda - 380) / 5][2] = zBar
//OSL_CONSTANT_DATA const float cie_colour_match[81][3] =
// Choose to access 1d array vs 2d to allow better code generation of gathers
OSL_CONSTANT_DATA const float cie_colour_match[81*3] __attribute__((aligned(64))) =
{
    0.0014,0.0000,0.0065, 0.0022,0.0001,0.0105, 0.0042,0.0001,0.0201,
    0.0076,0.0002,0.0362, 0.0143,0.0004,0.0679, 0.0232,0.0006,0.1102,
    0.0435,0.0012,0.2074, 0.0776,0.0022,0.3713, 0.1344,0.0040,0.6456,
    0.2148,0.0073,1.0391, 0.2839,0.0116,1.3856, 0.3285,0.0168,1.6230,
    0.3483,0.0230,1.7471, 0.3481,0.0298,1.7826, 0.3362,0.0380,1.7721,
    0.3187,0.0480,1.7441, 0.2908,0.0600,1.6692, 0.2511,0.0739,1.5281,
    0.1954,0.0910,1.2876, 0.1421,0.1126,1.0419, 0.0956,0.1390,0.8130,
    0.0580,0.1693,0.6162, 0.0320,0.2080,0.4652, 0.0147,0.2586,0.3533,
    0.0049,0.3230,0.2720, 0.0024,0.4073,0.2123, 0.0093,0.5030,0.1582,
    0.0291,0.6082,0.1117, 0.0633,0.7100,0.0782, 0.1096,0.7932,0.0573,
    0.1655,0.8620,0.0422, 0.2257,0.9149,0.0298, 0.2904,0.9540,0.0203,
    0.3597,0.9803,0.0134, 0.4334,0.9950,0.0087, 0.5121,1.0000,0.0057,
    0.5945,0.9950,0.0039, 0.6784,0.9786,0.0027, 0.7621,0.9520,0.0021,
    0.8425,0.9154,0.0018, 0.9163,0.8700,0.0017, 0.9786,0.8163,0.0014,
    1.0263,0.7570,0.0011, 1.0567,0.6949,0.0010, 1.0622,0.6310,0.0008,
    1.0456,0.5668,0.0006, 1.0026,0.5030,0.0003, 0.9384,0.4412,0.0002,
    0.8544,0.3810,0.0002, 0.7514,0.3210,0.0001, 0.6424,0.2650,0.0000,
    0.5419,0.2170,0.0000, 0.4479,0.1750,0.0000, 0.3608,0.1382,0.0000,
    0.2835,0.1070,0.0000, 0.2187,0.0816,0.0000, 0.1649,0.0610,0.0000,
    0.1212,0.0446,0.0000, 0.0874,0.0320,0.0000, 0.0636,0.0232,0.0000,
    0.0468,0.0170,0.0000, 0.0329,0.0119,0.0000, 0.0227,0.0082,0.0000,
    0.0158,0.0057,0.0000, 0.0114,0.0041,0.0000, 0.0081,0.0029,0.0000,
    0.0058,0.0021,0.0000, 0.0041,0.0015,0.0000, 0.0029,0.0010,0.0000,
    0.0020,0.0007,0.0000, 0.0014,0.0005,0.0000, 0.0010,0.0004,0.0000,
    0.0007,0.0002,0.0000, 0.0005,0.0002,0.0000, 0.0003,0.0001,0.0000,
    0.0002,0.0001,0.0000, 0.0002,0.0001,0.0000, 0.0001,0.0000,0.0000,
    0.0001,0.0000,0.0000, 0.0001,0.0000,0.0000, 0.0000,0.0000,0.0000
};



// For a given wavelength lambda (in nm), return the XYZ triple giving the
// XYZ color corresponding to that single wavelength;
OSL_HOSTDEVICE static Color3
wavelength_color_XYZ (float lambda_nm)
{
    float ii = (lambda_nm-380.0f) / 5.0f;  // scaled 0..80
    int i = (int) ii;
    // NOTE: bitwise OR to avoid branchiness logical OR introduces.
    // Also when left as logical OR, vectorizing with clang 11 did not
    // mask off gathering of out of range index values causing segfaults
    if ((i < 0) | (i >= 80))
        return Color3(0.0f,0.0f,0.0f);
    float remainder = ii - i;
    // Do not separate address calculation so that compiler can see that the
    // base pointer is uniform for all data lanes and 32bit indices can be
    // used for gather instructions.  Otherwise each data lane ends up with its
    // own 64bit address of c;
    // Furthermore, have all gathers use the same indices by having different
    // base registers for each of the 6 components to avoid need to manipulate
    // indices between gathers.
    constexpr int stride = sizeof(Color3)/sizeof(float);
    const int si = stride*i;
    Color3 XYZ = OIIO::lerp (Color3((cie_colour_match+0)[si], (cie_colour_match+1)[si], (cie_colour_match+2)[si]),
                             Color3((cie_colour_match+3)[si], (cie_colour_match+4)[si], (cie_colour_match+5)[si]), remainder);
#if 0
    float n = (XYZ[0] + XYZ[1] + XYZ[2]);
    float n_inv = (n >= 1.0e-6f ? 1.0f/n : 0.0f);
    XYZ *= n_inv;
#endif
    return XYZ;
}

}  // End anonymous namespace

// In order to speed up the blackbody computation, we have a table
// storing the precomputed BB values for a range of temperatures.  Less
// than BB_DRAPER always returns 0.  Greater than BB_MAX_TABLE_RANGE
// does the full computation, we think it'll be rare to inquire higher
// temperatures.
//
// Since the bb function is so nonlinear, we actually space the table
// entries nonlinearly, with the relationship between the table index i
// and the temperature T as follows:
//   i = ((T-Draper)/spacing)^(1/xpower)
//   T = pow(i, xpower) * spacing + Draper
// And furthermore, we store in the table the true value raised ^(1/5).
// I tuned this a bit, and with the current values we can have all
// blackbody results accurate to within 0.1% with a table size of 317
// (about 5 KB of data).
#define BB_DRAPER 800.0f /* really 798K, below this visible BB is negligible */
#define BB_MAX_TABLE_RANGE 12000.0f /* max temp for which we use the table */
#define BB_TABLE_XPOWER 1.5f       // NOTE: not used, hardcoded into expressions below
#define BB_TABLE_YPOWER 5.0f       // NOTE: decode is hardcoded
#define BB_TABLE_SPACING 2.0f

OSL_HOSTDEVICE inline float BB_TABLE_MAP(float i) {
    // return powf (i, BB_TABLE_XPOWER) * BB_TABLE_SPACING + BB_DRAPER;
    float is = sqrtf(i);
    float ip = is * is * is; // ^3/2
    return ip * BB_TABLE_SPACING + BB_DRAPER;
}

OSL_HOSTDEVICE inline float BB_TABLE_UNMAP(float T) {
    // return powf ((T - BB_DRAPER) / BB_TABLE_SPACING, 1.0f/BB_TABLE_XPOWER);
    float t  = (T - BB_DRAPER) / BB_TABLE_SPACING;
    //using cbrtf_impl = OIIO::fast_cbrtf;
    //using cbrtf_impl = ::cbrtf;
#if defined(__OSL_WIDE_PVT) && OSL_CLANG_VERSION && !OSL_INTEL_COMPILER
    // Clang doesn't have vectorizing version of cbrtf,
    // so use OIIO's instead (TODO: should we always use OIIO version?)
    float ic = OIIO::fast_cbrt(t);
#else
    float ic = cbrtf(t);
#endif
    return ic * ic; // ^2/3
}


// Spectral rendering routines inspired by those found at:
//   http://www.fourmilab.ch/documents/specrend/specrend.c
// which bore the notice:
//                Colour Rendering of Spectra
//                     by John Walker
//                  http://www.fourmilab.ch/
//         Last updated: March 9, 2003
//           This program is in the public domain.
//    For complete information about the techniques employed in
//    this program, see the World-Wide Web document:
//             http://www.fourmilab.ch/documents/specrend/


// Functor that calculates, by Planck's radiation law, the black body
// emittance at temperature (in Kelvin) and given wavelength (in nm).
// This is the differential (per unit of wavelength) flux density, in
// W/m^2 in the range [wavelength,wavelength+dwavelength].
class bb_spectrum {
public:
    OSL_HOSTDEVICE bb_spectrum (float temperature=5000) : m_temp(temperature) { }
    OSL_HOSTDEVICE float operator() (float wavelength_nm) const {
        // TODO: Evaluate if this math has to be double precision?
        double wlm = wavelength_nm * 1e-9;   // Wavelength in meters
        const double c1 = 3.74183e-16; // 2*pi*h*c^2, W*m^2
        const double c2 = 1.4388e-2;   // h*c/k, m*K
                                       // h is Planck's const, k is Boltzmann's
#if OSL_CLANG_VERSION && !OSL_INTEL_COMPILER
        // Clang doesn't have vectorizing version of expm1,
        // so define a double version based on OIIO::fast_expm1
        auto expm1 = [](double val)->double {
            if (abs(val) < 0.03) {
                double v = 1.0 - (1.0 - val); // handle denormals
                double v2 = v*v;
                return copysign(((0.5*v2) + v2), val);
            } else {
                return ::exp(val) - 1.0;
            }
        };
#else
        using ::expm1;
#endif

        return float((c1 * std::pow(wlm,-5.0)) / expm1(c2 / (wlm * m_temp)));
    }
private:
    double m_temp;
};


// Integrate the CIE color matching values, weighted by function
// spec_intens(lambda_nm), returning the aggregate XYZ color.
template<class SPECTRUM> OSL_HOSTDEVICE
static Color3
spectrum_to_XYZ (const SPECTRUM &spec_intens)
{
    float X = 0, Y = 0, Z = 0;
    const float dlambda = 5.0f * 1e-9;  // in meters
    for (int i = 0; i < 81; ++i) {
        float lambda = 380.0f + 5.0f * i;
        // N.B. spec_intens returns result in W/m^2 but it's a differential,
        // needs to be scaled by dlambda!
        float Me = spec_intens(lambda) * dlambda;
        constexpr int stride = sizeof(Color3)/sizeof(float);
        const int si = stride*i;
        X += Me * cie_colour_match[si + 0];
        Y += Me * cie_colour_match[si + 1];
        Z += Me * cie_colour_match[si + 2];
    }
    return Color3 (X, Y, Z);
}


template <typename COLOR3> OSL_HOSTDEVICE
static COLOR3
hsv_to_rgb (const COLOR3& hsv)
{
    // Reference for this technique: Foley & van Dam
    using FLOAT = typename ScalarFromVec<COLOR3>::type;
    FLOAT h = comp_x(hsv), s = comp_y(hsv), v = comp_z(hsv);
    if (s < 0.0001f) {
      return make_Color3 (v, v, v);
    } else {
        using std::floor;   // to pick up the float one
        using OIIO::ifloor;
        h = 6.0f * (h - floor(h));  // expand to [0..6)
        int hi = ifloor(h);
        FLOAT f = h - FLOAT(hi);
        FLOAT p = v * (1.0f-s);
        FLOAT q = v * (1.0f-s*f);
        FLOAT t = v * (1.0f-s*(1.0f-f));
#ifdef __OSL_WIDE_PVT
        // Avoid switch statement vectorizor doesn't like
        // Also avoid if/else nest which some optimizers might
        // convert back into a switch statement
#   if OSL_CLANG_VERSION && !OSL_INTEL_COMPILER
        // Clang was still transforming series of if's back into a switch.
        // Alternate between == and <= comparisons to avoid
#       define __OSL_ASC_EQ <=
#   else
#       define __OSL_ASC_EQ ==
#   endif
        if (hi == 0) {
            return make_Color3 (v, t, p);
        }
        if (hi __OSL_ASC_EQ 1) {
            return make_Color3 (q, v, p);
        }
        if (hi == 2) {
            return make_Color3 (p, v, t);
        }
        if (hi __OSL_ASC_EQ 3) {
            return make_Color3 (p, q, v);
        }
        if (hi == 4) {
            return make_Color3 (t, p, v);
        }
        return make_Color3 (v, p, q);
#    undef __OSL_ASC_EQ
#else
        // serial execution might be faster with switch
        switch (hi) {
        case 0 : return make_Color3 (v, t, p);
        case 1 : return make_Color3 (q, v, p);
        case 2 : return make_Color3 (p, v, t);
        case 3 : return make_Color3 (p, q, v);
        case 4 : return make_Color3 (t, p, v);
        default: return make_Color3 (v, p, q);
        }
#endif
    }
}

template <typename COLOR3> OSL_HOSTDEVICE
static inline COLOR3
rgb_to_hsv (const COLOR3& rgb)
{
    // See Foley & van Dam
    using FLOAT = typename ScalarFromVec<COLOR3>::type;
    FLOAT r = comp_x(rgb), g = comp_y(rgb), b = comp_z(rgb);
    FLOAT mincomp = sfm::min_val (r, sfm::min_val (g, b));
    FLOAT maxcomp = sfm::max_val (r, sfm::max_val (g, b));
    FLOAT delta = maxcomp - mincomp;  // chroma
    FLOAT v = maxcomp;
    FLOAT s = 0.0f;
    if (maxcomp > 0.0f)
        s = delta / maxcomp;
    FLOAT h = 0.0f;
    if (s > 0.0f)
    {
#if 0   // Reference version
        if      (r >= maxcomp) h = (g-b) / delta;
        else if (g >= maxcomp) h = 2.0f + (b-r) / delta;
        else                   h = 4.0f + (r-g) / delta;
        h *= (1.0f/6.0f);
#else
        // Avoid masked execution of math (sub, div, add) 3 times,
        // instead just setup the arguments conditionally
        // and perform math 1 time.  Also bake in the
        // divide by 6 into K constants and delta
        // which was already going to be a divisor.
        float k;
        FLOAT x, y;
        if      (r >= maxcomp) { k = 0.0f/6.0f; x = g; y = b; }
        else if (g >= maxcomp) { k = 2.0f/6.0f; x = b; y = r; }
        else                   { k = 4.0f/6.0f; x = r; y = g; }
        h = k + (x-y) / (6.0f*delta);
#endif
        if (h < 0.0f)
            h += 1.0f;
    }
    return make_Color3 (h, s, v);
}

template <typename COLOR3> OSL_HOSTDEVICE
static COLOR3
hsl_to_rgb (const COLOR3& hsl)
{
    using FLOAT = typename ScalarFromVec<COLOR3>::type;
    FLOAT h = comp_x(hsl), s = comp_y(hsl), l = comp_z(hsl);
    // Easiest to convert hsl -> hsv, then hsv -> RGB (per Foley & van Dam)
    FLOAT v = (l <= 0.5f) ? (l * (1.0f + s)) : (l * (1.0f - s) + s);
    if (v <= 0.0f) {
        return make_Color3 (0.0f, 0.0f, 0.0f);
    } else {
        FLOAT min = 2.0f * l - v;
        s = (v - min) / v;
        return hsv_to_rgb (make_Color3(h, s, v));
    }
}

template <typename COLOR3> OSL_HOSTDEVICE
static COLOR3
rgb_to_hsl (const COLOR3& rgb)
{
    // See Foley & van Dam
    // First convert rgb to hsv, then to hsl
    using FLOAT = typename ScalarFromVec<COLOR3>::type;
    FLOAT minval = sfm::min_val (comp_x(rgb), sfm::min_val (comp_y(rgb), comp_z(rgb)));
    COLOR3 hsv = rgb_to_hsv (rgb);
    FLOAT maxval = comp_z(hsv);   // v == maxval
    FLOAT h = comp_x(hsv), s, l = (minval+maxval) / 2.0f;
    if (equalVal (minval, maxval))
        s = 0.0f;  // special 'achromatic' case, hue is 0
    else if (l <= 0.5f)
        s = (maxval - minval) / (maxval + minval);
    else
        s = (maxval - minval) / (2.0f - maxval - minval);
    return make_Color3 (h, s, l);
}

template <typename COLOR3> OSL_HOSTDEVICE
static COLOR3
YIQ_to_rgb (const COLOR3& YIQ)
{
    return YIQ * Matrix33(1.0000,  1.0000,  1.0000,
                          0.9557, -0.2716, -1.1082,
                          0.6199, -0.6469,  1.7051);
}

template <typename COLOR3> OSL_HOSTDEVICE
static COLOR3
rgb_to_YIQ (const COLOR3& rgb)
{
    return rgb * Matrix33(0.299,  0.596,  0.212,
                          0.587, -0.275, -0.523,
                          0.114, -0.321,  0.311);
}

#if 0
OSL_HOSTDEVICE static inline Color3
XYZ_to_xyY (const Color3 &XYZ)
{
    float n = (XYZ[0] + XYZ[1] + XYZ[2]);
    float n_inv = (n >= 1.0e-6 ? 1.0f/n : 0.0f);
    return Color3 (XYZ[0]*n_inv, XYZ[1]*n_inv, XYZ[1]);
    // N.B. http://brucelindbloom.com/ suggests returning xy of the
    // reference white in the X+Y+Z==0 case.
}
#endif

template <typename COLOR3> OSL_HOSTDEVICE
static COLOR3
xyY_to_XYZ (const COLOR3 &xyY)
{
    using FLOAT = typename ScalarFromVec<COLOR3>::type;
    FLOAT Y = comp_z(xyY);
    FLOAT Y_y = (comp_y(xyY) > 1.0e-6f ? Y/comp_y(xyY) : 0.0f);
    FLOAT X = Y_y * comp_x(xyY);
    FLOAT Z = Y_y * (1.0f - comp_x(xyY) - comp_y(xyY));
    return make_Color3 (X, Y, Z);
}

template <typename COLOR3> OSL_HOSTDEVICE
static COLOR3
sRGB_to_linear (const COLOR3& srgb)
{
    // See Foley & van Dam
    using FLOAT = typename ScalarFromVec<COLOR3>::type;
    using namespace OIIO;
    //using safe_pow = std::conditional<is_Dual<COLOR3>::value, OSL::safe_pow, OIIO::safe_pow>::type;
    FLOAT r = comp_x(srgb), g = comp_y(srgb), b = comp_z(srgb);
    auto convert = [] (FLOAT x) -> FLOAT {
        return (x <= 0.04045f) ?
                     (x * (1.0f / 12.92f)) :
                     safe_pow((x + 0.055f) * (1.0f / 1.055f), FLOAT(2.4f));
    };
    return make_Color3 (convert(r), convert(g), convert(b));
}

template <typename COLOR3> OSL_HOSTDEVICE
static COLOR3
linear_to_sRGB (const COLOR3& rgb)
{
    // See Foley & van Dam
    using FLOAT = typename ScalarFromVec<COLOR3>::type;
    using namespace OIIO;
    //using safe_pow = std::conditional<is_Dual<COLOR3>::value, OSL::safe_pow, OIIO::safe_pow>::type;
    FLOAT r = comp_x(rgb), g = comp_y(rgb), b = comp_z(rgb);
    auto convert = [] (FLOAT x) -> FLOAT {
        return (x <= 0.0031308f) ?
                      (12.92f * x)           :
                      (1.055f * safe_pow(x, FLOAT(1.f / 2.4f)) - 0.055f);
    };

    return make_Color3 (convert(r), convert(g), convert(b));
}

OSL_HOSTDEVICE inline Color3
colpow (const Color3 &c, float p)
{
    return Color3 (powf(c.x,p), powf(c.y,p), powf(c.z,p));
}

}  // namespace __OSL_WIDE_PVT or pvt

namespace pvt {


#ifdef __OSL_WIDE_PVT
    using __OSL_WIDE_PVT::clamp_zero;
    using __OSL_WIDE_PVT::bb_spectrum;
    using __OSL_WIDE_PVT::spectrum_to_XYZ;
    using __OSL_WIDE_PVT::BB_TABLE_MAP;
    using __OSL_WIDE_PVT::BB_TABLE_UNMAP;
#endif


#ifdef __OSL_WIDE_PVT
OSL_HOSTDEVICE Color3
ColorSystem::blackbody_rgb (float T) const
{
    // Choose to access Color3 lookup table as 3 float components.
    // When vectorized, this helps generate individual gather instructions with
    // 32bit offsets with a common base address.  This is preferred to
    // dereferencing Color3 which was using 64bit gathers which can take
    // multiple registers to hold the 64bit offsets and multiple gather
    // instructions who's results need to be merged into a single register.
    const float *blackbody_components = reinterpret_cast<const float *>(m_blackbody_table);

    if (T < BB_DRAPER)
        return Color3(1.0e-6f,0.0f,0.0f);  // very very dim red
    if (T < BB_MAX_TABLE_RANGE) {
        float t = BB_TABLE_UNMAP(T);
        uint16_t ti = static_cast<uint16_t>(t);
        uint16_t ti_offset = static_cast<uint16_t>(ti*(sizeof(Color3)/sizeof(float)));
        float remainder = t - ti;
        // Color3 rgb = OIIO::lerp (m_blackbody_table[ti], m_blackbody_table[ti+1], t);
        Color3 rgb = OIIO::lerp (
            // Have all gathers use the same indices by having different base registers
            // for each of the 6 components
            Color3((blackbody_components+0)[ti_offset],
                   (blackbody_components+1)[ti_offset],
                   (blackbody_components+2)[ti_offset]),
            Color3((blackbody_components+3)[ti_offset],
                   (blackbody_components+4)[ti_offset],
                   (blackbody_components+5)[ti_offset]),
                   remainder);
        //return colpow(rgb, BB_TABLE_YPOWER);
        Color3 rgb2 = rgb * rgb;
        Color3 rgb4 = rgb2 * rgb2;
        return rgb4 * rgb; // ^5
    }

    // Otherwise, compute for real
    bb_spectrum spec (T);
    Color3 rgb = XYZ_to_RGB (spectrum_to_XYZ (spec));
    clamp_zero (rgb);
    return rgb;
}
#else
// TODO: Keeping original dereferencing Color3 vs. 3 floats until
// it can be determined if it is faster or not for scalar or GPU
OSL_HOSTDEVICE Color3
ColorSystem::blackbody_rgb (float T) const
{
    if (T < BB_DRAPER)
        return Color3(1.0e-6f,0.0f,0.0f);  // very very dim red
    if (T < BB_MAX_TABLE_RANGE) {
        float t = BB_TABLE_UNMAP(T);
        int ti = (int)t;
        t -= ti;
        Color3 rgb = OIIO::lerp (m_blackbody_table[ti], m_blackbody_table[ti+1], t);
        //return colpow(rgb, BB_TABLE_YPOWER);
        Color3 rgb2 = rgb * rgb;
        Color3 rgb4 = rgb2 * rgb2;
        return rgb4 * rgb; // ^5
    }
    // Otherwise, compute for real
    bb_spectrum spec (T);
    Color3 rgb = XYZ_to_RGB (spectrum_to_XYZ (spec));
    clamp_zero (rgb);
    return rgb;
}
#endif

OSL_HOSTDEVICE bool
ColorSystem::can_lookup_blackbody(float T /*Kelvin*/) const
{
    return (T < BB_MAX_TABLE_RANGE);
}

OSL_HOSTDEVICE Color3
ColorSystem::lookup_blackbody_rgb (float T /*Kelvin*/) const
{
    // Choose to access Color3 lookup table as 3 float components.
    // When vectorized, this helps generate individual gather instructions with
    // 32bit offsets with a common base address.  This is preferred to
    // dereferencing Color3 which was using 64bit gathers which can take
    // multiple registers to hold the 64bit offsets and multiple gather
    // instructions who's results need to be merged into a single register.
    const float *blackbody_components = reinterpret_cast<const float *>(m_blackbody_table);

    if (T < BB_DRAPER)
        return Color3(1.0e-6f,0.0f,0.0f);  // very very dim red

    // can_lookup_blackbody() should have been used
    OSL_DASSERT(T < BB_MAX_TABLE_RANGE);
    float t = BB_TABLE_UNMAP(T);
    uint16_t ti = static_cast<uint16_t>(t);
    uint16_t ti_offset = static_cast<uint16_t>(ti*(sizeof(Color3)/sizeof(float)));
    float remainder = t - ti;
    //Color3 rgb = OIIO::lerp (m_blackbody_table[ti], m_blackbody_table[ti+1], t);
    Color3 rgb = OIIO::lerp (
        // Have all gathers use the same indices by having different base registers
        // for each of the 6 components
        Color3((blackbody_components+0)[ti_offset],
               (blackbody_components+1)[ti_offset],
               (blackbody_components+2)[ti_offset]),
        Color3((blackbody_components+3)[ti_offset],
               (blackbody_components+4)[ti_offset],
               (blackbody_components+5)[ti_offset]),
       remainder);
    //return colpow(rgb, BB_TABLE_YPOWER);
    Color3 rgb2 = rgb * rgb;
    Color3 rgb4 = rgb2 * rgb2;
    return rgb4 * rgb; // ^5
}

OSL_HOSTDEVICE Color3
ColorSystem::compute_blackbody_rgb (float T /*Kelvin*/) const
{
    // compute for real
    bb_spectrum spec (T);
    Color3 rgb = XYZ_to_RGB (spectrum_to_XYZ (spec));
    clamp_zero (rgb);
    return rgb;
}


}  // namespace pvt
OSL_NAMESPACE_EXIT

