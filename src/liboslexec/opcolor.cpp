/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of color operations.
///
/////////////////////////////////////////////////////////////////////////

#include "oslexec_pvt.h"
#include <OSL/dual.h>
#include <OSL/dual_vec.h>
#include <OSL/Imathx.h>
#include <OSL/device_string.h>

#include <OpenImageIO/fmath.h>

#if defined(_MSC_VER) && _MSC_VER < 1800
using OIIO::expm1;
using OIIO::cbrtf;
#endif

#ifdef __CUDACC__
  #include <optix.h>
#endif

#include "opcolor.h"

OSL_NAMESPACE_ENTER

namespace pvt {

// CIE colour matching functions xBar, yBar, and zBar for
//   wavelengths from 380 through 780 nanometers, every 5
//   nanometers.  For a wavelength lambda in this range:
//        cie_colour_match[(lambda - 380) / 5][0] = xBar
//        cie_colour_match[(lambda - 380) / 5][1] = yBar
//        cie_colour_match[(lambda - 380) / 5][2] = zBar
OSL_CONSTANT_DATA const static float cie_colour_match[81][3] =
{
    {0.0014,0.0000,0.0065}, {0.0022,0.0001,0.0105}, {0.0042,0.0001,0.0201},
    {0.0076,0.0002,0.0362}, {0.0143,0.0004,0.0679}, {0.0232,0.0006,0.1102},
    {0.0435,0.0012,0.2074}, {0.0776,0.0022,0.3713}, {0.1344,0.0040,0.6456},
    {0.2148,0.0073,1.0391}, {0.2839,0.0116,1.3856}, {0.3285,0.0168,1.6230},
    {0.3483,0.0230,1.7471}, {0.3481,0.0298,1.7826}, {0.3362,0.0380,1.7721},
    {0.3187,0.0480,1.7441}, {0.2908,0.0600,1.6692}, {0.2511,0.0739,1.5281},
    {0.1954,0.0910,1.2876}, {0.1421,0.1126,1.0419}, {0.0956,0.1390,0.8130},
    {0.0580,0.1693,0.6162}, {0.0320,0.2080,0.4652}, {0.0147,0.2586,0.3533},
    {0.0049,0.3230,0.2720}, {0.0024,0.4073,0.2123}, {0.0093,0.5030,0.1582},
    {0.0291,0.6082,0.1117}, {0.0633,0.7100,0.0782}, {0.1096,0.7932,0.0573},
    {0.1655,0.8620,0.0422}, {0.2257,0.9149,0.0298}, {0.2904,0.9540,0.0203},
    {0.3597,0.9803,0.0134}, {0.4334,0.9950,0.0087}, {0.5121,1.0000,0.0057},
    {0.5945,0.9950,0.0039}, {0.6784,0.9786,0.0027}, {0.7621,0.9520,0.0021},
    {0.8425,0.9154,0.0018}, {0.9163,0.8700,0.0017}, {0.9786,0.8163,0.0014},
    {1.0263,0.7570,0.0011}, {1.0567,0.6949,0.0010}, {1.0622,0.6310,0.0008},
    {1.0456,0.5668,0.0006}, {1.0026,0.5030,0.0003}, {0.9384,0.4412,0.0002},
    {0.8544,0.3810,0.0002}, {0.7514,0.3210,0.0001}, {0.6424,0.2650,0.0000},
    {0.5419,0.2170,0.0000}, {0.4479,0.1750,0.0000}, {0.3608,0.1382,0.0000},
    {0.2835,0.1070,0.0000}, {0.2187,0.0816,0.0000}, {0.1649,0.0610,0.0000},
    {0.1212,0.0446,0.0000}, {0.0874,0.0320,0.0000}, {0.0636,0.0232,0.0000},
    {0.0468,0.0170,0.0000}, {0.0329,0.0119,0.0000}, {0.0227,0.0082,0.0000},
    {0.0158,0.0057,0.0000}, {0.0114,0.0041,0.0000}, {0.0081,0.0029,0.0000},
    {0.0058,0.0021,0.0000}, {0.0041,0.0015,0.0000}, {0.0029,0.0010,0.0000},
    {0.0020,0.0007,0.0000}, {0.0014,0.0005,0.0000}, {0.0010,0.0004,0.0000},
    {0.0007,0.0002,0.0000}, {0.0005,0.0002,0.0000}, {0.0003,0.0001,0.0000},
    {0.0002,0.0001,0.0000}, {0.0002,0.0001,0.0000}, {0.0001,0.0000,0.0000},
    {0.0001,0.0000,0.0000}, {0.0001,0.0000,0.0000}, {0.0000,0.0000,0.0000}
};

// White point chromaticities.
#define IlluminantC   0.3101, 0.3162          /* For NTSC television */
#define IlluminantD65 0.3127, 0.3291          /* For EBU and SMPTE */
#define IlluminantE   0.33333333, 0.33333333  /* CIE equal-energy illuminant */

OSL_CONSTANT_DATA const static ColorSystem::Chroma k_color_systems[11] = {
   // Index, Name       xRed    yRed   xGreen  yGreen   xBlue  yBlue    White point
   /* 0  Rec709    */ { 0.64,   0.33,   0.30,   0.60,   0.15,   0.06,   IlluminantD65 },
   /* 1  sRGB      */ { 0.64,   0.33,   0.30,   0.60,   0.15,   0.06,   IlluminantD65 },
   /* 2  NTSC      */ { 0.67,   0.33,   0.21,   0.71,   0.14,   0.08,   IlluminantC },
   /* 3  EBU       */ { 0.64,   0.33,   0.29,   0.60,   0.15,   0.06,   IlluminantD65 },
   /* 4  PAL       */ { 0.64,   0.33,   0.29,   0.60,   0.15,   0.06,   IlluminantD65 },
   /* 5  SECAM     */ { 0.64,   0.33,   0.29,   0.60,   0.15,   0.06,   IlluminantD65 },
   /* 6  SMPTE     */ { 0.630,  0.340,  0.310,  0.595,  0.155,  0.070,  IlluminantD65 },
   /* 7  HDTV      */ { 0.670,  0.330,  0.210,  0.710,  0.150,  0.060,  IlluminantD65 },
   /* 8  CIE       */ { 0.7355, 0.2645, 0.2658, 0.7243, 0.1669, 0.0085, IlluminantE },
   /* 9  AdobeRGB  */ { 0.64,   0.33,   0.21,   0.71,   0.15,   0.06,   IlluminantD65 },
   /* 10 XYZ       */ { 1.0,    0.0,    0.0,    1.0,    0.0,    0.0,    IlluminantE },
};


OSL_HOSTDEVICE const ColorSystem::Chroma*
ColorSystem::fromString(StringParam colorspace) {
    if (colorspace == StringParams::Rec709)
        return &k_color_systems[0];
    if (colorspace == StringParams::sRGB)
        return &k_color_systems[1];
    if (colorspace == StringParams::NTSC)
        return &k_color_systems[2];
    if (colorspace == StringParams::EBU)
        return &k_color_systems[3];
    if (colorspace == StringParams::PAL)
        return &k_color_systems[4];
    if (colorspace == StringParams::SECAM)
        return &k_color_systems[5];
    if (colorspace == StringParams::SMPTE)
        return &k_color_systems[6];
    if (colorspace == StringParams::HDTV)
        return &k_color_systems[7];
    if (colorspace == StringParams::CIE)
        return &k_color_systems[8];
    if (colorspace == StringParams::AdobeRGB)
        return &k_color_systems[9];
    if (colorspace == StringParams::XYZ)
        return &k_color_systems[10];
    return nullptr;
}



namespace {


template <typename COLOR3> OSL_HOSTDEVICE
static COLOR3
hsv_to_rgb (const COLOR3& hsv)
{
    // Reference for this technique: Foley & van Dam
    using FLOAT = typename ScalarFromVec<COLOR3>::type;
    FLOAT h = comp(hsv,0), s = comp(hsv,1), v = comp(hsv,2);
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
        switch (hi) {
        case 0 : return make_Color3 (v, t, p);
        case 1 : return make_Color3 (q, v, p);
        case 2 : return make_Color3 (p, v, t);
        case 3 : return make_Color3 (p, q, v);
        case 4 : return make_Color3 (t, p, v);
        default: return make_Color3 (v, p, q);
	    }
    }
}


template <typename COLOR3> OSL_HOSTDEVICE
static COLOR3
rgb_to_hsv (const COLOR3& rgb)
{
    // See Foley & van Dam
    using FLOAT = typename ScalarFromVec<COLOR3>::type;
    FLOAT r = comp(rgb,0), g = comp(rgb,1), b = comp(rgb,2);
    FLOAT mincomp = std::min (r, std::min (g, b));
    FLOAT maxcomp = std::max (r, std::max (g, b));
    FLOAT delta = maxcomp - mincomp;  // chroma
    FLOAT h, s, v;
    v = maxcomp;
    if (maxcomp > 0.0f)
        s = delta / maxcomp;
    else s = 0.0f;
    if (s <= 0.0f)
        h = 0.0f;
    else {
        if      (r >= maxcomp) h = (g-b) / delta;
        else if (g >= maxcomp) h = 2.0f + (b-r) / delta;
        else                   h = 4.0f + (r-g) / delta;
        h *= (1.0f/6.0f);
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
    FLOAT h = comp(hsl,0), s = comp(hsl,1), l = comp(hsl,2);
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
    FLOAT minval = std::min (comp(rgb,0), std::min (comp(rgb,1), comp(rgb,2)));
    COLOR3 hsv = rgb_to_hsv (rgb);
    FLOAT maxval = comp(hsv,2);   // v == maxval
    FLOAT h = comp(hsv,0), s, l = (minval+maxval) / 2.0f;
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
OSL_HOSTDEVICE inline Color3
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
    FLOAT Y = comp(xyY,2);
    FLOAT Y_y = (comp(xyY,1) > 1.0e-6f ? Y/comp(xyY,1) : 0.0f);
    FLOAT X = Y_y * comp(xyY,0);
    FLOAT Z = Y_y * (1.0f - comp(xyY,0) - comp(xyY,1));
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
    FLOAT r = comp(srgb,0), g = comp(srgb,1), b = comp(srgb,2);
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
    FLOAT r = comp(rgb,0), g = comp(rgb,1), b = comp(rgb,2);
    auto convert = [] (FLOAT x) -> FLOAT {
        return (x <= 0.0031308f) ?
                      (12.92f * x)           :
                      (1.055f * safe_pow(x, FLOAT(1.f / 2.4f)) - 0.055f);
    };

    return make_Color3 (convert(r), convert(g), convert(b));
}




// Spectral rendering routines inspired by those found at:
//   http://www.fourmilab.ch/documents/specrend/specrend.c
// which bore the notice:
//                Colour Rendering of Spectra
//                     by John Walker
//                  http://www.fourmilab.ch/
//		 Last updated: March 9, 2003
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
        double wlm = wavelength_nm * 1e-9;   // Wavelength in meters
        const double c1 = 3.74183e-16; // 2*pi*h*c^2, W*m^2
        const double c2 = 1.4388e-2;   // h*c/k, m*K
                                       // h is Planck's const, k is Boltzmann's
        return float((c1 * std::pow(wlm,-5.0)) / ::expm1(c2 / (wlm * m_temp)));
    }
private:
    double m_temp;
};



// For a given wavelength lambda (in nm), return the XYZ triple giving the
// XYZ color corresponding to that single wavelength;
OSL_HOSTDEVICE static Color3
wavelength_color_XYZ (float lambda_nm)
{
    float ii = (lambda_nm-380.0f) / 5.0f;  // scaled 0..80
    int i = (int) ii;
    if (i < 0 || i >= 80)
        return Color3(0.0f,0.0f,0.0f);
    ii -= i;
    const float *c = cie_colour_match[i];
    Color3 XYZ = OIIO::lerp (Color3(c[0], c[1], c[2]),
                             Color3(c[3], c[4], c[5]), ii);
#if 0
    float n = (XYZ[0] + XYZ[1] + XYZ[2]);
    float n_inv = (n >= 1.0e-6f ? 1.0f/n : 0.0f);
    XYZ *= n_inv;
#endif
    return XYZ;
}



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
        X += Me * cie_colour_match[i][0];
        Y += Me * cie_colour_match[i][1];
        Z += Me * cie_colour_match[i][2];
    }
    return Color3 (X, Y, Z);
}


#if 0
// If the requested RGB shade contains a negative weight for one of the
// primaries, it lies outside the colour gamut accessible from the given
// triple of primaries.  Desaturate it by adding white, equal quantities
// of R, G, and B, enough to make RGB all positive.  The function
// returns true if the components were modified, zero otherwise.
OSL_HOSTDEVICE inline bool
constrain_rgb (Color3 &rgb)
{
    // Amount of white needed is w = - min(0,r,g,b)
    float w = 0.0f;
    w = (0 < rgb[0]) ? w : rgb[0];
    w = (w < rgb[1]) ? w : rgb[1];
    w = (w < rgb[2]) ? w : rgb[2];
    w = -w;

    // Add just enough white to make r, g, b all positive.
    if (w > 0) {
        rgb[0] += w;  rgb[1] += w;  rgb[2] += w;
        return true;   // Color modified to fit RGB gamut
    }

    return false;  // color was within RGB gamut
}



// Rescale rgb so its largest component is 1.0, and return the original
// largest component.
OSL_HOSTDEVICE inline float
norm_rgb (Color3 &rgb)
{
    float greatest = std::max(rgb[0], std::max(rgb[1], rgb[2]));
    if (greatest > 1.0e-12f)
        rgb *= 1.0f/greatest;
    return greatest;
}
#endif


OSL_HOSTDEVICE inline void clamp_zero (Color3 &c)
{
    if (c[0] < 0.0f)
        c[0] = 0.0f;
    if (c[1] < 0.0f)
        c[1] = 0.0f;
    if (c[2] < 0.0f)
        c[2] = 0.0f;
}



OSL_HOSTDEVICE inline Color3
colpow (const Color3 &c, float p)
{
    return Color3 (powf(c[0],p), powf(c[1],p), powf(c[2],p));
}


};  // End anonymous namespace



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
    float ic = cbrtf(t);
    return ic * ic; // ^2/3
}


OSL_HOSTDEVICE bool
ColorSystem::set_colorspace (StringParam colorspace)
{
    if (colorspace == m_colorspace)
        return true;

    const Chroma* chroma = fromString(colorspace);
    if (!chroma)
        return false;

    // Record the current colorspace
    m_colorspace = colorspace;

    m_Red.setValue (chroma->xRed, chroma->yRed, 0.0f);
    m_Green.setValue (chroma->xGreen, chroma->yGreen, 0.0f);
    m_Blue.setValue (chroma->xBlue, chroma->yBlue, 0.0f);
    m_White.setValue (chroma->xWhite, chroma->yWhite, 0.0f);
    // set z values to normalize
    m_Red[2]   = 1.0f - (m_Red[0]   + m_Red[1]);
    m_Green[2] = 1.0f - (m_Green[0] + m_Green[1]);
    m_Blue[2]  = 1.0f - (m_Blue[0]  + m_Blue[1]);
    m_White[2] = 1.0f - (m_White[0] + m_White[1]);

    const Color3 &R(m_Red), &G(m_Green), &B(m_Blue), &W(m_White);
    // xyz -> rgb matrix, before scaling to white.
    Color3 r (G[1]*B[2] - B[1]*G[2], B[0]*G[2] - G[0]*B[2], G[0]*B[1] - B[0]*G[1]);
    Color3 g (B[1]*R[2] - R[1]*B[2], R[0]*B[2] - B[0]*R[2], B[0]*R[1] - R[0]*B[1]);
    Color3 b (R[1]*G[2] - G[1]*R[2], G[0]*R[2] - R[0]*G[2], R[0]*G[1] - G[0]*R[1]);
    Color3 w (r.dot(W), g.dot(W), b.dot(W));  // White scaling factor
    if (W[1] != 0.0f)  // divide by W[1] to scale luminance to 1.0
        w *= 1.0f/W[1];
    // xyz -> rgb matrix, correctly scaled to white.
    r /= w[0];
    g /= w[1];
    b /= w[2];
    m_XYZ2RGB = Matrix33 (r[0], g[0], b[0],
                          r[1], g[1], b[1],
                          r[2], g[2], b[2]);
    m_RGB2XYZ = m_XYZ2RGB.inverse();
    m_luminance_scale = Color3 (m_RGB2XYZ[0][1], m_RGB2XYZ[1][1], m_RGB2XYZ[2][1]);

    // Mathematical imprecision can lead to the luminance scale not
    // quite summing to 1.0.  If it's very close, adjust to make it
    // exact.
    float lum2 = (1.0f - m_luminance_scale[0] - m_luminance_scale[1]);
    if (fabsf(lum2 - m_luminance_scale[2]) < 0.001f)
        m_luminance_scale[2] = lum2;

    // Precompute a table of blackbody values
    // FIXME: With c++14 and constexpr cbrtf, this could be static_assert
    assert( std::ceil(BB_TABLE_UNMAP(BB_MAX_TABLE_RANGE)) <
            std::extent<decltype(m_blackbody_table)>::value);

    float lastT = 0;
    for (int i = 0;  lastT <= BB_MAX_TABLE_RANGE; ++i) {
        float T = BB_TABLE_MAP(float(i));
        lastT = T;
        bb_spectrum spec (T);
        Color3 rgb = XYZ_to_RGB (spectrum_to_XYZ (spec));
        clamp_zero (rgb);
        rgb = colpow (rgb, 1.0f/BB_TABLE_YPOWER);
        m_blackbody_table[i] = rgb;
        // std::cout << "Table[" << i << "; T=" << T << "] = " << rgb << "\n";
    }

#if 0 && !defined(__CUDACC__)
    std::cout << "Made " << m_blackbody_table.size() << " table entries for blackbody\n";

    // Sanity checks
    std::cout << "m_XYZ2RGB = " << m_XYZ2RGB << "\n";
    std::cout << "m_RGB2XYZ = " << m_RGB2XYZ << "\n";
    std::cout << "m_luminance_scale = " << m_luminance_scale << "\n";
#endif
    return true;
}

// For Optix, this will be defined by the renderer. Otherwise inline a getter.
#ifdef __CUDACC__
    extern "C"  __device__
    void osl_printf (void* sg_, char* fmt_str, void* args) __attribute__((weak));

    extern "C" __device__ int
    rend_get_userdata (StringParam name, void* data, int data_size,
                       const OSL::TypeDesc& type, int index) __attribute__((weak));

    OSL_HOSTDEVICE void
    ColorSystem::error(StringParam src, StringParam dst, Context sg) {
        const char* args[2] = { src.c_str(), dst.c_str() };
        osl_printf (sg,
            (char*) // FIXME!
            "ERROR: Unknown color space transformation \"%s\" -> \"%s\"\n",
            args);
    }

    __device__ static inline ColorSystem& op_color_colorsystem (void *sg) {
        void* ptr;
        rend_get_userdata(StringParams::colorsystem, &ptr, 8, OSL::TypeDesc::PTR, 0);
        return *((ColorSystem*)ptr);
    }

    __device__ static inline void* op_color_context (void *sg) {
        return sg;
    }
#else
    void
    ColorSystem::error(StringParam src, StringParam dst, Context context) {
        context->error("Unknown color space transformation"
                       " \"%s\" -> \"%s\"", src, dst);
    }

    static inline ColorSystem& op_color_colorsystem (void *sg) {
        return ((ShaderGlobals *)sg)->context->shadingsys().colorsystem();
    }

    static inline OSL::ShadingContext* op_color_context (void *sg) {
        return (ShadingContext*)((ShaderGlobals *)sg)->context;
    }
#endif



template <typename Color> OSL_HOSTDEVICE Color
ColorSystem::ocio_transform (StringParam fromspace, StringParam tospace,
                             const Color& C, Context ctx)
{
#if OIIO_HAS_COLORPROCESSOR
    Color Cout;
    if (ctx->shadingsys().ocio_transform(fromspace, tospace, C, Cout))
        return Cout;
#endif

    error (fromspace, tospace, ctx);
    return C;
}



OSL_HOSTDEVICE Color3
ColorSystem::to_rgb (StringParam fromspace, const Color3& C, Context context)
{
    if (fromspace == StringParams::RGB || fromspace == StringParams::rgb
         || fromspace == m_colorspace)
        return C;
    if (fromspace == StringParams::hsv)
        return hsv_to_rgb (C);
    if (fromspace == StringParams::hsl)
        return hsl_to_rgb (C);
    if (fromspace == StringParams::YIQ)
        return YIQ_to_rgb (C);
    if (fromspace == StringParams::XYZ)
        return XYZ_to_RGB (C);
    if (fromspace == StringParams::xyY)
        return XYZ_to_RGB (xyY_to_XYZ (C));
    else
        return ocio_transform (fromspace, StringParams::RGB, C, context);
}



OSL_HOSTDEVICE Color3
ColorSystem::from_rgb (StringParam tospace, const Color3& C, Context context)
{
    if (tospace == StringParams::RGB || tospace == StringParams::rgb
         || tospace == m_colorspace)
        return C;
    if (tospace == StringParams::hsv)
        return rgb_to_hsv (C);
    if (tospace == StringParams::hsl)
        return rgb_to_hsl (C);
    if (tospace == StringParams::YIQ)
        return rgb_to_YIQ (C);
    if (tospace == StringParams::XYZ)
        return RGB_to_XYZ (C);
    if (tospace == StringParams::xyY)
        return RGB_to_XYZ (xyY_to_XYZ (C));
    else
        return ocio_transform (StringParams::RGB, tospace, C, context);
}



template <typename COLOR> OSL_HOSTDEVICE COLOR
ColorSystem::transformc (StringParam fromspace, StringParam tospace,
                         const COLOR& C, Context context)
{
    bool use_colorconfig = false;
    COLOR Crgb;
    if (fromspace == StringParams::RGB || fromspace == StringParams::rgb
         || fromspace == m_colorspace)
        Crgb = C;
    else if (fromspace == StringParams::hsv)
        Crgb = hsv_to_rgb (C);
    else if (fromspace == StringParams::hsl)
        Crgb = hsl_to_rgb (C);
    else if (fromspace == StringParams::YIQ)
        Crgb = YIQ_to_rgb (C);
    else if (fromspace == StringParams::XYZ)
        Crgb = XYZ_to_RGB (C);
    else if (fromspace == StringParams::xyY)
        Crgb = XYZ_to_RGB (xyY_to_XYZ (C));
    else if (fromspace == StringParams::sRGB)
        Crgb = sRGB_to_linear (C);
    else {
        use_colorconfig = true;
    }

    COLOR Cto;
    if (use_colorconfig) {
        // do things the ColorConfig way, so skip all these other clauses...
    }
    else if (tospace == StringParams::RGB || tospace == StringParams::rgb
         || tospace == m_colorspace)
        Cto = Crgb;
    else if (tospace == StringParams::hsv)
        Cto = rgb_to_hsv (Crgb);
    else if (tospace == StringParams::hsl)
        Cto = rgb_to_hsl (Crgb);
    else if (tospace == StringParams::YIQ)
        Cto = rgb_to_YIQ (Crgb);
    else if (tospace == StringParams::XYZ)
        Cto = RGB_to_XYZ (Crgb);
    else if (tospace == StringParams::xyY)
        Cto = RGB_to_XYZ (xyY_to_XYZ (Crgb));
    else if (tospace == StringParams::sRGB)
        Cto = linear_to_sRGB (Crgb);
    else {
        use_colorconfig = true;
    }

    if (use_colorconfig) {
        Cto = ocio_transform (fromspace, tospace, C, context);
    }

    return Cto;
}



OSL_HOSTDEVICE Dual2<Color3>
ColorSystem::transformc (StringParam fromspace, StringParam tospace,
                         const Dual2<Color3>& color, Context ctx) {
    return transformc<Dual2<Color3>>(fromspace, tospace, color, ctx);
}



OSL_HOSTDEVICE Color3
ColorSystem::transformc (StringParam fromspace, StringParam tospace,
                         const Color3& color, Context ctx) {
    return transformc<Color3>(fromspace, tospace, color, ctx);
}



OSL_HOSTDEVICE Color3
ColorSystem::blackbody_rgb (float T)
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



} // namespace pvt



OSL_SHADEOP OSL_HOSTDEVICE void osl_blackbody_vf (void *sg, void *out, float temp)
{
    ColorSystem &cs = op_color_colorsystem(sg);
    *(Color3 *)out = cs.blackbody_rgb (temp);
}



OSL_SHADEOP OSL_HOSTDEVICE void osl_wavelength_color_vf (void *sg, void *out, float lambda)
{
    ColorSystem &cs = op_color_colorsystem(sg);
    Color3 rgb = cs.XYZ_to_RGB (wavelength_color_XYZ (lambda));
//    constrain_rgb (rgb);
    rgb *= 1.0/2.52;    // Empirical scale from lg to make all comps <= 1
//    norm_rgb (rgb);
    clamp_zero (rgb);
    *(Color3 *)out = rgb;
}



OSL_SHADEOP OSL_HOSTDEVICE void osl_luminance_fv (void *sg, void *out, void *c)
{
    ColorSystem &cs = op_color_colorsystem(sg);
    ((float *)out)[0] = cs.luminance (((const Color3 *)c)[0]);
}



OSL_SHADEOP OSL_HOSTDEVICE void osl_luminance_dfdv (void *sg, void *out, void *c)
{
    ColorSystem &cs = op_color_colorsystem(sg);
    ((float *)out)[0] = cs.luminance (((const Color3 *)c)[0]);
    ((float *)out)[1] = cs.luminance (((const Color3 *)c)[1]);
    ((float *)out)[2] = cs.luminance (((const Color3 *)c)[2]);
}



OSL_SHADEOP OSL_HOSTDEVICE void
osl_prepend_color_from (void *sg, void *c_, const char *from)
{
    ColorSystem &cs = op_color_colorsystem(sg);
    COL(c_) = cs.to_rgb (HDSTR(from), COL(c_), op_color_context(sg));
}



OSL_SHADEOP OSL_HOSTDEVICE int
osl_transformc (void *sg, void *Cin, int Cin_derivs,
                void *Cout, int Cout_derivs,
                void *from_, void *to_)
{
    ColorSystem &cs = op_color_colorsystem(sg);
    StringParam from = HDSTR(from_);
    StringParam to = HDSTR(to_);

    if (Cout_derivs) {
        if (Cin_derivs) {
            DCOL(Cout) = cs.transformc (from, to, DCOL(Cin), op_color_context(sg));
            return true;
        } else {
            // We had output derivs, but not input. Zero the output
            // derivs and fall through to the non-deriv case.
            ((Color3 *)Cout)[1].setValue (0.0f, 0.0f, 0.0f);
            ((Color3 *)Cout)[2].setValue (0.0f, 0.0f, 0.0f);
        }
    }

    // No-derivs case
    COL(Cout) = cs.transformc (from, to, COL(Cin), op_color_context(sg));
    return true;
}



OSL_NAMESPACE_EXIT
