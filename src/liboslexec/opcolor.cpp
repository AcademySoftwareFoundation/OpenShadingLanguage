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

#include <OpenImageIO/fmath.h>

#include <iostream>
#include <cmath>

#include "oslexec_pvt.h"
#include "dual.h"

#ifdef _MSC_VER
using OIIO::expm1;
#endif

OSL_NAMESPACE_ENTER
namespace pvt {

// This symbol is strictly to force linkage of this file when building
// static library.
int opcolor_cpp_dummy = 1;


namespace {


static Color3
hsv_to_rgb (float h, float s, float v)
{
    // Reference for this technique: Foley & van Dam
    if (s < 0.0001f) {
      return Color3 (v, v, v);
    } else {
        h = 6.0f * (h - floorf(h));  // expand to [0..6)
        int hi = (int) h;
        float f = h - hi;
        float p = v * (1.0f-s);
        float q = v * (1.0f-s*f);
        float t = v * (1.0f-s*(1.0f-f));
        switch (hi) {
        case 0 : return Color3 (v, t, p);
        case 1 : return Color3 (q, v, p);
        case 2 : return Color3 (p, v, t);
        case 3 : return Color3 (p, q, v);
        case 4 : return Color3 (t, p, v);
        default: return Color3 (v, p, q);
	}
    }
}



static Color3
hsl_to_rgb (float h, float s, float l)
{
    // Easiest to convert hsl -> hsv, then hsv -> RGB (per Foley & van Dam)
    float v = (l <= 0.5) ? (l * (1.0f + s)) : (l * (1.0f - s) + s);
    if (v <= 0.0f) {
        return Color3 (0.0f, 0.0f, 0.0f);
    } else {
	float min = 2.0f * l - v;
	s = (v - min) / v;
	return hsv_to_rgb (h, s, v);
    }
}



static Color3
YIQ_to_rgb (float Y, float I, float Q)
{
    return Color3 (Y + 0.9557f * I + 0.6199f * Q,
                   Y - 0.2716f * I - 0.6469f * Q,
                   Y - 1.1082f * I + 1.7051f * Q);
}


#if 0
inline Color3
XYZ_to_xyY (const Color3 &XYZ)
{
    float n = (XYZ[0] + XYZ[1] + XYZ[2]);
    float n_inv = (n >= 1.0e-6 ? 1.0f/n : 0.0f);
    return Color3 (XYZ[0]*n_inv, XYZ[1]*n_inv, XYZ[1]);
    // N.B. http://brucelindbloom.com/ suggests returning xy of the
    // reference white in the X+Y+Z==0 case.
}
#endif


inline Color3
xyY_to_XYZ (const Color3 &xyY)
{
    float Y = xyY[2];
    float Y_y = (xyY[1] > 1.0e-6 ? Y/xyY[1] : 0.0f);
    float X = Y_y * xyY[0];
    float Z = Y_y * (1.0f - xyY[0] - xyY[1]);
    return Color3 (X, Y, Z);
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
    bb_spectrum (float temperature=5000) : m_temp(temperature) { }
    float operator() (float wavelength_nm) const {
        double wlm = wavelength_nm * 1e-9;   // Wavelength in meters
        const double c1 = 3.74183e-16; // 2*pi*h*c^2, W*m^2
        const double c2 = 1.4388e-2;   // h*c/k, m*K
                                       // h is Planck's const, k is Boltzmann's
        return float((c1 * std::pow(wlm,-5.0)) / ::expm1(c2 / (wlm * m_temp)));
    }
private:
    double m_temp;
};



// CIE colour matching functions xBar, yBar, and zBar for
//   wavelengths from 380 through 780 nanometers, every 5
//   nanometers.  For a wavelength lambda in this range:
//        cie_colour_match[(lambda - 380) / 5][0] = xBar
//        cie_colour_match[(lambda - 380) / 5][1] = yBar
//        cie_colour_match[(lambda - 380) / 5][2] = zBar
static float cie_colour_match[81][3] = {
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



// For a given wavelength lambda (in nm), return the XYZ triple giving the
// XYZ color corresponding to that single wavelength;
static Color3
wavelength_color_XYZ (float lambda_nm)
{
    float ii = (lambda_nm-380.0f) / 5.0f;  // scaled 0..80
    int i = (int) ii;
    if (i < 0 || i >= 80)
        return Color3(0.0f,0.0f,0.0f);
    ii -= i;
    const float *c = cie_colour_match[i];
    Color3 XYZ = lerp (Color3(c[0], c[1], c[2]),
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
template<class SPECTRUM>
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
inline bool
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
inline float
norm_rgb (Color3 &rgb)
{
    float greatest = std::max(rgb[0], std::max(rgb[1], rgb[2]));
    if (greatest > 1.0e-12f)
        rgb *= 1.0f/greatest;
    return greatest;
}
#endif


inline void clamp_zero (Color3 &c)
{
    if (c[0] < 0.0f)
        c[0] = 0.0f;
    if (c[1] < 0.0f)
        c[1] = 0.0f;
    if (c[2] < 0.0f)
        c[2] = 0.0f;
}



inline Color3
colpow (const Color3 &c, float p)
{
    return Color3 (powf(c[0],p), powf(c[1],p), powf(c[2],p));
}


};  // End anonymous namespace


// A colour system is defined by the CIE x and y coordinates of its
// three primary illuminants and its white point.
struct colorSystem {
    const char *name;
    float  xRed, yRed,
           xGreen, yGreen,
           xBlue, yBlue,
           xWhite, yWhite;
};

// White point chromaticities.
#define IlluminantC   0.3101, 0.3162          /* For NTSC television */
#define IlluminantD65 0.3127, 0.3291          /* For EBU and SMPTE */
#define IlluminantE   0.33333333, 0.33333333  /* CIE equal-energy illuminant */


static colorSystem colorSystems[] = {
   // Name      xRed    yRed   xGreen  yGreen   xBlue  yBlue    White point
   { "Rec709",  0.64,   0.33,   0.30,   0.60,   0.15,   0.06,   IlluminantD65 },
   { "sRGB",    0.64,   0.33,   0.30,   0.60,   0.15,   0.06,   IlluminantD65 },
   { "NTSC",    0.67,   0.33,   0.21,   0.71,   0.14,   0.08,   IlluminantC },
   { "EBU",     0.64,   0.33,   0.29,   0.60,   0.15,   0.06,   IlluminantD65 },
   { "PAL",     0.64,   0.33,   0.29,   0.60,   0.15,   0.06,   IlluminantD65 },
   { "SECAM",   0.64,   0.33,   0.29,   0.60,   0.15,   0.06,   IlluminantD65 },
   { "SMPTE",   0.630,  0.340,  0.310,  0.595,  0.155,  0.070,  IlluminantD65 },
   { "HDTV",    0.670,  0.330,  0.210,  0.710,  0.150,  0.060,  IlluminantD65 },
   { "CIE",     0.7355, 0.2645, 0.2658, 0.7243, 0.1669, 0.0085, IlluminantE },
   { "AdobeRGB",0.64,   0.33,   0.21,   0.71,   0.15,   0.06,   IlluminantD65 },
   { "XYZ",     1.0,    0.0,    0.0,    1.0,    0.0,    0.0,    IlluminantE },
   { NULL }
};



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
#define BB_TABLE_XPOWER 1.5f
#define BB_TABLE_YPOWER 5.0f
#define BB_TABLE_SPACING 2.0f



bool
ShadingSystemImpl::set_colorspace (ustring colorspace)
{
    for (int i = 0;  colorSystems[i].name;  ++i) {
        if (colorspace == colorSystems[i].name) {
            m_Red.setValue (colorSystems[i].xRed, colorSystems[i].yRed, 0.0f);
            m_Green.setValue (colorSystems[i].xGreen, colorSystems[i].yGreen, 0.0f);
            m_Blue.setValue (colorSystems[i].xBlue, colorSystems[i].yBlue, 0.0f);
            m_White.setValue (colorSystems[i].xWhite, colorSystems[i].yWhite, 0.0f);
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
            m_blackbody_table.clear ();
            float lastT = 0;
            for (int i = 0;  lastT <= BB_MAX_TABLE_RANGE;  ++i) {
                float T = powf (float(i), BB_TABLE_XPOWER) * BB_TABLE_SPACING + BB_DRAPER;
                lastT = T;
                bb_spectrum spec (T);
                Color3 rgb = XYZ_to_RGB (spectrum_to_XYZ (spec));
                clamp_zero (rgb);
                rgb = colpow (rgb, 1.0f/BB_TABLE_YPOWER);
                m_blackbody_table.push_back (rgb);
                // std::cout << "Table[" << i << "; T=" << T << "] = " << rgb << "\n";
            }
            // std::cout << "Made " << m_blackbody_table.size() << " table entries for blackbody\n";

#if 0
            // Sanity checks
            std::cout << "m_XYZ2RGB = " << m_XYZ2RGB << "\n";
            std::cout << "m_RGB2XYZ = " << m_RGB2XYZ << "\n";
            std::cout << "m_luminance_scale = " << m_luminance_scale << "\n";
#endif
            return true;
        }
    }
    return false;
}



Color3
ShadingSystemImpl::to_rgb (ustring fromspace, float a, float b, float c)
{
    if (fromspace == Strings::RGB || fromspace == Strings::rgb)
        return Color3 (a, b, c);
    if (fromspace == Strings::hsv)
        return hsv_to_rgb (a, b, c);
    if (fromspace == Strings::hsl)
        return hsl_to_rgb (a, b, c);
    if (fromspace == Strings::YIQ)
        return YIQ_to_rgb (a, b, c);
    if (fromspace == Strings::XYZ)
        return XYZ_to_RGB (a, b, c);
    if (fromspace == Strings::xyY)
        return XYZ_to_RGB (xyY_to_XYZ (Color3(a,b,c)));
    error ("Unknown color space \"%s\"", fromspace.c_str());
    return Color3 (a, b, c);
}



Color3
ShadingSystemImpl::blackbody_rgb (float T)
{
    if (T < BB_DRAPER)
        return Color3(1.0e-6f,0.0f,0.0f);  // very very dim red
    if (T < BB_MAX_TABLE_RANGE) {
        float t = powf ((T - BB_DRAPER) / BB_TABLE_SPACING, 1.0f/BB_TABLE_XPOWER);
        int ti = (int)t;
        t -= ti;
        Color3 rgb = lerp (m_blackbody_table[ti], m_blackbody_table[ti+1], t);
        return colpow(rgb, BB_TABLE_YPOWER);
    }
    // Otherwise, compute for real
    bb_spectrum spec (T);
    Color3 rgb = XYZ_to_RGB (spectrum_to_XYZ (spec));
    clamp_zero (rgb);
    return rgb;
}



OSL_SHADEOP void osl_blackbody_vf (void *sg, void *out, float temp)
{
    ShadingContext *ctx = (ShadingContext *)((ShaderGlobals *)sg)->context;
    *(Color3 *)out = ctx->shadingsys().blackbody_rgb (temp);
}



OSL_SHADEOP void osl_wavelength_color_vf (void *sg, void *out, float lambda)
{
    ShadingContext *ctx = (ShadingContext *)((ShaderGlobals *)sg)->context;
    Color3 rgb = ctx->shadingsys().XYZ_to_RGB (wavelength_color_XYZ (lambda));
//    constrain_rgb (rgb);
    rgb *= 1.0/2.52;    // Empirical scale from lg to make all comps <= 1
//    norm_rgb (rgb);
    clamp_zero (rgb);
    *(Color3 *)out = rgb;
}



OSL_SHADEOP void osl_luminance_fv (void *sg, void *out, void *c)
{
    ShadingContext *ctx = (ShadingContext *)((ShaderGlobals *)sg)->context;
    ((float *)out)[0] = ctx->shadingsys().luminance (((const Color3 *)c)[0]);
}



OSL_SHADEOP void osl_luminance_dfdv (void *sg, void *out, void *c)
{
    ShadingContext *ctx = (ShadingContext *)((ShaderGlobals *)sg)->context;
    ((float *)out)[0] = ctx->shadingsys().luminance (((const Color3 *)c)[0]);
    ((float *)out)[1] = ctx->shadingsys().luminance (((const Color3 *)c)[1]);
    ((float *)out)[2] = ctx->shadingsys().luminance (((const Color3 *)c)[2]);
}



#define USTR(cstr) (*((ustring *)&cstr))

OSL_SHADEOP void
osl_prepend_color_from (void *sg, void *c_, const char *from)
{
    ShadingContext *ctx (((ShaderGlobals *)sg)->context);
    Color3 &c (*(Color3*)c_);
    c = ctx->shadingsys().to_rgb (USTR(from), c[0], c[1], c[2]);
}


} // namespace pvt
OSL_NAMESPACE_EXIT
