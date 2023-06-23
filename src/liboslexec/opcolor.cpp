// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of color operations.
///
/////////////////////////////////////////////////////////////////////////

#include "oslexec_pvt.h"
#include <OSL/Imathx/Imathx.h>
#include <OSL/device_string.h>
#include <OSL/dual.h>
#include <OSL/dual_vec.h>
#include <OSL/fmt_util.h>

#include <OpenImageIO/fmath.h>

#include "opcolor.h"

#include "opcolor_impl.h"

OSL_NAMESPACE_ENTER

namespace pvt {

// clang-format off

// White point chromaticities.
#define IlluminantC    0.3101, 0.3162          /* For NTSC television */
#define IlluminantD65  0.3127, 0.3291          /* For EBU and SMPTE */
#define IlluminantE    0.33333333, 0.33333333  /* CIE equal-energy illuminant */
#define IlluminantACES 0.32168, 0.33767        /* For ACES, approximate D60 */

namespace {  // anon namespace to avoid duplicate OptiX symbols
OSL_CONSTANT_DATA const ColorSystem::Chroma k_color_systems[13] = {
   // Index, Name        xRed    yRed   xGreen  yGreen   xBlue   yBlue    White point
   /* 0  Rec709     */ { 0.64,   0.33,   0.30,   0.60,   0.15,   0.06,   IlluminantD65 },
   /* 1  sRGB       */ { 0.64,   0.33,   0.30,   0.60,   0.15,   0.06,   IlluminantD65 },
   /* 2  NTSC       */ { 0.67,   0.33,   0.21,   0.71,   0.14,   0.08,   IlluminantC },
   /* 3  EBU        */ { 0.64,   0.33,   0.29,   0.60,   0.15,   0.06,   IlluminantD65 },
   /* 4  PAL        */ { 0.64,   0.33,   0.29,   0.60,   0.15,   0.06,   IlluminantD65 },
   /* 5  SECAM      */ { 0.64,   0.33,   0.29,   0.60,   0.15,   0.06,   IlluminantD65 },
   /* 6  SMPTE      */ { 0.630,  0.340,  0.310,  0.595,  0.155,  0.070,  IlluminantD65 },
   /* 7  HDTV       */ { 0.670,  0.330,  0.210,  0.710,  0.150,  0.060,  IlluminantD65 },
   /* 8  CIE        */ { 0.7355, 0.2645, 0.2658, 0.7243, 0.1669, 0.0085, IlluminantE },
   /* 9  AdobeRGB   */ { 0.64,   0.33,   0.21,   0.71,   0.15,   0.06,   IlluminantD65 },
   /* 10 XYZ        */ { 1.0,    0.0,    0.0,    1.0,    0.0,    0.0,    IlluminantE },
   /* 11 ACES2065-1 */ { 0.7347, 0.2653, 0.0,    1.0,    0.0001, -0.077, IlluminantACES },
   /* 12 ACEScg     */ { 0.713,  0.293,  0.165,  0.83,   0.128,  0.044,  IlluminantACES },
};
}  // namespace

// clang-format on



OSL_HOSTDEVICE const ColorSystem::Chroma*
ColorSystem::fromString(StringParam colorspace)
{
    if (colorspace == STRING_PARAMS(Rec709))
        return &k_color_systems[0];
    if (colorspace == STRING_PARAMS(sRGB))
        return &k_color_systems[1];
    if (colorspace == STRING_PARAMS(NTSC))
        return &k_color_systems[2];
    if (colorspace == STRING_PARAMS(EBU))
        return &k_color_systems[3];
    if (colorspace == STRING_PARAMS(PAL))
        return &k_color_systems[4];
    if (colorspace == STRING_PARAMS(SECAM))
        return &k_color_systems[5];
    if (colorspace == STRING_PARAMS(SMPTE))
        return &k_color_systems[6];
    if (colorspace == STRING_PARAMS(HDTV))
        return &k_color_systems[7];
    if (colorspace == STRING_PARAMS(CIE))
        return &k_color_systems[8];
    if (colorspace == STRING_PARAMS(AdobeRGB))
        return &k_color_systems[9];
    if (colorspace == STRING_PARAMS(XYZ))
        return &k_color_systems[10];
    if (colorspace == STRING_PARAMS(ACES2065_1))
        return &k_color_systems[11];
    if (colorspace == STRING_PARAMS(ACEScg))
        return &k_color_systems[12];
    return nullptr;
}



namespace {


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
    w = (0 < rgb.x) ? w : rgb.x;
    w = (w < rgb.y) ? w : rgb.y;
    w = (w < rgb.z) ? w : rgb.z;
    w = -w;

    // Add just enough white to make r, g, b all positive.
    if (w > 0) {
        rgb.x += w;  rgb.y += w;  rgb.z += w;
        return true;   // Color modified to fit RGB gamut
    }

    return false;  // color was within RGB gamut
}



// Rescale rgb so its largest component is 1.0, and return the original
// largest component.
OSL_HOSTDEVICE inline float
norm_rgb (Color3 &rgb)
{
    float greatest = std::max(rgb.x, std::max(rgb.y, rgb.z));
    if (greatest > 1.0e-12f)
        rgb *= 1.0f/greatest;
    return greatest;
}
#endif

};  // End anonymous namespace



OSL_HOSTDEVICE bool
ColorSystem::set_colorspace(StringParam colorspace)
{
    if (colorspace == m_colorspace)
        return true;

    const Chroma* chroma = fromString(colorspace);
    if (!chroma)
        return false;

    // Record the current colorspace
    m_colorspace = colorspace;

    m_Red.setValue(chroma->xRed, chroma->yRed, 0.0f);
    m_Green.setValue(chroma->xGreen, chroma->yGreen, 0.0f);
    m_Blue.setValue(chroma->xBlue, chroma->yBlue, 0.0f);
    m_White.setValue(chroma->xWhite, chroma->yWhite, 0.0f);
    // set z values to normalize
    m_Red.z   = 1.0f - (m_Red.x + m_Red.y);
    m_Green.z = 1.0f - (m_Green.x + m_Green.y);
    m_Blue.z  = 1.0f - (m_Blue.x + m_Blue.y);
    m_White.z = 1.0f - (m_White.x + m_White.y);

    const Color3 &R(m_Red), &G(m_Green), &B(m_Blue), &W(m_White);
    // xyz -> rgb matrix, before scaling to white.
    Color3 r(G.y * B.z - B.y * G.z, B.x * G.z - G.x * B.z,
             G.x * B.y - B.x * G.y);
    Color3 g(B.y * R.z - R.y * B.z, R.x * B.z - B.x * R.z,
             B.x * R.y - R.x * B.y);
    Color3 b(R.y * G.z - G.y * R.z, G.x * R.z - R.x * G.z,
             R.x * G.y - G.x * R.y);
    Color3 w(r.dot(W), g.dot(W), b.dot(W));  // White scaling factor
    if (W.y != 0.0f)  // divide by W.y to scale luminance to 1.0
        w *= 1.0f / W.y;
    // xyz -> rgb matrix, correctly scaled to white.
    r /= w.x;
    g /= w.y;
    b /= w.z;
    m_XYZ2RGB         = Matrix33(r.x, g.x, b.x, r.y, g.y, b.y, r.z, g.z, b.z);
    m_RGB2XYZ         = m_XYZ2RGB.inverse();
    m_luminance_scale = Color3(m_RGB2XYZ.x[0][1], m_RGB2XYZ.x[1][1],
                               m_RGB2XYZ.x[2][1]);

    // Mathematical imprecision can lead to the luminance scale not
    // quite summing to 1.0.  If it's very close, adjust to make it
    // exact.
    float lum2 = (1.0f - m_luminance_scale.x - m_luminance_scale.y);
    if (fabsf(lum2 - m_luminance_scale.z) < 0.001f)
        m_luminance_scale.z = lum2;

    // Precompute a table of blackbody values
    // FIXME: With c++14 and constexpr cbrtf, this could be static_assert
    assert(std::ceil(BB_TABLE_UNMAP(BB_MAX_TABLE_RANGE))
           < std::extent<decltype(m_blackbody_table)>::value);

    float lastT = 0;
    for (int i = 0; lastT <= BB_MAX_TABLE_RANGE; ++i) {
        float T = BB_TABLE_MAP(float(i));
        lastT   = T;
        bb_spectrum spec(T);
        Color3 rgb = XYZ_to_RGB(spectrum_to_XYZ(spec));
        clamp_zero(rgb);
        rgb                  = colpow(rgb, 1.0f / BB_TABLE_YPOWER);
        m_blackbody_table[i] = rgb;
#if !defined(__CUDACC__)
        //std::cout << "Table[" << i << "; T=" << T << "] = " << rgb << "\n";
#endif
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

template<typename Color>
OSL_HOSTDEVICE Color
ColorSystem::ocio_transform(StringParam fromspace, StringParam tospace,
                            const Color& C, ShadingContext* ctx,
                            ExecContextPtr ec) const
{
// Currently CPU only supports ocio by going through ShadingContext
#if !defined(__CUDA_ARCH__)
    Color Cout;

    assert(ctx);

    if (ctx->ocio_transform(fromspace, tospace, C, Cout))
        return Cout;

    if (ec == nullptr) {
        // Batched isn't using the new error reporting interface,
        // so continue to go through CPU only ShadingContext
        ctx->errorfmt("Unknown color space transformation \"{}\" -> \"{}\"",
                      fromspace, tospace);
    } else {
        OSL::errorfmt(ec, "Unknown color space transformation \"{}\" -> \"{}\"",
                      fromspace, tospace);
    }
#endif  // !define(__CUDA_ARCH__)

    return C;
}



OSL_HOSTDEVICE Dual2<Color3>
ColorSystem::ocio_transform(StringParam fromspace, StringParam tospace,
                            const Dual2<Color3>& C, ShadingContext* ctx,
                            ExecContextPtr ec) const
{
    return ocio_transform<Dual2<Color3>>(fromspace, tospace, C, ctx, ec);
}



OSL_HOSTDEVICE Color3
ColorSystem::ocio_transform(StringParam fromspace, StringParam tospace,
                            const Color3& C, ShadingContext* ctx,
                            ExecContextPtr ec) const
{
    return ocio_transform<Color3>(fromspace, tospace, C, ctx, ec);
}



OSL_HOSTDEVICE Color3
ColorSystem::to_rgb(StringParam fromspace, const Color3& C, ShadingContext* ctx,
                    ExecContextPtr ec) const
{
    // NOTE: any changes here should be mirrored
    // in wide_prepend_color_from in wide_opcolor.cpp
    if (fromspace == STRING_PARAMS(RGB) || fromspace == STRING_PARAMS(rgb)
        || fromspace == m_colorspace)
        return C;
    if (fromspace == STRING_PARAMS(hsv))
        return hsv_to_rgb(C);
    if (fromspace == STRING_PARAMS(hsl))
        return hsl_to_rgb(C);
    if (fromspace == STRING_PARAMS(YIQ))
        return YIQ_to_rgb(C);
    if (fromspace == STRING_PARAMS(XYZ))
        return XYZ_to_RGB(C);
    if (fromspace == STRING_PARAMS(xyY))
        return XYZ_to_RGB(xyY_to_XYZ(C));
    else
        return ocio_transform(fromspace, STRING_PARAMS(RGB), C, ctx, ec);
}



OSL_HOSTDEVICE Color3
ColorSystem::from_rgb(StringParam tospace, const Color3& C, ShadingContext* ctx,
                      ExecContextPtr ec) const
{
    if (tospace == STRING_PARAMS(RGB) || tospace == STRING_PARAMS(rgb)
        || tospace == m_colorspace)
        return C;
    if (tospace == STRING_PARAMS(hsv))
        return rgb_to_hsv(C);
    if (tospace == STRING_PARAMS(hsl))
        return rgb_to_hsl(C);
    if (tospace == STRING_PARAMS(YIQ))
        return rgb_to_YIQ(C);
    if (tospace == STRING_PARAMS(XYZ))
        return RGB_to_XYZ(C);
    if (tospace == STRING_PARAMS(xyY))
        return XYZ_to_xyY(RGB_to_XYZ(C));
    else
        return ocio_transform(STRING_PARAMS(RGB), tospace, C, ctx, ec);
}



template<typename COLOR>
OSL_HOSTDEVICE COLOR
ColorSystem::transformc(StringParam fromspace, StringParam tospace,
                        const COLOR& C, ShadingContext* ctx,
                        ExecContextPtr ec) const
{
    // NOTE: any changes here should be mirrored
    // in wide_transformc in wide_opcolor.cpp
    bool use_colorconfig = false;
    COLOR Crgb;
    if (fromspace == STRING_PARAMS(RGB) || fromspace == STRING_PARAMS(rgb)
        || fromspace == STRING_PARAMS(linear) || fromspace == m_colorspace)
        Crgb = C;
    else if (fromspace == STRING_PARAMS(hsv))
        Crgb = hsv_to_rgb(C);
    else if (fromspace == STRING_PARAMS(hsl))
        Crgb = hsl_to_rgb(C);
    else if (fromspace == STRING_PARAMS(YIQ))
        Crgb = YIQ_to_rgb(C);
    else if (fromspace == STRING_PARAMS(XYZ))
        Crgb = XYZ_to_RGB(C);
    else if (fromspace == STRING_PARAMS(xyY))
        Crgb = XYZ_to_RGB(xyY_to_XYZ(C));
    else if (fromspace == STRING_PARAMS(sRGB))
        Crgb = sRGB_to_linear(C);
    else {
        use_colorconfig = true;
    }

    COLOR Cto;
    if (use_colorconfig) {
        // do things the ColorConfig way, so skip all these other clauses...
    } else if (tospace == STRING_PARAMS(RGB) || tospace == STRING_PARAMS(rgb)
               || tospace == STRING_PARAMS(linear) || tospace == m_colorspace)
        Cto = Crgb;
    else if (tospace == STRING_PARAMS(hsv))
        Cto = rgb_to_hsv(Crgb);
    else if (tospace == STRING_PARAMS(hsl))
        Cto = rgb_to_hsl(Crgb);
    else if (tospace == STRING_PARAMS(YIQ))
        Cto = rgb_to_YIQ(Crgb);
    else if (tospace == STRING_PARAMS(XYZ))
        Cto = RGB_to_XYZ(Crgb);
    else if (tospace == STRING_PARAMS(xyY))
        Cto = XYZ_to_xyY(RGB_to_XYZ(Crgb));
    else if (tospace == STRING_PARAMS(sRGB))
        Cto = linear_to_sRGB(Crgb);
    else {
        use_colorconfig = true;
    }

    if (use_colorconfig) {
        Cto = ocio_transform(fromspace, tospace, C, ctx, ec);
    }

    return Cto;
}



OSL_HOSTDEVICE Dual2<Color3>
ColorSystem::transformc(StringParam fromspace, StringParam tospace,
                        const Dual2<Color3>& color, ShadingContext* ctx,
                        ExecContextPtr ec) const
{
    return transformc<Dual2<Color3>>(fromspace, tospace, color, ctx, ec);
}



OSL_HOSTDEVICE Color3
ColorSystem::transformc(StringParam fromspace, StringParam tospace,
                        const Color3& color, ShadingContext* ctx,
                        ExecContextPtr ec) const
{
    return transformc<Color3>(fromspace, tospace, color, ctx, ec);
}

}  // namespace pvt


// For Optix, this will be defined by the renderer. Otherwise inline a getter.
#ifdef __CUDACC__
extern "C" __device__ int
rend_get_userdata(StringParam name, void* data, int data_size,
                  const OSL::TypeDesc& type, int index);

namespace {

__device__ static inline const ColorSystem&
get_colorsystem(OSL::OpaqueExecContextPtr /*oec*/)
{
    void* ptr;
    rend_get_userdata(STRING_PARAMS(colorsystem), &ptr, 8, OSL::TypeDesc::PTR,
                      0);
    return *((ColorSystem*)ptr);
}

}  // namespace

#else

namespace {

inline const ColorSystem&
get_colorsystem(OSL::OpaqueExecContextPtr oec)
{
    auto ec                  = pvt::get_ec(oec);
    ShadingStateUniform* ssu = (ShadingStateUniform*)ec->shadingStateUniform;
    return ssu->m_colorsystem;
}

}  // namespace

#endif

OSL_SHADEOP OSL_HOSTDEVICE void
osl_blackbody_vf(OpaqueExecContextPtr oec, void* out, float temp)
{
    const ColorSystem& cs = get_colorsystem(oec);
    *(Color3*)out         = cs.blackbody_rgb(temp);
}



OSL_SHADEOP OSL_HOSTDEVICE void
osl_wavelength_color_vf(OpaqueExecContextPtr oec, void* out, float lambda)
{
    const ColorSystem& cs = get_colorsystem(oec);
    Color3 rgb            = cs.XYZ_to_RGB(wavelength_color_XYZ(lambda));
    //    constrain_rgb (rgb);
    rgb *= 1.0 / 2.52;  // Empirical scale from lg to make all comps <= 1
                        //    norm_rgb (rgb);
    clamp_zero(rgb);
    *(Color3*)out = rgb;
}



OSL_SHADEOP OSL_HOSTDEVICE void
osl_luminance_fv(OpaqueExecContextPtr oec, void* out, void* c)
{
    const ColorSystem& cs = get_colorsystem(oec);
    ((float*)out)[0]      = cs.luminance(((const Color3*)c)[0]);
}



OSL_SHADEOP OSL_HOSTDEVICE void
osl_luminance_dfdv(OpaqueExecContextPtr oec, void* out, void* c)
{
    const ColorSystem& cs = get_colorsystem(oec);
    ((float*)out)[0]      = cs.luminance(((const Color3*)c)[0]);
    ((float*)out)[1]      = cs.luminance(((const Color3*)c)[1]);
    ((float*)out)[2]      = cs.luminance(((const Color3*)c)[2]);
}



OSL_SHADEOP OSL_HOSTDEVICE void
osl_prepend_color_from(OpaqueExecContextPtr oec, void* c_, const char* from)
{
    const ColorSystem& cs = get_colorsystem(oec);
    auto ec               = pvt::get_ec(oec);
    COL(c_)               = cs.to_rgb(HDSTR(from), COL(c_), ec->context, ec);
}



OSL_SHADEOP OSL_HOSTDEVICE int
osl_transformc(OpaqueExecContextPtr oec, void* Cin, int Cin_derivs, void* Cout,
               int Cout_derivs, void* from_, void* to_)
{
    const ColorSystem& cs = get_colorsystem(oec);
    StringParam from      = HDSTR(from_);
    StringParam to        = HDSTR(to_);

    auto ec = pvt::get_ec(oec);

    if (Cout_derivs) {
        if (Cin_derivs) {
            DCOL(Cout) = cs.transformc(from, to, DCOL(Cin), ec->context, ec);
            return true;
        } else {
            // We had output derivs, but not input. Zero the output
            // derivs and fall through to the non-deriv case.
            ((Color3*)Cout)[1].setValue(0.0f, 0.0f, 0.0f);
            ((Color3*)Cout)[2].setValue(0.0f, 0.0f, 0.0f);
        }
    }

    // No-derivs case
    COL(Cout) = cs.transformc(from, to, COL(Cin), ec->context, ec);
    return true;
}



OSL_NAMESPACE_EXIT
