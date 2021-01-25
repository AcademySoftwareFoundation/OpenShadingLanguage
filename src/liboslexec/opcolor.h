// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Classes to share color-operation data between CPU & CUDA.
///
/////////////////////////////////////////////////////////////////////////

#include <OSL/dual.h>
#include <OSL/dual_vec.h>
#include <OSL/Imathx/Imathx.h>
#include <OSL/device_string.h>

#include <OpenImageIO/color.h>

#ifdef __CUDACC__
  #undef OIIO_HAS_COLORPROCESSOR
#endif


OSL_NAMESPACE_ENTER

class ShadingContext;

namespace pvt {


class ColorSystem {
#ifdef __CUDACC__
    using Context = void*;
#else
    using Context = ShadingContext*;
#endif
public:
    // A colour system is defined by the CIE x and y coordinates of its
    // three primary illuminants and its white point.
    struct Chroma {
        float  xRed, yRed,
               xGreen, yGreen,
               xBlue, yBlue,
               xWhite, yWhite;
    };

    OSL_HOSTDEVICE static const Chroma* fromString(StringParam colorspace);

    /// Convert an XYZ color to RGB in our preferred color space.
    template <typename T> OSL_HOSTDEVICE T
    XYZ_to_RGB (const T &XYZ) { return XYZ * m_XYZ2RGB; }

    /// Convert an RGB color in our preferred color space to XYZ.
    template <typename T> OSL_HOSTDEVICE T
    RGB_to_XYZ (const T &RGB) { return RGB * m_RGB2XYZ; }

    /// Return the luminance of an RGB color in the current color space.
    OSL_HOSTDEVICE float
    luminance (const Color3 &RGB) { return RGB.dot(m_luminance_scale); }

    /// Return the RGB in the current color space for blackbody radiation
    /// at temperature T (in Kelvin).
    OSL_HOSTDEVICE Color3
    blackbody_rgb (float T /*Kelvin*/);

    /// Set the current color space.
    OSL_HOSTDEVICE bool
    set_colorspace (StringParam colorspace);

    OSL_HOSTDEVICE Color3
    to_rgb (StringParam fromspace, const Color3& C, Context);

    OSL_HOSTDEVICE Color3
    from_rgb (StringParam fromspace, const Color3& C, Context);

    OSL_HOSTDEVICE Dual2<Color3>
    transformc (StringParam fromspace, StringParam tospace,
                const Dual2<Color3>& color, Context ctx);

    OSL_HOSTDEVICE Color3
    transformc (StringParam fromspace, StringParam tospace,
                const Color3& color, Context ctx);

    template <typename Color> OSL_HOSTDEVICE Color
    ocio_transform (StringParam fromspace, StringParam tospace, const Color& C, Context);

    OSL_HOSTDEVICE const StringParam& colorspace() const { return m_colorspace; }

    OSL_HOSTDEVICE void error(StringParam src, StringParam dst, Context);

private:
    template <typename Color> OSL_HOSTDEVICE Color
    transformc (StringParam fromspace, StringParam tospace, const Color& C, Context);

    // Derived/cached calculations from options:
    Color3 m_Red, m_Green, m_Blue;   ///< Color primaries (xyY)
    Color3 m_White;                  ///< White point (xyY)
    Matrix33 m_XYZ2RGB;              ///< XYZ to RGB conversion matrix
    Matrix33 m_RGB2XYZ;              ///< RGB to XYZ conversion matrix
    Color3 m_luminance_scale;        ///< Scaling for RGB->luma
    Color3 m_blackbody_table[317];   ///< Precomputed blackbody table

    // Keep this last so the CUDA device string can be easily set
    StringParam m_colorspace;        ///< What RGB colors mean
};


class OCIOColorSystem {
#if OIIO_HAS_COLORPROCESSOR
public:

    OIIO::ColorProcessorHandle
    load_transform(StringParam fromspace, StringParam tospace);

    const OIIO::ColorConfig& colorconfig () const { return m_colorconfig; }

private:

    OIIO::ColorConfig m_colorconfig; ///< OIIO/OCIO color configuration

    // 1-item cache for the last requested custom color conversion processor
    OIIO::ColorProcessorHandle m_last_colorproc;
    ustring m_last_colorproc_fromspace;
    ustring m_last_colorproc_tospace;
#endif
};

} // namespace pvt


OSL_NAMESPACE_EXIT
