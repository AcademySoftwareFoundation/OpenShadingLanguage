/*
Copyright (c) 2009-2019 Sony Pictures Imageworks Inc., et al.
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

#pragma once

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Classes to share color-operation data between CPU & CUDA.
///
/////////////////////////////////////////////////////////////////////////

#include <OSL/dual.h>
#include <OSL/dual_vec.h>
#include <OSL/Imathx.h>
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
    OSL_HOSTDEVICE Color3
    XYZ_to_RGB (const Color3 &XYZ)         { return XYZ * m_XYZ2RGB; }

    OSL_HOSTDEVICE Dual2<Vec3>
    XYZ_to_RGB (const Dual2<Vec3> &XYZ)    { return XYZ * m_XYZ2RGB; }

    OSL_HOSTDEVICE Color3
    XYZ_to_RGB (float X, float Y, float Z) { return Color3(X,Y,Z) * m_XYZ2RGB; }

    /// Convert an RGB color in our preferred color space to XYZ.
    OSL_HOSTDEVICE Color3
    RGB_to_XYZ (const Color3 &RGB)         { return RGB * m_RGB2XYZ; }

    OSL_HOSTDEVICE Dual2<Vec3>
    RGB_to_XYZ (const Dual2<Vec3> &RGB)    { return RGB * m_RGB2XYZ; }

    OSL_HOSTDEVICE Color3
    RGB_to_XYZ (float R, float G, float B) { return Color3(R,G,B) * m_RGB2XYZ; }

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

    OSL_HOSTDEVICE StringParam colorspace() const { return m_colorspace; }

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
