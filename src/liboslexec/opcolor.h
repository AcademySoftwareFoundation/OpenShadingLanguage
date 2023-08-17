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

#include <OSL/Imathx/Imathx.h>
#include <OSL/device_string.h>
#include <OSL/dual.h>
#include <OSL/dual_vec.h>

#include <OpenImageIO/color.h>



OSL_NAMESPACE_ENTER

//Forward declare
class ShadingContext;
struct ShaderGlobals;
typedef ShaderGlobals ExecContext;
typedef ExecContext* ExecContextPtr;

namespace pvt {

// By default, color transformation errors are reported through ShadingContext,
// unless the optional ExecutionContextPtr was provided which causes errors
// to be reported through the renderer services
class OSLEXECPUBLIC ColorSystem {
public:
    // A colour system is defined by the CIE x and y coordinates of its
    // three primary illuminants and its white point.
    struct Chroma {
        float xRed, yRed, xGreen, yGreen, xBlue, yBlue, xWhite, yWhite;
    };

    OSL_HOSTDEVICE static const Chroma* fromString(StringParam colorspace);

    /// Convert an XYZ color to RGB in our preferred color space.
    template<typename T> OSL_HOSTDEVICE T XYZ_to_RGB(const T& XYZ) const
    {
        return XYZ * m_XYZ2RGB;
    }

    /// Convert an RGB color in our preferred color space to XYZ.
    template<typename T> OSL_HOSTDEVICE T RGB_to_XYZ(const T& RGB) const
    {
        return RGB * m_RGB2XYZ;
    }

    /// Return the luminance of an RGB color in the current color space.
    OSL_HOSTDEVICE float luminance(const Color3& RGB) const
    {
        return RGB.dot(m_luminance_scale);
    }

    /// Return the luminance scale  of the current color space.
    const Color3& luminance_scale() const { return m_luminance_scale; }

    /// Return the RGB in the current color space for blackbody radiation
    /// at temperature T (in Kelvin).
    OSL_HOSTDEVICE inline Color3 blackbody_rgb(float T /*Kelvin*/) const;

    // Interface to access underlying optimized lookup table for blackbody
    // When can_lookup_blackbody() returns true,
    // then lookup_blackbody_rgb can be safely called,
    // otherwise the compute_blackbody_rgb is required.
    OSL_HOSTDEVICE inline bool can_lookup_blackbody(float T /*Kelvin*/) const;

    OSL_HOSTDEVICE inline Color3 lookup_blackbody_rgb(float T /*Kelvin*/) const;

    // Expensive real computation without lookup table
    OSL_HOSTDEVICE inline Color3 compute_blackbody_rgb(float T /*Kelvin*/) const;


    /// Set the current color space.
    OSL_HOSTDEVICE bool set_colorspace(StringParam colorspace);

    OSL_HOSTDEVICE Color3 to_rgb(StringParam fromspace, const Color3& C,
                                 ShadingContext*,
                                 ExecContextPtr ec = nullptr) const;

    OSL_HOSTDEVICE Color3 from_rgb(StringParam fromspace, const Color3& C,
                                   ShadingContext*,
                                   ExecContextPtr ec = nullptr) const;

    OSL_HOSTDEVICE Dual2<Color3> transformc(StringParam fromspace,
                                            StringParam tospace,
                                            const Dual2<Color3>& color,
                                            ShadingContext*,
                                            ExecContextPtr ec = nullptr) const;

    OSL_HOSTDEVICE Color3 transformc(StringParam fromspace, StringParam tospace,
                                     const Color3& color, ShadingContext*,
                                     ExecContextPtr ec = nullptr) const;

    OSL_HOSTDEVICE Dual2<Color3>
    ocio_transform(StringParam fromspace, StringParam tospace,
                   const Dual2<Color3>& C, ShadingContext*,
                   ExecContextPtr ec = nullptr) const;

    OSL_HOSTDEVICE Color3 ocio_transform(StringParam fromspace,
                                         StringParam tospace, const Color3& C,
                                         ShadingContext*,
                                         ExecContextPtr ec = nullptr) const;

    OSL_HOSTDEVICE const StringParam& colorspace() const
    {
        return m_colorspace;
    }

private:
    template<typename Color>
    OSL_HOSTDEVICE inline Color
    transformc(StringParam fromspace, StringParam tospace, const Color& C,
               ShadingContext*, ExecContextPtr ec = nullptr) const;

    template<typename Color>
    OSL_HOSTDEVICE inline Color
    ocio_transform(StringParam fromspace, StringParam tospace, const Color& C,
                   ShadingContext*, ExecContextPtr ec = nullptr) const;


    // Derived/cached calculations from options:
    Color3 m_Red, m_Green, m_Blue;  ///< Color primaries (xyY)
    Color3 m_White;                 ///< White point (xyY)
    Matrix33 m_XYZ2RGB;             ///< XYZ to RGB conversion matrix
    Matrix33 m_RGB2XYZ;             ///< RGB to XYZ conversion matrix
    Color3 m_luminance_scale;       ///< Scaling for RGB->luma
    Color3 m_blackbody_table[317];  ///< Precomputed blackbody table

    // Keep this last so the CUDA device string can be easily set
    StringParam m_colorspace;  ///< What RGB colors mean
};


class OCIOColorSystem {
#ifndef __CUDACC__
public:
    OIIO::ColorProcessorHandle load_transform(StringParam fromspace,
                                              StringParam tospace,
                                              ShadingSystemImpl* shadingsys);

private:
    const OIIO::ColorConfig& colorconfig(ShadingSystemImpl* shadingsys);

    std::shared_ptr<OIIO::ColorConfig>
        m_colorconfig;  ///< OIIO/OCIO color configuration

    // 1-item cache for the last requested custom color conversion processor
    OIIO::ColorProcessorHandle m_last_colorproc;
    ustring m_last_colorproc_fromspace;
    ustring m_last_colorproc_tospace;
#endif
};

}  // namespace pvt


OSL_NAMESPACE_EXIT
