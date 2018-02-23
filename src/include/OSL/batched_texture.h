/*
Copyright (c) 2009-2013 Sony Pictures Imageworks Inc., et al.
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

#include "wide.h"

OSL_NAMESPACE_ENTER

namespace Tex {

// Future OIIO 1.9 interfaces for batched texturing
// When OSL moves to OIIO 1.9, can replace these enum definitions with 'using' or typedefs
//using OIIO::Tex::Wrap;
//using OIIO::Tex::MipMode;
//using OIIO::Tex::InterpMode;

/// Wrap mode describes what happens when texture coordinates describe
/// a value outside the usual [0,1] range where a texture is defined.
enum class Wrap {
    Default,        ///< Use the default found in the file
    Black,          ///< Black outside [0..1]
    Clamp,          ///< Clamp to [0..1]
    Periodic,       ///< Periodic mod 1
    Mirror,         ///< Mirror the image
    PeriodicPow2,   ///< Periodic, but only for powers of 2!!!
    PeriodicSharedBorder,  ///< Periodic with shared border (env)
    Last            ///< Mark the end -- don't use this!
};

/// Mip mode determines if/how we use mipmaps
///
enum class MipMode {
    Default,      ///< Default high-quality lookup
    NoMIP,        ///< Just use highest-res image, no MIP mapping
    OneLevel,     ///< Use just one mipmap level
    Trilinear,    ///< Use two MIPmap levels (trilinear)
    Aniso         ///< Use two MIPmap levels w/ anisotropic
};

/// Interp mode determines how we sample within a mipmap level
///
enum class InterpMode {
    Closest,      ///< Force closest texel
    Bilinear,     ///< Force bilinear lookup within a mip level
    Bicubic,      ///< Force cubic lookup within a mip level
    SmartBicubic  ///< Bicubic when maxifying, else bilinear
};

} // namespace Tex

struct UniformTextureOptions {
    // Options that must be the same for all points we're texturing at once
    int firstchannel = 0;                 ///< First channel of the lookup
    int subimage = 0;                     ///< Subimage or face ID
    ustring subimagename;                 ///< Subimage name
    Tex::Wrap swrap = Tex::Wrap::Default; ///< Wrap mode in the s direction
    Tex::Wrap twrap = Tex::Wrap::Default; ///< Wrap mode in the t direction
    Tex::Wrap rwrap = Tex::Wrap::Default; ///< Wrap mode in the r direction (volumetric)
    Tex::MipMode mipmode = Tex::MipMode::Default;  ///< Mip mode
    Tex::InterpMode interpmode = Tex::InterpMode::SmartBicubic;  ///< Interpolation mode
    int anisotropic = 32;                 ///< Maximum anisotropic ratio
    int conservative_filter = 1;      ///< True: over-blur rather than alias
    float fill = 0.0f;                    ///< Fill value for missing channels
    const float *missingcolor = nullptr;  ///< Color for missing texture
};

template<int WidthT>
struct alignas(sizeof(float)*WidthT) VaryingTextureOptions {
    Wide<float> sblur;    ///< Blur amount
    Wide<float> tblur;
    Wide<float> rblur;    // For 3D volume texture lookups only:
    Wide<float> swidth;   ///< Multiplier for derivatives
    Wide<float> twidth;
    Wide<float> rwidth;   // For 3D volume texture lookups only:
};

struct BatchedTextureOptions {
    VaryingTextureOptions<SimdLaneCount> varying;
    UniformTextureOptions uniform;

    // Options set INTERNALLY by libtexture after the options are passed
    // by the user.  Users should not attempt to alter these!
    int private_envlayout = 0;               // Layout for environment wrap

    // Implementation detail
    // keep order synchronized to the data members in this structure
    enum class LLVMMemberIndex
    {
        sblur = 0,
        tblur,
        rblur,
        swidth,
        twidth,
        rwidth,
        firstchannel,
        subimage,
        subimagename,
        swrap,
        twrap,
        rwrap,
        mipmode,
        interpmode,
        anisotropic,
        conservative_filter,
        fill,
        missingcolor,
        private_envlayout,
        count
    };
};

//#define __OSL_VALIDATE_BATCHED_TEXTURE_OPTIONS 1
#ifdef __OSL_VALIDATE_BATCHED_TEXTURE_OPTIONS
    // Code below is "preview" of upcoming OIIOv1.9 data layout for TextureOptBatch
    // Code here is to validate our OSL BatchedTextureOptions is binary compatible
    // and safe to reinterpret_cast<TextureOptBatch*>

    static constexpr int BatchWidth = SimdLaneCount;
    static constexpr int BatchAlign = BatchWidth * sizeof(float);
/// Texture options for a batch of Tex::BatchWidth points and run mask.
class OIIO_API TextureOptBatch {
public:
    /// Create a TextureOptBatch with all fields initialized to reasonable
    /// defaults.
    TextureOptBatch () {}   // use inline initializers

    // Options that may be different for each point we're texturing
    alignas(BatchAlign) float sblur[BatchWidth];    ///< Blur amount
    alignas(BatchAlign) float tblur[BatchWidth];
    alignas(BatchAlign) float rblur[BatchWidth];
    alignas(BatchAlign) float swidth[BatchWidth];   ///< Multiplier for derivatives
    alignas(BatchAlign) float twidth[BatchWidth];
    alignas(BatchAlign) float rwidth[BatchWidth];
    // Note: rblur,rwidth only used for volumetric lookups

    // Options that must be the same for all points we're texturing at once
    int firstchannel = 0;                 ///< First channel of the lookup
    int subimage = 0;                     ///< Subimage or face ID
    ustring subimagename;                 ///< Subimage name
    Tex::Wrap swrap = Tex::Wrap::Default; ///< Wrap mode in the s direction
    Tex::Wrap twrap = Tex::Wrap::Default; ///< Wrap mode in the t direction
    Tex::Wrap rwrap = Tex::Wrap::Default; ///< Wrap mode in the r direction (volumetric)
    Tex::MipMode mipmode = Tex::MipMode::Default;  ///< Mip mode
    Tex::InterpMode interpmode = Tex::InterpMode::SmartBicubic;  ///< Interpolation mode
    int anisotropic = 32;                 ///< Maximum anisotropic ratio
    int conservative_filter = 1;          ///< True: over-blur rather than alias
    float fill = 0.0f;                    ///< Fill value for missing channels
    const float *missingcolor = nullptr;  ///< Color for missing texture

private:
    // Options set INTERNALLY by libtexture after the options are passed
    // by the user.  Users should not attempt to alter these!
    int envlayout = 0;               // Layout for environment wrap

    //friend class pvt::TextureSystemImpl;
};

namespace  validate_offsets {
    static constexpr size_t uniform_offset = offsetof(BatchedTextureOptions,uniform);
    static constexpr size_t varying_offset = offsetof(BatchedTextureOptions,varying);
    typedef VaryingTextureOptions<SimdLaneCount> VTO;

    static_assert(offsetof(TextureOptBatch, sblur)%64 == 0, "oops unaligned wide variable");
    static_assert(offsetof(TextureOptBatch, tblur)%64 == 0, "oops unaligned wide variable");
    static_assert(offsetof(TextureOptBatch, rblur)%64 == 0, "oops unaligned wide variable");
    static_assert(offsetof(TextureOptBatch, swidth)%64 == 0, "oops unaligned wide variable");
    static_assert(offsetof(TextureOptBatch, twidth)%64 == 0, "oops unaligned wide variable");
    static_assert(offsetof(TextureOptBatch, rwidth)%64 == 0, "oops unaligned wide variable");

    static_assert(sizeof(TextureOptBatch) == sizeof(BatchedTextureOptions), "BatchedTextureOptions size differs from OIIO::TextureOptBatch");

    static_assert(offsetof(TextureOptBatch, sblur) == varying_offset + offsetof(VTO, sblur), "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
    static_assert(offsetof(TextureOptBatch, tblur) == varying_offset + offsetof(VTO, tblur), "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
    static_assert(offsetof(TextureOptBatch, rblur) == varying_offset + offsetof(VTO, rblur), "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
    static_assert(offsetof(TextureOptBatch, swidth) == varying_offset + offsetof(VTO, swidth), "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
    static_assert(offsetof(TextureOptBatch, twidth) == varying_offset + offsetof(VTO, twidth), "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
    static_assert(offsetof(TextureOptBatch, rwidth) == varying_offset + offsetof(VTO, rwidth), "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");

    static_assert(offsetof(TextureOptBatch, firstchannel) == uniform_offset + offsetof(UniformTextureOptions, firstchannel), "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
    static_assert(offsetof(TextureOptBatch, subimage) == uniform_offset + offsetof(UniformTextureOptions, subimage), "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
    static_assert(offsetof(TextureOptBatch, subimagename) == uniform_offset + offsetof(UniformTextureOptions, subimagename), "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
    static_assert(offsetof(TextureOptBatch, swrap) == uniform_offset + offsetof(UniformTextureOptions, swrap), "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
    static_assert(offsetof(TextureOptBatch, twrap) == uniform_offset + offsetof(UniformTextureOptions, twrap), "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
    static_assert(offsetof(TextureOptBatch, rwrap) == uniform_offset + offsetof(UniformTextureOptions, rwrap), "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
    static_assert(offsetof(TextureOptBatch, mipmode) == uniform_offset + offsetof(UniformTextureOptions, mipmode), "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
    static_assert(offsetof(TextureOptBatch, interpmode) == uniform_offset + offsetof(UniformTextureOptions, interpmode), "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
    static_assert(offsetof(TextureOptBatch, anisotropic) == uniform_offset + offsetof(UniformTextureOptions, anisotropic), "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
    static_assert(offsetof(TextureOptBatch, conservative_filter) == uniform_offset + offsetof(UniformTextureOptions, conservative_filter), "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
    static_assert(offsetof(TextureOptBatch, fill) == uniform_offset + offsetof(UniformTextureOptions, fill), "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
    static_assert(offsetof(TextureOptBatch, missingcolor) == uniform_offset + offsetof(UniformTextureOptions, missingcolor), "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
} // namespace validate_offsets

#endif


// Wrapper class to provide outputs resusing existing MaskedDataRef wrapper
// one new method added "bool MaskedDataRef::valid()"
// The wrapper class itself exists to get the 3 different MaskedDataRef classes
// to all share the same mask value (after inlining) vs. 3 different copies
// NOTE: detection and access to derivatives for result and alpha can be done
// using methods "has_derivs", "maskedDx()", and "maskedDy()"
// Detection of nchannels shouldn't be necessary, instead check results().is<float>() or results.is<Color3>()
class BatchedTextureOutputs
{
public:
	explicit
	BatchedTextureOutputs(void* result, bool resultHasDerivs, int chans,
                          void* alpha, bool alphaHasDerivs,
                          void* errormessage, Mask mask)
        : m_result(result),
          m_resultHasDerivs(resultHasDerivs),
          m_resultType((chans == 1) ? TypeDesc::TypeFloat : TypeDesc::TypeColor),
          m_alpha(alpha),
          m_alphaHasDerivs(alphaHasDerivs),
          m_errormessage(errormessage),
          m_mask(mask)
    {
        ASSERT(chans == 1 || chans == 3);
    }

    OSL_INLINE Mask mask() const
    {
        return m_mask;
    }

    OSL_INLINE MaskedDataRef result()
    {
        //ASSERT(result().is<float>() || result().is<Color3>());
        //ASSERT(result().has_derivs() == true);
        return MaskedDataRef(m_resultType, m_resultHasDerivs, m_mask, m_result);
    }

    OSL_INLINE MaskedDataRef alpha()
    {
        // ASSERT(alpha().is<float>());
        // ASSERT(alpha().valid() == true || alpha().valid() == false);
        // ASSERT(alpha().has_derivs() == true || alpha().has_derivs() == false);
        return MaskedDataRef(TypeDesc::TypeFloat, m_alphaHasDerivs, m_mask, m_alpha);
    }

    OSL_INLINE MaskedDataRef errormessage()
    {
        // ASSERT(errormessage().is<ustring>());
        // ASSERT(errormessage().valid() == true || errormessage().valid() == false);
        // ASSERT(errormessage().has_derivs() == false);
        return MaskedDataRef(TypeDesc::TypeString, false, m_mask, m_errormessage);
    }

private:
    void* m_result;
    bool m_resultHasDerivs;
    TypeDesc m_resultType;
    void* m_alpha;
    bool m_alphaHasDerivs;
    void* m_errormessage;
    Mask m_mask;
};



OSL_NAMESPACE_EXIT
