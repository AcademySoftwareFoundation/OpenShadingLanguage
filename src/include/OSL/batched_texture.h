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

#ifdef OSL_EXPERIMENTAL_BATCHED_TEXTURE


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

#else

class BatchedTextureOptionProvider
{
public:
    enum Options {
        SWIDTH = 0,         // int | float
        TWIDTH,             // int | float
        RWIDTH,             // int | float
        SBLUR,              // int | float
        TBLUR,              // int | float
        RBLUR,              // int | float
        SWRAP,              // int | string
        TWRAP,              // int | string
        RWRAP,              // int | string
        FILL,               // int | float
        TIME,               // int | float
        FIRSTCHANNEL,       // int
        SUBIMAGE,           // int | string
        INTERP,             // int | string
        MISSINGCOLOR,       // color
        MISSINGALPHA,       // float

        MAX_OPTIONS
    };
    enum DataType {
        INT = 0,
        COLOR = 0,
        FLOAT = 1,
        STRING = 1,
    };

    static constexpr unsigned int maskSize = 32;
    static_assert(MAX_OPTIONS <= maskSize, "expecting MAX_OPTIONS <= maskSize");
    typedef WideMask<maskSize> Mask;

    struct OptionData
    {
        Mask active;
        Mask varying;
        Mask type; // data type is int = 0 or float = 1
        Mask ALIGN; // not used, for 64 bit data alignment
        void* options[MAX_OPTIONS];
    };

private:
    const OptionData * m_opt;
    float m_missingcolor[4];

public:
    explicit
	BatchedTextureOptionProvider(const OptionData * data)
    : m_opt(data)
     ,m_missingcolor{0.f,0.f,0.f,0.f}
    {}

    void updateOption(TextureOpt &opt, unsigned int l)
    {
        // check we actually have valid option data.
        if (m_opt == nullptr) return;

#if 0
        std::cout << "size: " << sizeof(TextureOptions) << std::endl;
        std::cout << "active: " << &m_opt->active << " " << m_opt->active.value() << std::endl;
        std::cout << "varying: " << m_opt->varying.value() << std::endl;
        std::cout << "type: " << m_opt->type.value() << std::endl;
        for (int i = 0; i < m_opt->active.count(); ++i) {
            std::cout << "void* " << m_opt->options[i] << std::endl;
            std::cout << "int " << *(int*)m_opt->options[i] << std::endl;
        }
#endif
        int j = 0; // offset index to next void pointer

#define OPTION_CASE(i, optName)                                                             \
        if (m_opt->active[i]) {                                                                  \
            if (m_opt->varying[i]) {                                                             \
                if (m_opt->type[i] == static_cast<bool>(INT)) {                                  \
                    ConstWideAccessor<int> wideResult(m_opt->options[j]);    \
                    opt.optName = static_cast<float>(wideResult[l]);                    \
                }                                                                           \
                else {                                                                      \
                    ConstWideAccessor<float> wideResult(m_opt->options[j]);    \
                    opt.optName = wideResult[l];                                        \
                }                                                                           \
            }                                                                               \
            else {                                                                          \
                if (m_opt->type[i] == static_cast<bool>(INT)) {                                  \
                    opt.optName = static_cast<float>(*reinterpret_cast<int*>(m_opt->options[j]));\
                }                                                                           \
                else  {                                                                     \
                    opt.optName = *reinterpret_cast<float*>(m_opt->options[j]);                  \
                }                                                                           \
            }                                                                               \
            ++j;                                                                            \
        }

#define OPTION_CASE_DECODE(i, optName, decode, typeCast)                                    \
        if (m_opt->active[i]) {                                                                  \
            if (m_opt->varying[i]) {                                                             \
                if (m_opt->type[i] == static_cast<bool>(STRING)) {                               \
                    ConstWideAccessor<ustring> wideResult(m_opt->options[j]);    \
                    opt.optName = decode(static_cast<const ustring>(wideResult[l]));                                \
                }                                                                           \
                else {                                                                      \
                    ConstWideAccessor<int> wideResult(m_opt->options[j]);    \
                    opt.optName = (typeCast)static_cast<int>(wideResult[l]);                              \
                }                                                                           \
            }                                                                               \
            else {                                                                          \
                if (m_opt->type[i] == static_cast<bool>(STRING)) {                               \
                    ustring& castValue = *reinterpret_cast<ustring*>(m_opt->options[j]);         \
                    opt.optName = decode(castValue);                                        \
                }                                                                           \
                else                                                                        \
                    opt.optName = (typeCast)*reinterpret_cast<int*>(m_opt->options[j]);          \
            }                                                                               \
            ++j;                                                                            \
        }

        // Check all options
        OPTION_CASE(SWIDTH, swidth)
        OPTION_CASE(TWIDTH, twidth)
        OPTION_CASE(RWIDTH, rwidth)
        OPTION_CASE(SBLUR, sblur)
        OPTION_CASE(TBLUR, tblur)
        OPTION_CASE(RBLUR, rblur)
        OPTION_CASE_DECODE(SWRAP, swrap, TextureOpt::decode_wrapmode, TextureOpt::Wrap)
        OPTION_CASE_DECODE(TWRAP, twrap, TextureOpt::decode_wrapmode, TextureOpt::Wrap)
        OPTION_CASE_DECODE(RWRAP, rwrap, TextureOpt::decode_wrapmode, TextureOpt::Wrap)
        OPTION_CASE(FILL, fill)
        OPTION_CASE(TIME, time)
        if (m_opt->active[FIRSTCHANNEL]) {
            if (m_opt->varying[FIRSTCHANNEL]) {
                ConstWideAccessor<int> wideResult(m_opt->options[j]);    \
                opt.firstchannel = wideResult[l];
            }
            else {
                opt.firstchannel = *reinterpret_cast<int*>(m_opt->options[j]);
            }
            ++j;
        }
        if (m_opt->active[SUBIMAGE]) {
            if (m_opt->varying[SUBIMAGE]) {
                if (m_opt->type[SUBIMAGE] == static_cast<bool>(STRING)) {
                    ConstWideAccessor<ustring> wideResult(m_opt->options[j]);    \
                    opt.subimagename = wideResult[l];           \
                }
                else {
                    ConstWideAccessor<int> wideResult(m_opt->options[j]);    \
                    opt.subimage = wideResult[l];
                }
            }
            else {
                if (m_opt->type[SUBIMAGE] == static_cast<bool>(STRING)) {
                    ustring& castValue = *reinterpret_cast<ustring*>(m_opt->options[j]);
                    opt.subimagename = castValue;                   \
                }
                else
                    opt.subimage = *reinterpret_cast<int*>(m_opt->options[j]);
            }
            ++j;
        }
        OPTION_CASE_DECODE(INTERP, interpmode, texInterpToCode, TextureOpt::InterpMode)
        if (m_opt->active[MISSINGCOLOR]) {
            Color3 missingcolor;
            if (m_opt->varying[MISSINGCOLOR]) {
                ConstWideAccessor<Color3> wideResult(m_opt->options[j]);    \
                missingcolor = wideResult[l];
            }
            else {
                missingcolor = *reinterpret_cast<Color3*>(m_opt->options[j]);
            }
            m_missingcolor[0] = missingcolor.x;
            m_missingcolor[1] = missingcolor.y;
            m_missingcolor[2] = missingcolor.z;
            opt.missingcolor = m_missingcolor;
            ++j;
        }
        if (m_opt->active[MISSINGALPHA]) {
            if (m_opt->varying[MISSINGALPHA]) {
                ConstWideAccessor<float> wideResult(m_opt->options[j]);    \
                m_missingcolor[3] = wideResult[l];
            }
            else {
                m_missingcolor[3] = *reinterpret_cast<float*>(m_opt->options[j]);
            }
            opt.missingcolor = m_missingcolor;
            ++j;
        }
#undef OPTION_CASE
#undef OPTION_CASE_DECODE
    }

private:
    // this should be refactored into OIIO texture.h?
    OSL_INLINE TextureOpt::InterpMode texInterpToCode (ustring modename) const
    {
        static ustring u_linear ("linear");
        static ustring u_smartcubic ("smartcubic");
        static ustring u_cubic ("cubic");
        static ustring u_closest ("closest");

        TextureOpt::InterpMode mode = TextureOpt::InterpClosest;
        if (modename == u_smartcubic)
            mode = TextureOpt::InterpSmartBicubic;
        else if (modename == u_linear)
            mode = TextureOpt::InterpBilinear;
        else if (modename == u_cubic)
            mode = TextureOpt::InterpBicubic;
        else if (modename == u_closest)
            mode = TextureOpt::InterpClosest;
        return mode;
    }
};
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
