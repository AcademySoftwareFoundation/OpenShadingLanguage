// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/wide.h>

OSL_NAMESPACE_ENTER

namespace Tex {

using OIIO::Tex::InterpMode;
using OIIO::Tex::MipMode;
using OIIO::Tex::Wrap;

}  // namespace Tex

struct UniformTextureOptions {
    // Options that must be the same for all points we're texturing at once
    int firstchannel = 0;                  ///< First channel of the lookup
    int subimage     = 0;                  ///< Subimage or face ID
    ustring subimagename;                  ///< Subimage name
    Tex::Wrap swrap = Tex::Wrap::Default;  ///< Wrap mode in the s direction
    Tex::Wrap twrap = Tex::Wrap::Default;  ///< Wrap mode in the t direction
    Tex::Wrap rwrap
        = Tex::Wrap::Default;  ///< Wrap mode in the r direction (volumetric)
    Tex::MipMode mipmode = Tex::MipMode::Default;  ///< Mip mode
    Tex::InterpMode interpmode
        = Tex::InterpMode::SmartBicubic;  ///< Interpolation mode
    int anisotropic           = 32;       ///< Maximum anisotropic ratio
    int conservative_filter   = 1;        ///< True: over-blur rather than alias
    float fill                = 0.0f;     ///< Fill value for missing channels
    const float* missingcolor = nullptr;  ///< Color for missing texture
};

template<int WidthT> struct VaryingTextureOptions {
    Block<float, WidthT> sblur;  ///< Blur amount
    Block<float, WidthT> tblur;
    Block<float, WidthT> rblur;   // For 3D volume texture lookups only:
    Block<float, WidthT> swidth;  ///< Multiplier for derivatives
    Block<float, WidthT> twidth;
    Block<float, WidthT> rwidth;  // For 3D volume texture lookups only:
#if OIIO_VERSION_GREATER_EQUAL(2, 4, 0)
    Block<float, WidthT> rnd;  // For stochastic sampling
#endif
};
static_assert(std::alignment_of<VaryingTextureOptions<16>>::value
                  == VecReg<16>::alignment,
              "Expect alignment of data member to set alignment of struct");
static_assert(std::alignment_of<VaryingTextureOptions<8>>::value
                  == VecReg<8>::alignment,
              "Expect alignment of data member to set alignment of struct");

template<int WidthT> struct BatchedTextureOptions {
    VaryingTextureOptions<WidthT> varying;
    UniformTextureOptions uniform;

    // Options set INTERNALLY by libtexture after the options are passed
    // by the user.  Users should not attempt to alter these!
    int private_envlayout = 0;  // Layout for environment wrap

    // Implementation detail
    // keep order synchronized to the data members in this structure
    enum class LLVMMemberIndex {
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
static_assert(std::alignment_of<BatchedTextureOptions<16>>::value
                  == VecReg<16>::alignment,
              "Expect alignment of data member to set alignment of struct");
static_assert(std::alignment_of<BatchedTextureOptions<8>>::value
                  == VecReg<8>::alignment,
              "Expect alignment of data member to set alignment of struct");

#ifdef OIIO_TEXTURE_SIMD_BATCH_WIDTH
// Code here is to validate our OSL BatchedTextureOptions<WidthT> is binary compatible
// and safe to reinterpret_cast<TextureOptBatch*>
static_assert((OIIO::Tex::BatchWidth == 16) || (OIIO::Tex::BatchWidth == 8),
              "This validation requires OIIO_TEXTURE_SIMD_BATCH_WIDTH=16");

namespace validate_offsets {

OSL_PRAGMA_WARNING_PUSH
OSL_GCC_PRAGMA(GCC diagnostic ignored "-Winvalid-offsetof")
typedef BatchedTextureOptions<OIIO::Tex::BatchWidth> BTO;
static constexpr size_t uniform_offset = offsetof(BTO, uniform);
static constexpr size_t varying_offset = offsetof(BTO, varying);

typedef VaryingTextureOptions<OIIO::Tex::BatchWidth> VTO;

static_assert(offsetof(OIIO::TextureOptBatch, sblur) % 64 == 0,
              "oops unaligned wide variable");
static_assert(offsetof(OIIO::TextureOptBatch, tblur) % 64 == 0,
              "oops unaligned wide variable");
static_assert(offsetof(OIIO::TextureOptBatch, rblur) % 64 == 0,
              "oops unaligned wide variable");
static_assert(offsetof(OIIO::TextureOptBatch, swidth) % 64 == 0,
              "oops unaligned wide variable");
static_assert(offsetof(OIIO::TextureOptBatch, twidth) % 64 == 0,
              "oops unaligned wide variable");
static_assert(offsetof(OIIO::TextureOptBatch, rwidth) % 64 == 0,
              "oops unaligned wide variable");
#    if OIIO_VERSION_GREATER_EQUAL(2, 4, 0)
static_assert(offsetof(OIIO::TextureOptBatch, rnd) % 64 == 0,
              "oops unaligned wide variable");
#    endif

static_assert(sizeof(OIIO::TextureOptBatch)
                  == sizeof(BatchedTextureOptions<16>),
              "BatchedTextureOptions size differs from OIIO::TextureOptBatch");

static_assert(
    offsetof(OIIO::TextureOptBatch, sblur)
        == varying_offset + offsetof(VTO, sblur),
    "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
static_assert(
    offsetof(OIIO::TextureOptBatch, tblur)
        == varying_offset + offsetof(VTO, tblur),
    "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
static_assert(
    offsetof(OIIO::TextureOptBatch, rblur)
        == varying_offset + offsetof(VTO, rblur),
    "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
static_assert(
    offsetof(OIIO::TextureOptBatch, swidth)
        == varying_offset + offsetof(VTO, swidth),
    "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
static_assert(
    offsetof(OIIO::TextureOptBatch, twidth)
        == varying_offset + offsetof(VTO, twidth),
    "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
static_assert(
    offsetof(OIIO::TextureOptBatch, rwidth)
        == varying_offset + offsetof(VTO, rwidth),
    "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");

static_assert(
    offsetof(OIIO::TextureOptBatch, firstchannel)
        == uniform_offset + offsetof(UniformTextureOptions, firstchannel),
    "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
static_assert(
    offsetof(OIIO::TextureOptBatch, subimage)
        == uniform_offset + offsetof(UniformTextureOptions, subimage),
    "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
static_assert(
    offsetof(OIIO::TextureOptBatch, subimagename)
        == uniform_offset + offsetof(UniformTextureOptions, subimagename),
    "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
static_assert(
    offsetof(OIIO::TextureOptBatch, swrap)
        == uniform_offset + offsetof(UniformTextureOptions, swrap),
    "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
static_assert(
    offsetof(OIIO::TextureOptBatch, twrap)
        == uniform_offset + offsetof(UniformTextureOptions, twrap),
    "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
static_assert(
    offsetof(OIIO::TextureOptBatch, rwrap)
        == uniform_offset + offsetof(UniformTextureOptions, rwrap),
    "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
static_assert(
    offsetof(OIIO::TextureOptBatch, mipmode)
        == uniform_offset + offsetof(UniformTextureOptions, mipmode),
    "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
static_assert(
    offsetof(OIIO::TextureOptBatch, interpmode)
        == uniform_offset + offsetof(UniformTextureOptions, interpmode),
    "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
static_assert(
    offsetof(OIIO::TextureOptBatch, anisotropic)
        == uniform_offset + offsetof(UniformTextureOptions, anisotropic),
    "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
static_assert(
    offsetof(OIIO::TextureOptBatch, conservative_filter)
        == uniform_offset
               + offsetof(UniformTextureOptions, conservative_filter),
    "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
static_assert(
    offsetof(OIIO::TextureOptBatch, fill)
        == uniform_offset + offsetof(UniformTextureOptions, fill),
    "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");
static_assert(
    offsetof(OIIO::TextureOptBatch, missingcolor)
        == uniform_offset + offsetof(UniformTextureOptions, missingcolor),
    "BatchedTextureOptions members offset different that OIIO::TextureOptBatch");

OSL_PRAGMA_WARNING_POP
}  // namespace validate_offsets

#endif


// The wrapper class itself exists to get the 3 different MaskedDataRef classes
// to all share the same mask value (after inlining) vs. 3 different copies
// NOTE: detection and access to derivatives for result and alpha can be done
// using methods "has_derivs", "maskedDx()", and "maskedDy()"
// Detection of nchannels shouldn't be necessary, instead check results().is<float>() or results.is<Color3>()
template<int WidthT> class BatchedTextureOutputs {
public:
    explicit BatchedTextureOutputs(void* result, bool resultHasDerivs,
                                   int chans, void* alpha, bool alphaHasDerivs,
                                   void* errormessage, Mask<WidthT> mask)
        : m_result(result)
        , m_resultHasDerivs(resultHasDerivs)
        , m_resultType((chans == 1) ? TypeDesc::TypeFloat : TypeDesc::TypeColor)
        , m_alpha(alpha)
        , m_alphaHasDerivs(alphaHasDerivs)
        , m_errormessage(errormessage)
        , m_mask(mask)
    {
        ASSERT(chans == 1 || chans == 3);
    }

    OSL_FORCEINLINE Mask<WidthT> mask() const { return m_mask; }

    // The return value will be Masked<float, WidthT> or Masked<Vec3, WidthT>
    OSL_FORCEINLINE MaskedData<WidthT> result()
    {
        return MaskedData<WidthT>(m_resultType, m_resultHasDerivs, m_mask,
                                  m_result);
    }

    // The return value maybe invalid or be Masked<float, WidthT>
    OSL_FORCEINLINE MaskedData<WidthT> alpha()
    {
        return MaskedData<WidthT>(TypeDesc::TypeFloat, m_alphaHasDerivs, m_mask,
                                  m_alpha);
    }

    // The return value maybe invalid or be Masked<ustring, WidthT>
    OSL_FORCEINLINE MaskedData<WidthT> errormessage()
    {
        return MaskedData<WidthT>(TypeDesc::TypeString, false, m_mask,
                                  m_errormessage);
    }

private:
    void* m_result;
    bool m_resultHasDerivs;
    TypeDesc m_resultType;
    void* m_alpha;
    bool m_alphaHasDerivs;
    void* m_errormessage;
    Mask<WidthT> m_mask;
};

#define __OSL_USING_BATCHED_TEXTURE(WIDTH_OF_OSL_DATA)             \
    using BatchedTextureOutputs                                    \
        = OSL_NAMESPACE::BatchedTextureOutputs<WIDTH_OF_OSL_DATA>; \
    using BatchedTextureOptions                                    \
        = OSL_NAMESPACE::BatchedTextureOptions<WIDTH_OF_OSL_DATA>;

#undef OSL_USING_DATA_WIDTH
#ifdef __OSL_USING_SHADERGLOBALS
#    define OSL_USING_DATA_WIDTH(WIDTH_OF_OSL_DATA)  \
        __OSL_USING_WIDE(WIDTH_OF_OSL_DATA)          \
        __OSL_USING_SHADERGLOBALS(WIDTH_OF_OSL_DATA) \
        __OSL_USING_BATCHED_TEXTURE(WIDTH_OF_OSL_DATA)
#else
#    define OSL_USING_DATA_WIDTH(WIDTH_OF_OSL_DATA) \
        __OSL_USING_WIDE(WIDTH_OF_OSL_DATA)         \
        __OSL_USING_BATCHED_TEXTURE(WIDTH_OF_OSL_DATA)
#endif

OSL_NAMESPACE_EXIT
