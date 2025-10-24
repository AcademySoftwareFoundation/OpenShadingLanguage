// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of color operations.
///
/////////////////////////////////////////////////////////////////////////

#include <OSL/oslconfig.h>

#include <OSL/batched_rendererservices.h>
#include <OSL/batched_shaderglobals.h>
#include <OSL/wide.h>

#include <OpenImageIO/fmath.h>

#include "oslexec_pvt.h"

#include "opcolor_impl.h"
#include "opcolor.h"

OSL_NAMESPACE_BEGIN
namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)

#include "define_opname_macros.h"

namespace {

OSL_FORCEINLINE ShadingContext*
context_from_bsg(void* bsg_)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    return bsg->uniform.context;
}

OSL_FORCEINLINE const ColorSystem&
cs_from_bsg(void* bsg)
{
    return context_from_bsg(bsg)->shadingsys().colorsystem();
}

};  // End anonymous namespace


OSL_BATCHOP void
__OSL_OP(blackbody_vf)(void* bsg_, void* out, float temp)
{
    const ColorSystem& cs = cs_from_bsg(bsg_);
    *(Color3*)out         = cs.blackbody_rgb(temp);
}



OSL_PRAGMA_WARNING_PUSH
OSL_NONINTEL_CLANG_PRAGMA(GCC diagnostic ignored "-Wpass-failed")

OSL_BATCHOP void
__OSL_MASKED_OP2(blackbody, Wv, Wf)(void* bsg_, void* wout_, void* wtemp_,
                                    unsigned int mask_value)
{
    const ColorSystem& cs = cs_from_bsg(bsg_);

    Masked<Color3> wR(wout_, Mask(mask_value));  //output
    Wide<const float> wL(wtemp_);                //input lambda

    Block<int> computeRequiredBlock;
    Wide<int> wcomputeRequired(computeRequiredBlock);

    OSL_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
    for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
        float temperature      = wL[lane];
        bool canNotLookup      = !cs.can_lookup_blackbody(temperature);
        wcomputeRequired[lane] = canNotLookup & wR.mask()[lane];
        if (canNotLookup) {
            // We choose to run computation unmasked so that
            // clang will treat the ColorSystem as uniform
            // which will avoid many gathers

            // Ensure temperature values from disabled lanes
            // are inbounds.
            temperature = 0.0f;
        }
        Color3 rgb = cs.lookup_blackbody_rgb(temperature);
        wR[lane]   = rgb;
    }

    if (testIfAnyLaneIsNonZero(wcomputeRequired)) {
        // Complex nested loop in the real computation may not vectorize in
        // in all compilers, which is why we have split off the fast path of
        // using the lookup table so it can be vectorized independently
        OSL_OMP_COMPLEX_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            float temperature   = wL[lane];
            int computeRequired = wcomputeRequired[lane];
            if (computeRequired != 0) {
                OSL_DASSERT(
                    wR.mask()[lane]
                    && "computeRequired should have already considered the result mask");
                Color3 rgb           = cs.compute_blackbody_rgb(temperature);
                wR[ActiveLane(lane)] = rgb;
            }
        }
    }
}

OSL_PRAGMA_WARNING_POP



OSL_BATCHOP void
__OSL_OP(wavelength_color_vf)(void* bsg_, void* out, float lambda)
{
    const ColorSystem& cs = cs_from_bsg(bsg_);

    Color3 rgb = cs.XYZ_to_RGB(wavelength_color_XYZ(lambda));
    //    constrain_rgb (rgb);
    rgb *= 1.0 / 2.52;  // Empirical scale from lg to make all comps <= 1
                        //    norm_rgb (rgb);
    clamp_zero(rgb);
    *(Color3*)out = rgb;
}



OSL_BATCHOP void
__OSL_MASKED_OP2(wavelength_color, Wv, Wf)(void* bsg_, void* wout_,
                                           void* wlambda_,
                                           unsigned int mask_value)
{
    const ColorSystem& cs = cs_from_bsg(bsg_);
    Masked<Color3> wR(wout_, Mask(mask_value));  //output
    Wide<const float> wL(wlambda_);              //input lambda

    OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))

    for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
        float lambda = wL[lane];

        // Normally we want to access all data we can before applying the mask
        // so those loads can be unmasked.  When accessing an object like
        // ColorSystem, it is tricky.  By calling a method of a non stack based
        // object (like ColorSystem) inside the masked region, the compiler may
        // consider it illegal to dereference the pointer unmasked.
        // When vectorizing with Clang, this effectively changed a uniform
        // load/broadcast of Matrix33 ColorSystem::m_XYZ2RGB to a series of
        // masked gathers.  One solution is to create a ColorSystem copy on the
        // stack before the loop, but incurs large overhead to copy object.
        // Another solution is to just copy out the Matrix33 from the
        // ColorSystem and not call any methods by reproducing the method's
        // code here.  In this case though, because the conversion code can
        // handle any input lambda value, we will just perform all the work
        // unmasked which avoids the issue.
        //if (wR.mask()[lane]) {
        Color3 rgb = cs.XYZ_to_RGB(wavelength_color_XYZ(lambda));
        rgb *= 1.0 / 2.52;  // Empirical scale from lg to make all comps <= 1

        clamp_zero(rgb);
        //}
        wR[lane] = rgb;
    }
}



OSL_BATCHOP void
__OSL_OP(prepend_color_from_vs)(void* bsg_, void* c_, const char* from)
{
    const ColorSystem& cs = cs_from_bsg(bsg_);

    Color3& c(*(Color3*)c_);
    c = cs.to_rgb(USTR(from), c, context_from_bsg(bsg_));
}

namespace {

// NOTE: keep implementation as mirror of ColorSystem::to_rgb
void
wide_prepend_color_from(ShadingContext* ctx, const ColorSystem& cs,
                        Masked<Color3> wR, ustring fromspace)
{
    // Rather than attempt outer loop vectorization of ColorSystem::to_rgb
    // we will pull it's implementation up and insert SIMD loops inside
    // the uniform branches
    if (fromspace == Strings::RGB || fromspace == Strings::rgb
        || fromspace == cs.colorspace()) {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Color3 C = wR[lane];
            wR[lane] = C;
        }
        return;
    }
    if (fromspace == Strings::hsv) {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Color3 C = wR[lane];
            if (wR.mask()[lane]) {
                Color3 R             = hsv_to_rgb(C);
                wR[ActiveLane(lane)] = R;
            }
        }
        return;
    }
    if (fromspace == Strings::hsl) {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Color3 C = wR[lane];
            if (wR.mask()[lane]) {
                Color3 R             = hsl_to_rgb(C);
                wR[ActiveLane(lane)] = R;
            }
        }
        return;
    }
    if (fromspace == Strings::YIQ) {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Color3 C = wR[lane];
            if (wR.mask()[lane]) {
                Color3 R             = YIQ_to_rgb(C);
                wR[ActiveLane(lane)] = R;
            }
        }
        return;
    }
    if (fromspace == Strings::XYZ) {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Color3 C = wR[lane];
            if (wR.mask()[lane]) {
                Color3 R             = cs.XYZ_to_RGB(C);
                wR[ActiveLane(lane)] = R;
            }
        }
        return;
    }
    if (fromspace == Strings::xyY) {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Color3 C = wR[lane];
            if (wR.mask()[lane]) {
                Color3 R             = cs.XYZ_to_RGB(xyY_to_XYZ(C));
                wR[ActiveLane(lane)] = R;
            }
        }
        return;
    }

    // Serialize calls to ocio
    wR.mask().foreach ([=, &cs](ActiveLane lane) -> void {
        Color3 C = wR[lane];
        Color3 R = cs.ocio_transform(fromspace, Strings::RGB, C, ctx);
        wR[lane] = R;
    });
}

}  // namespace



OSL_BATCHOP void
__OSL_MASKED_OP2(prepend_color_from, Wv, s)(void* bsg_, void* c_,
                                            const char* from,
                                            unsigned int mask_value)
{
    const ColorSystem& cs = cs_from_bsg(bsg_);
    ShadingContext* ctx   = context_from_bsg(bsg_);

    Masked<Color3> wR(c_, Mask(mask_value));
    ustring fromspace = USTR(from);

    wide_prepend_color_from(ctx, cs, wR, fromspace);
}



OSL_BATCHOP void
__OSL_MASKED_OP2(prepend_color_from, Wv, Ws)(void* bsg_, void* c_, void* from_,
                                             unsigned int mask_value)
{
    const ColorSystem& cs = cs_from_bsg(bsg_);
    ShadingContext* ctx   = context_from_bsg(bsg_);

    Wide<const ustring> wFrom(from_);
    foreach_unique(wFrom, Mask(mask_value),
                   [=, &cs](const ustring& from, Mask from_mask) {
                       // Reuse the uniform from implementation by restricting results to
                       // just the lanes with the same value of "from".
                       Masked<Color3> wsub_result(c_, from_mask);
                       wide_prepend_color_from(ctx, cs, wsub_result, from);
                   });
}



namespace {

// Note: Clang 14 seems to no longer allow vectorizing these loops
#if ((OSL_CLANG_VERSION && OSL_CLANG_VERSION < 140000) \
     || OSL_INTEL_CLASSIC_COMPILER_VERSION || OSL_INTEL_LLVM_COMPILER_VERSION)
#    define WIDE_TRANSFORMC_OMP_SIMD_LOOP(...) OSL_OMP_SIMD_LOOP(__VA_ARGS__)
#else
#    define WIDE_TRANSFORMC_OMP_SIMD_LOOP(...)
#endif

template<typename COLOR>
OSL_NOINLINE void
wide_transformc(const ColorSystem cs, ustring fromspace, ustring tospace,
                Masked<COLOR> wOutput, Wide<const COLOR> wInput,
                ShadingContext* context);

// NOTE: keep implementation as mirror of ColorSystem::transformc
template<typename COLOR>
void
wide_transformc(const ColorSystem cs, ustring fromspace, ustring tospace,
                Masked<COLOR> wOutput, Wide<const COLOR> wInput,
                ShadingContext* context)
{
    // Rather than attempt outer loop vectorization of ColorSystem::transformc
    // we will pull it's implementation up and insert SIMD loops inside
    // the uniform branches
    bool use_colorconfig = false;
    Block<COLOR> bCrgb;
    Wide<COLOR> wCrgb(bCrgb);
    if (fromspace == Strings::RGB || fromspace == Strings::rgb
        || fromspace == Strings::linear || fromspace == cs.colorspace()) {
        WIDE_TRANSFORMC_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            COLOR C     = wInput[lane];
            wCrgb[lane] = C;
        }
    } else if (fromspace == Strings::hsv) {
        WIDE_TRANSFORMC_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            COLOR C = wInput[lane];
            if (wOutput.mask()[lane]) {
                COLOR R                 = hsv_to_rgb(C);
                wCrgb[ActiveLane(lane)] = R;
            }
        }
    } else if (fromspace == Strings::hsl) {
        WIDE_TRANSFORMC_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            COLOR C = wInput[lane];
            if (wOutput.mask()[lane]) {
                COLOR R                 = hsl_to_rgb(C);
                wCrgb[ActiveLane(lane)] = R;
            }
        }
    } else if (fromspace == Strings::YIQ) {
        WIDE_TRANSFORMC_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            COLOR C = wInput[lane];
            if (wOutput.mask()[lane]) {
                COLOR R                 = YIQ_to_rgb(C);
                wCrgb[ActiveLane(lane)] = R;
            }
        }
    } else if (fromspace == Strings::XYZ) {
        WIDE_TRANSFORMC_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            COLOR C = wInput[lane];
            if (wOutput.mask()[lane]) {
                COLOR R                 = cs.XYZ_to_RGB(C);
                wCrgb[ActiveLane(lane)] = R;
            }
        }
    } else if (fromspace == Strings::xyY) {
        WIDE_TRANSFORMC_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            COLOR C = wInput[lane];
            if (wOutput.mask()[lane]) {
                COLOR R                 = cs.XYZ_to_RGB(xyY_to_XYZ(C));
                wCrgb[ActiveLane(lane)] = R;
            }
        }
    } else if (fromspace == Strings::sRGB) {
        WIDE_TRANSFORMC_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            COLOR C = wInput[lane];
            if (wOutput.mask()[lane]) {
                COLOR R                 = sRGB_to_linear(C);
                wCrgb[ActiveLane(lane)] = R;
            }
        }
    } else {
        use_colorconfig = true;
    }

    if (use_colorconfig) {
        // do things the ColorConfig way, so skip all these other clauses...
    } else if (tospace == Strings::RGB || tospace == Strings::rgb
               || tospace == Strings::linear || tospace == cs.colorspace()) {
        WIDE_TRANSFORMC_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            COLOR C       = wCrgb[lane];
            wOutput[lane] = C;
        }
    } else if (tospace == Strings::hsv) {
        WIDE_TRANSFORMC_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            COLOR Crgb = wCrgb[lane];
            if (wOutput.mask()[lane]) {
                COLOR Cto                 = rgb_to_hsv(Crgb);
                wOutput[ActiveLane(lane)] = Cto;
            }
        }
    } else if (tospace == Strings::hsl) {
        WIDE_TRANSFORMC_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            COLOR Crgb = wCrgb[lane];
            if (wOutput.mask()[lane]) {
                COLOR Cto                 = rgb_to_hsl(Crgb);
                wOutput[ActiveLane(lane)] = Cto;
            }
        }
    } else if (tospace == Strings::YIQ) {
        WIDE_TRANSFORMC_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            COLOR Crgb = wCrgb[lane];
            if (wOutput.mask()[lane]) {
                COLOR Cto                 = rgb_to_YIQ(Crgb);
                wOutput[ActiveLane(lane)] = Cto;
            }
        }
    } else if (tospace == Strings::XYZ) {
        WIDE_TRANSFORMC_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            COLOR Crgb = wCrgb[lane];
            if (wOutput.mask()[lane]) {
                COLOR Cto                 = cs.RGB_to_XYZ(Crgb);
                wOutput[ActiveLane(lane)] = Cto;
            }
        }
    } else if (tospace == Strings::xyY) {
        WIDE_TRANSFORMC_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            COLOR Crgb = wCrgb[lane];
            if (wOutput.mask()[lane]) {
                COLOR Cto                 = XYZ_to_xyY(cs.RGB_to_XYZ(Crgb));
                wOutput[ActiveLane(lane)] = Cto;
            }
        }
    } else if (tospace == Strings::sRGB) {
        WIDE_TRANSFORMC_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            COLOR Crgb = wCrgb[lane];
            if (wOutput.mask()[lane]) {
                COLOR Cto                 = linear_to_sRGB(Crgb);
                wOutput[ActiveLane(lane)] = Cto;
            }
        }
    } else {
        use_colorconfig = true;
    }

    if (use_colorconfig) {
        // Serialize calls to ocio
        wOutput.mask().foreach ([=, &cs](ActiveLane lane) -> void {
            COLOR C       = wInput[lane];
            COLOR Cto     = cs.ocio_transform(fromspace, tospace, C, context);
            wOutput[lane] = Cto;
        });
    }
}

#undef WIDE_TRANSFORMC_OMP_SIMD_LOOP

}  // namespace



OSL_BATCHOP void
__OSL_MASKED_OP3(transform_color, Wv, s,
                 s)(void* bsg_, void* Cin, int Cin_derivs, void* Cout,
                    int Cout_derivs, ustring_pod from_, ustring_pod to_,
                    unsigned int mask_value)
{
    const ColorSystem& cs = cs_from_bsg(bsg_);
    ShadingContext* ctx   = context_from_bsg(bsg_);

    const ustring& from = USTR(from_);
    const ustring& to   = USTR(to_);

    if (Cout_derivs) {
        if (Cin_derivs) {
            Masked<Dual2<Color3>> wOutput(Cout, Mask(mask_value));
            Wide<const Dual2<Color3>> wInput(Cin);

            wide_transformc(cs, from, to, wOutput, wInput, ctx);
            return;
        } else {
            // We had output derivs, but not input. Zero the output
            // derivs and fall through to the non-deriv case.
            MaskedDx<Color3> wOutputDx(Cout, Mask(mask_value));
            MaskedDy<Color3> wOutputDy(Cout, Mask(mask_value));

            assign_all(wOutputDx, Color3(0.0f));
            assign_all(wOutputDy, Color3(0.0f));
        }
    }

    // No-derivs case
    Masked<Color3> wOutput(Cout, Mask(mask_value));
    Wide<const Color3> wInput(Cin);
    wide_transformc(cs, from, to, wOutput, wInput, ctx);
    return;
}



OSL_BATCHOP void
__OSL_OP3(transform_color, v, s, s)(void* bsg_, void* Cin, int Cin_derivs,
                                    void* Cout, int Cout_derivs,
                                    ustring_pod from_, ustring_pod to_)
{
    const ColorSystem& cs = cs_from_bsg(bsg_);
    ShadingContext* ctx   = context_from_bsg(bsg_);

    const ustring& from = USTR(from_);
    const ustring& to   = USTR(to_);

    if (Cout_derivs) {
        if (Cin_derivs) {
            DCOL(Cout) = cs.transformc(from, to, DCOL(Cin), ctx);
            return;
        } else {
            // We had output derivs, but not input. Zero the output
            // derivs and fall through to the non-deriv case.
            ((Color3*)Cout)[1].setValue(0.0f, 0.0f, 0.0f);
            ((Color3*)Cout)[2].setValue(0.0f, 0.0f, 0.0f);
        }
    }

    // No-derivs case
    COL(Cout) = cs.transformc(from, to, COL(Cin), ctx);
    return;
}


}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_END

#include "undef_opname_macros.h"
