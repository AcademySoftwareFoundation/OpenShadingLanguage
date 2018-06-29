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

#include <limits>

#include "oslexec_pvt.h"
#include <OSL/oslnoise.h>
#include <OSL/dual_vec.h>
#include <OSL/Imathx.h>

#include <OpenImageIO/fmath.h>

#include "opnoise.h"

using namespace OSL;

OSL_NAMESPACE_ENTER
namespace pvt {



void gabor (MaskedAccessor<Dual2<float>,__OSL_SIMD_LANE_COUNT> wResult,
            ConstWideAccessor<Dual2<float>,__OSL_SIMD_LANE_COUNT> wX,
            const NoiseParams *opt);

void gabor (MaskedAccessor<Dual2<float>,__OSL_SIMD_LANE_COUNT> wResult,
            ConstWideAccessor<Dual2<float>,__OSL_SIMD_LANE_COUNT> wX,
            ConstWideAccessor<Dual2<float>,__OSL_SIMD_LANE_COUNT> wY,
            const NoiseParams *opt);

void gabor (MaskedAccessor<Dual2<float>,__OSL_SIMD_LANE_COUNT> wResult,
            ConstWideAccessor<Dual2<Vec3>,__OSL_SIMD_LANE_COUNT> wP,
            const NoiseParams *opt);



void gabor3 (MaskedAccessor<Dual2<Vec3>,__OSL_SIMD_LANE_COUNT> wResult,
             ConstWideAccessor<Dual2<float>,__OSL_SIMD_LANE_COUNT> wX,
             const NoiseParams *opt);

void gabor3 (MaskedAccessor<Dual2<Vec3>,__OSL_SIMD_LANE_COUNT> wResult,
             ConstWideAccessor<Dual2<float>,__OSL_SIMD_LANE_COUNT> wX,
             ConstWideAccessor<Dual2<float>,__OSL_SIMD_LANE_COUNT> wY,
             const NoiseParams *opt);

void gabor3 (MaskedAccessor<Dual2<Vec3>,__OSL_SIMD_LANE_COUNT> wResult,
             ConstWideAccessor<Dual2<Vec3>,__OSL_SIMD_LANE_COUNT> wP,
             const NoiseParams *opt);





void pgabor (MaskedAccessor<Dual2<float>,__OSL_SIMD_LANE_COUNT> wResult,
            ConstWideAccessor<Dual2<float>,__OSL_SIMD_LANE_COUNT> wX,
            ConstWideAccessor<float,__OSL_SIMD_LANE_COUNT> wPX,
            const NoiseParams *opt);

void pgabor (MaskedAccessor<Dual2<float>,__OSL_SIMD_LANE_COUNT> wResult,
            ConstWideAccessor<Dual2<float>,__OSL_SIMD_LANE_COUNT> wX,
            ConstWideAccessor<Dual2<float>,__OSL_SIMD_LANE_COUNT> wY,
            ConstWideAccessor<float,__OSL_SIMD_LANE_COUNT> wPX,
            ConstWideAccessor<float,__OSL_SIMD_LANE_COUNT> wPY,
            const NoiseParams *opt);

void pgabor (MaskedAccessor<Dual2<float>,__OSL_SIMD_LANE_COUNT> wResult,
            ConstWideAccessor<Dual2<Vec3>,__OSL_SIMD_LANE_COUNT> wP,
            ConstWideAccessor<Vec3,__OSL_SIMD_LANE_COUNT> wPP,
            const NoiseParams *opt);



void pgabor3 (MaskedAccessor<Dual2<Vec3>,__OSL_SIMD_LANE_COUNT> wResult,
             ConstWideAccessor<Dual2<float>,__OSL_SIMD_LANE_COUNT> wX,
             ConstWideAccessor<float,__OSL_SIMD_LANE_COUNT> wPX,
             const NoiseParams *opt);

void pgabor3 (MaskedAccessor<Dual2<Vec3>,__OSL_SIMD_LANE_COUNT> wResult,
             ConstWideAccessor<Dual2<float>,__OSL_SIMD_LANE_COUNT> wX,
             ConstWideAccessor<Dual2<float>,__OSL_SIMD_LANE_COUNT> wY,
             ConstWideAccessor<float,__OSL_SIMD_LANE_COUNT> wPX,
             ConstWideAccessor<float,__OSL_SIMD_LANE_COUNT> wPY,
             const NoiseParams *opt);

void pgabor3 (MaskedAccessor<Dual2<Vec3>,__OSL_SIMD_LANE_COUNT> wResult,
             ConstWideAccessor<Dual2<Vec3>,__OSL_SIMD_LANE_COUNT> wP,
             ConstWideAccessor<Vec3,__OSL_SIMD_LANE_COUNT> wPP,
             const NoiseParams *opt);



/***********************************************************************
 * noise routines callable by the LLVM-generated code.
 */



NOISE_IMPL (cellnoise, CellNoise)
#define __OSL_XMACRO_ARGS (cellnoise, CellNoise, __OSL_SIMD_LANE_COUNT)
#include "opnoise_impl_wide_xmacro.h"


#if 0 // moved to opnoise_perlin.cpp to enable parallel builds
	NOISE_IMPL (noise, Noise)
	NOISE_IMPL_DERIV (noise, Noise)
#endif
#if 0 // moved to opnoise_uperlin.cpp to enable parallel builds
	NOISE_IMPL (snoise, SNoise)
	NOISE_IMPL_DERIV (snoise, SNoise)
#endif

#if 0 // moved to opnoise_simplex.cpp to enable parallel builds
	NOISE_IMPL (simplexnoise, SimplexNoise)
	NOISE_IMPL_DERIV (simplexnoise, SimplexNoise)
#endif

#if 0 // moved to opnoise_usimplex.cpp to enable parallel builds
    NOISE_IMPL (usimplexnoise, USimplexNoise)
    NOISE_IMPL_DERIV (usimplexnoise, USimplexNoise)
#endif

PNOISE_IMPL (pcellnoise, PeriodicCellNoise)
#define __OSL_XMACRO_ARGS (pcellnoise, PeriodicCellNoise, __OSL_SIMD_LANE_COUNT)
#include "opnoise_periodic_impl_wide_xmacro.h"


#if 0 // moved to oppnoise_uperlin.cpp to enable parallel builds
PNOISE_IMPL (pnoise, PeriodicNoise)
PNOISE_IMPL_DERIV (pnoise, PeriodicNoise)
#endif
#if 0 // moved to oppnoise_perlin.cpp to enable parallel builds
PNOISE_IMPL (psnoise, PeriodicSNoise)
PNOISE_IMPL_DERIV (psnoise, PeriodicSNoise)
#endif



struct GaborNoise {
    GaborNoise () { }

    // Gabor always uses derivatives, so dual versions only

    inline void operator() (ustring noisename, Dual2<float> &result,
                            const Dual2<float> &x,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = gabor (x, opt);
    }

    inline void operator() (ustring noisename, Dual2<float> &result,
                            const Dual2<float> &x, const Dual2<float> &y,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = gabor (x, y, opt);
    }

    inline void operator() (ustring noisename, Dual2<float> &result,
                            const Dual2<Vec3> &p,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = gabor (p, opt);
    }
    
    inline void operator() (ustring noisename, Dual2<float> &result,
                            const Dual2<Vec3> &p, const Dual2<float> &t,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        // FIXME -- This is very broken, we are ignoring 4D!
        result = gabor (p, opt);
    }

    inline void operator() (ustring noisename, Dual2<Vec3> &result,
                            const Dual2<float> &x,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = gabor3 (x, opt);
    }

    inline void operator() (ustring noisename, Dual2<Vec3> &result,
                            const Dual2<float> &x, const Dual2<float> &y,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = gabor3 (x, y, opt);
    }

    inline void operator() (ustring noisename, Dual2<Vec3> &result,
                            const Dual2<Vec3> &p,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = gabor3 (p, opt);
    }

    inline void operator() (ustring noisename, Dual2<Vec3> &result,
                            const Dual2<Vec3> &p, const Dual2<float> &t,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        // FIXME -- This is very broken, we are ignoring 4D!
        result = gabor3 (p, opt);
    }

};



struct GaborPNoise {
    GaborPNoise () { }

    // Gabor always uses derivatives, so dual versions only

    inline void operator() (ustring noisename, Dual2<float> &result,
                            const Dual2<float> &x, float px,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = pgabor (x, px, opt);
    }

    inline void operator() (ustring noisename, Dual2<float> &result,
                            const Dual2<float> &x, const Dual2<float> &y,
                            float px, float py,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = pgabor (x, y, px, py, opt);
    }

    inline void operator() (ustring noisename, Dual2<float> &result,
                            const Dual2<Vec3> &p, const Vec3 &pp,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = pgabor (p, pp, opt);
    }

    inline void operator() (ustring noisename, Dual2<float> &result,
                            const Dual2<Vec3> &p, const Dual2<float> &t,
                            const Vec3 &pp, float tp,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        // FIXME -- This is very broken, we are ignoring 4D!
        result = pgabor (p, pp, opt);
    }

    inline void operator() (ustring noisename, Dual2<Vec3> &result,
                            const Dual2<float> &x, float px,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = pgabor3 (x, px, opt);
    }

    inline void operator() (ustring noisename, Dual2<Vec3> &result,
                            const Dual2<float> &x, const Dual2<float> &y,
                            float px, float py,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = pgabor3 (x, y, px, py, opt);
    }

    inline void operator() (ustring noisename, Dual2<Vec3> &result,
                            const Dual2<Vec3> &p, const Vec3 &pp,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        result = pgabor3 (p, pp, opt);
    }

    inline void operator() (ustring noisename, Dual2<Vec3> &result,
                            const Dual2<Vec3> &p, const Dual2<float> &t,
                            const Vec3 &pp, float tp,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        // FIXME -- This is very broken, we are ignoring 4D!
        result = pgabor3 (p, pp, opt);
    }
};



NOISE_IMPL_DERIV_OPT (gabornoise, GaborNoise)
#define __OSL_XMACRO_ARGS (gabornoise, gabor, __OSL_SIMD_LANE_COUNT)
#include "opnoise_impl_opt_wide_xmacro.h"

PNOISE_IMPL_DERIV_OPT (gaborpnoise, GaborPNoise)
//#define __OSL_XMACRO_ARGS (gaborpnoise, pgabor, __OSL_SIMD_LANE_COUNT)
//#include "opnoise_periodic_impl_opt_wide_xmacro.h"



struct NullNoise {
    NullNoise () { }
    inline void operator() (float &result, float x) const { result = 0.0f; }
    inline void operator() (float &result, float x, float y) const { result = 0.0f; }
    inline void operator() (float &result, const Vec3 &p) const { result = 0.0f; }
    inline void operator() (float &result, const Vec3 &p, float t) const { result = 0.0f; }
    inline void operator() (Vec3 &result, float x) const { result = v(); }
    inline void operator() (Vec3 &result, float x, float y) const { result = v(); }
    inline void operator() (Vec3 &result, const Vec3 &p) const { result = v(); }
    inline void operator() (Vec3 &result, const Vec3 &p, float t) const { result = v(); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x,
                            int seed=0) const { result.set (0.0f, 0.0f, 0.0f); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x,
                            const Dual2<float> &y, int seed=0) const { result.set (0.0f, 0.0f, 0.0f); }
    inline void operator() (Dual2<float> &result, const Dual2<Vec3> &p,
                            int seed=0) const { result.set (0.0f, 0.0f, 0.0f); }
    inline void operator() (Dual2<float> &result, const Dual2<Vec3> &p,
                            const Dual2<float> &t, int seed=0) const { result.set (0.0f, 0.0f, 0.0f); }
    inline void operator() (Dual2<Vec3> &result, const Dual2<float> &x) const { result.set (v(), v(), v()); }
    inline void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y) const {  result.set (v(), v(), v()); }
    inline void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p) const {  result.set (v(), v(), v()); }
    inline void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const { result.set (v(), v(), v()); }
    inline Vec3 v () const { return Vec3(0.0f, 0.0f, 0.0f); };
};

struct UNullNoise {
    UNullNoise () { }
    inline void operator() (float &result, float x) const { result = 0.5f; }
    inline void operator() (float &result, float x, float y) const { result = 0.5f; }
    inline void operator() (float &result, const Vec3 &p) const { result = 0.5f; }
    inline void operator() (float &result, const Vec3 &p, float t) const { result = 0.5f; }
    inline void operator() (Vec3 &result, float x) const { result = v(); }
    inline void operator() (Vec3 &result, float x, float y) const { result = v(); }
    inline void operator() (Vec3 &result, const Vec3 &p) const { result = v(); }
    inline void operator() (Vec3 &result, const Vec3 &p, float t) const { result = v(); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x,
                            int seed=0) const { result.set (0.5f, 0.5f, 0.5f); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x,
                            const Dual2<float> &y, int seed=0) const { result.set (0.5f, 0.5f, 0.5f); }
    inline void operator() (Dual2<float> &result, const Dual2<Vec3> &p,
                            int seed=0) const { result.set (0.5f, 0.5f, 0.5f); }
    inline void operator() (Dual2<float> &result, const Dual2<Vec3> &p,
                            const Dual2<float> &t, int seed=0) const { result.set (0.5f, 0.5f, 0.5f); }
    inline void operator() (Dual2<Vec3> &result, const Dual2<float> &x) const { result.set (v(), v(), v()); }
    inline void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y) const {  result.set (v(), v(), v()); }
    inline void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p) const {  result.set (v(), v(), v()); }
    inline void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const { result.set (v(), v(), v()); }
    inline Vec3 v () const { return Vec3(0.5f, 0.5f, 0.5f); };
};

NOISE_IMPL (nullnoise, NullNoise)
NOISE_IMPL_DERIV (nullnoise, NullNoise)
NOISE_IMPL (unullnoise, UNullNoise)
NOISE_IMPL_DERIV (unullnoise, UNullNoise)




struct GenericNoise {
    GenericNoise () { }

    // Template on R, S, and T to be either float or Vec3

    // dual versions -- this is always called with derivs

    template<class R, class S>
    inline void operator() (ustring name, Dual2<R> &result, const Dual2<S> &s,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        if (name == Strings::uperlin || name == Strings::noise) {
            Noise noise;
            noise(result, s);
        } else if (name == Strings::perlin || name == Strings::snoise) {
            SNoise snoise;
            snoise(result, s);
        } else if (name == Strings::simplexnoise || name == Strings::simplex) {
            SimplexNoise simplexnoise;
            simplexnoise(result, s);
        } else if (name == Strings::usimplexnoise || name == Strings::usimplex) {
            USimplexNoise usimplexnoise;
            usimplexnoise(result, s);
        } else if (name == Strings::cell) {
            CellNoise cellnoise;
            cellnoise(result.val(), s.val());
            result.clear_d();
        } else if (name == Strings::gabor) {
            GaborNoise gnoise;
            gnoise (name, result, s, sg, opt);
        } else if (name == Strings::null) {
            NullNoise noise; noise(result, s);
        } else if (name == Strings::unull) {
            UNullNoise noise; noise(result, s);
        } else {
            ((ShadingContext *)sg->context)->error ("Unknown noise type \"%s\"", name.c_str());
        }
    }

    template<class R, class S, class T>
    inline void operator() (ustring name, Dual2<R> &result,
                            const Dual2<S> &s, const Dual2<T> &t,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        if (name == Strings::uperlin || name == Strings::noise) {
            Noise noise;
            noise(result, s, t);
        } else if (name == Strings::perlin || name == Strings::snoise) {
            SNoise snoise;
            snoise(result, s, t);
        } else if (name == Strings::simplexnoise || name == Strings::simplex) {
            SimplexNoise simplexnoise;
            simplexnoise(result, s, t);
        } else if (name == Strings::usimplexnoise || name == Strings::usimplex) {
            USimplexNoise usimplexnoise;
            usimplexnoise(result, s, t);
        } else if (name == Strings::cell) {
            CellNoise cellnoise;
            cellnoise(result.val(), s.val(), t.val());
            result.clear_d();
        } else if (name == Strings::gabor) {
            GaborNoise gnoise;
            gnoise (name, result, s, t, sg, opt);
        } else if (name == Strings::null) {
            NullNoise noise; noise(result, s, t);
        } else if (name == Strings::unull) {
            UNullNoise noise; noise(result, s, t);
        } else {
            ((ShadingContext *)sg->context)->error ("Unknown noise type \"%s\"", name.c_str());
        }
    }
};


NOISE_IMPL_DERIV_OPT (genericnoise, GenericNoise)


struct GenericPNoise {
    GenericPNoise () { }

    // Template on R, S, and T to be either float or Vec3

    // dual versions -- this is always called with derivs

    template<class R, class S>
    inline void operator() (ustring name, Dual2<R> &result, const Dual2<S> &s,
                            const S &sp,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        if (name == Strings::uperlin || name == Strings::noise) {
            PeriodicNoise noise;
            noise(result, s, sp);
        } else if (name == Strings::perlin || name == Strings::snoise) {
            PeriodicSNoise snoise;
            snoise(result, s, sp);
        } else if (name == Strings::cell) {
            PeriodicCellNoise cellnoise;
            cellnoise(result.val(), s.val(), sp);
            result.clear_d();
        } else if (name == Strings::gabor) {
            GaborPNoise gnoise;
            gnoise (name, result, s, sp, sg, opt);
        } else {
            ((ShadingContext *)sg->context)->error ("Unknown noise type \"%s\"", name.c_str());
        }
    }

    template<class R, class S, class T>
    inline void operator() (ustring name, Dual2<R> &result,
                            const Dual2<S> &s, const Dual2<T> &t,
                            const S &sp, const T &tp,
                            ShaderGlobals *sg, const NoiseParams *opt) const {
        if (name == Strings::uperlin || name == Strings::noise) {
            PeriodicNoise noise;
            noise(result, s, t, sp, tp);
        } else if (name == Strings::perlin || name == Strings::snoise) {
            PeriodicSNoise snoise;
            snoise(result, s, t, sp, tp);
        } else if (name == Strings::cell) {
            PeriodicCellNoise cellnoise;
            cellnoise(result.val(), s.val(), t.val(), sp, tp);
            result.clear_d();
        } else if (name == Strings::gabor) {
            GaborPNoise gnoise;
            gnoise (name, result, s, t, sp, tp, sg, opt);
        } else {
            ((ShadingContext *)sg->context)->error ("Unknown noise type \"%s\"", name.c_str());
        }
    }
};


PNOISE_IMPL_DERIV_OPT (genericpnoise, GenericPNoise)


// Utility: retrieve a pointer to the ShadingContext's noise params
// struct, also re-initialize its contents.
OSL_SHADEOP void *
osl_get_noise_options (void *sg_)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    RendererServices::NoiseOpt *opt = sg->context->noise_options_ptr ();
    new (opt) RendererServices::NoiseOpt;
    return opt;
}

// Utility: retrieve a pointer to the ShadingContext's noise params
// struct, also re-initialize its contents.
OSL_SHADEOP void *
osl_wide_get_noise_options (void *sgb_)
{
	ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    RendererServices::NoiseOpt *opt = sgb->uniform().context->noise_options_ptr ();
    new (opt) RendererServices::NoiseOpt;
    return opt;
}
 

OSL_SHADEOP void
osl_noiseparams_set_anisotropic (void *opt, int a)
{
    ((RendererServices::NoiseOpt *)opt)->anisotropic = a;
}



OSL_SHADEOP void
osl_noiseparams_set_do_filter (void *opt, int a)
{
    ((RendererServices::NoiseOpt *)opt)->do_filter = a;
}



OSL_SHADEOP void
osl_noiseparams_set_direction (void *opt, void *dir)
{
    ((RendererServices::NoiseOpt *)opt)->direction = VEC(dir);
}



OSL_SHADEOP void
osl_noiseparams_set_bandwidth (void *opt, float b)
{
    ((RendererServices::NoiseOpt *)opt)->bandwidth = b;
}



OSL_SHADEOP void
osl_noiseparams_set_impulses (void *opt, float i)
{
    ((RendererServices::NoiseOpt *)opt)->impulses = i;
}



OSL_SHADEOP void
osl_count_noise (void *sg_)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    sg->context->shadingsys().count_noise ();
}


} // namespace pvt
OSL_NAMESPACE_EXIT

