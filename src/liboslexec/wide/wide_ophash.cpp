// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of hash operations.
///
/////////////////////////////////////////////////////////////////////////
#include <OSL/oslconfig.h>

#include <OSL/oslnoise.h>
#include <OSL/wide.h>

#include "oslexec_pvt.h"


OSL_NAMESPACE_ENTER
namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)

#include "define_opname_macros.h"

namespace {

OSL_FORCEINLINE OSL_HOSTDEVICE int
bitcast_to_uint(float x)
{
    return OSL::bitcast<unsigned int>(x);
}


// TODO: to avoid ansi aliasing issues,
// suggest replacing inthashf (const float *x) with inthashv
inline OSL_HOSTDEVICE int
inthashv(const Vec3& v)
{
    return static_cast<int>(pvt::inthash(bitcast_to_uint(v.x),
                                         bitcast_to_uint(v.y),
                                         bitcast_to_uint(v.z)));
}


// TODO: to avoid ansi aliasing issues,
// suggest replacing inthashf (const float *x, float y) with inthashvf
inline OSL_HOSTDEVICE int
inthashvf(const Vec3& v, float y)
{
    return static_cast<int>(
        pvt::inthash(bitcast_to_uint(v.x), bitcast_to_uint(v.y),
                     bitcast_to_uint(v.z), bitcast_to_uint(y)));
}


};  // End anonymous namespace


OSL_BATCHOP void
__OSL_OP2(hash, Wi, Wi)(void* r_, void* val_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const int> wval(val_);
        Wide<int> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            int val  = wval[lane];
            wr[lane] = pvt::inthashi(val);
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP2(hash, Wi, Wi)(void* r_, void* val_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const int> wval(val_);
        Masked<int> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            int val  = wval[lane];
            wr[lane] = pvt::inthashi(val);
        }
    }
}


OSL_BATCHOP void
__OSL_OP2(hash, Wi, Wf)(void* r_, void* val_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> wval(val_);
        Wide<int> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            float val = wval[lane];
            wr[lane]  = pvt::inthashf(val);
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP2(hash, Wi, Wf)(void* r_, void* val_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> wval(val_);
        Masked<int> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            float val = wval[lane];
            wr[lane]  = pvt::inthashf(val);
        }
    }
}



OSL_BATCHOP void
__OSL_OP3(hash, Wi, Wf, Wf)(void* r_, void* val_, void* val2_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> wval(val_);
        Wide<const float> wval2(val2_);
        Wide<int> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            float val  = wval[lane];
            float val2 = wval2[lane];
            wr[lane]   = pvt::inthashf(val, val2);
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(hash, Wi, Wf, Wf)(void* r_, void* val_, void* val2_,
                                   unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> wval(val_);
        Wide<const float> wval2(val2_);
        Masked<int> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            float val  = wval[lane];
            float val2 = wval2[lane];
            wr[lane]   = pvt::inthashf(val, val2);
        }
    }
}



OSL_BATCHOP void
__OSL_OP2(hash, Wi, Wv)(void* r_, void* val_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wval(val_);
        Wide<int> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 val = wval[lane];
            wr[lane] = inthashv(val);
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP2(hash, Wi, Wv)(void* r_, void* val_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wval(val_);
        Masked<int> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 val = wval[lane];
            wr[lane] = inthashv(val);
        }
    }
}



OSL_BATCHOP void
__OSL_OP3(hash, Wi, Wv, Wf)(void* r_, void* val_, void* val2_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wval(val_);
        Wide<const float> wval2(val2_);
        Wide<int> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 val   = wval[lane];
            float val2 = wval2[lane];
            wr[lane]   = inthashvf(val, val2);
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(hash, Wi, Wv, Wf)(void* r_, void* val_, void* val2_,
                                   unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wval(val_);
        Wide<const float> wval2(val2_);
        Masked<int> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 val   = wval[lane];
            float val2 = wval2[lane];
            wr[lane]   = inthashvf(val, val2);
        }
    }
}

}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_EXIT
