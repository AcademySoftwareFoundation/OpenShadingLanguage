// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <optix.h>
#include <optix_device.h>

#define OPTIX_COMPATIBILITY 7
#include <OSL/oslclosure.h>

#include <cuda_runtime.h>
#include <optix_device.h>

#include "rend_lib.h"

#define MEMCPY_ALIGNED(dst, src, size, alignment)    \
    memcpy(__builtin_assume_aligned(dst, alignment), \
           __builtin_assume_aligned(src, alignment), size);

OSL_NAMESPACE_BEGIN
namespace pvt {
extern __device__ CUdeviceptr s_color_system;
extern __device__ CUdeviceptr osl_printf_buffer_start;
extern __device__ CUdeviceptr osl_printf_buffer_end;
extern __device__ uint64_t test_str_1;
extern __device__ uint64_t test_str_2;
extern __device__ uint64_t num_named_xforms;
extern __device__ CUdeviceptr xform_name_buffer;
extern __device__ CUdeviceptr xform_buffer;
}  // namespace pvt
OSL_NAMESPACE_END

// These functions are declared extern to prevent name mangling.
extern "C" {

// add OptiX entry point to prevent OptiX from discarding the module
__global__ void
__direct_callable__dummy_rend_lib()
{
}


#define IS_STRING(type) (type.basetype == OSL::TypeDesc::STRING)
#define IS_PTR(type)    (type.basetype == OSL::TypeDesc::PTR)
#define IS_COLOR(type)  (type.vecsemantics == OSL::TypeDesc::COLOR)


__device__ bool
rend_get_userdata(OSL::ustringhash name, void* data, int data_size,
                  const OSL::TypeDesc& type, int index)
{
    // Perform a userdata lookup using the parameter name, type, and
    // userdata index. If there is a match, memcpy the value into data and
    // return 1.

    if (IS_PTR(type) && name == OSL::Hashes::colorsystem) {
        *(void**)data = *reinterpret_cast<void**>(&OSL::pvt::s_color_system);
        return true;
    }
    // TODO: This is temporary code for initial testing and demonstration.
    if (IS_STRING(type) && name == HDSTR(OSL::pvt::test_str_1)) {
        MEMCPY_ALIGNED(data, &OSL::pvt::test_str_2, 8, alignof(float));
        return true;
    }

    return false;
}

#undef IS_COLOR
#undef IS_STRING
#undef IS_PTR


__device__ int
osl_bind_interpolated_param(void* sg_, OSL::ustringhash_pod name_,
                            long long type, int userdata_has_derivs,
                            void* userdata_data, int symbol_has_derivs,
                            void* symbol_data, int symbol_data_size,
                            char* userdata_initialized, int userdata_index)
{
    char status = *userdata_initialized;
    if (status == 0) {
        OSL::ustringhash name = OSL::ustringhash_from(name_);
        bool ok = rend_get_userdata(name, userdata_data, symbol_data_size,
                                    (*(OSL::TypeDesc*)&type), userdata_index);
        *userdata_initialized = status = 1 + ok;
    }
    if (status == 2) {
        MEMCPY_ALIGNED(symbol_data, userdata_data, symbol_data_size,
                       alignof(float));
        return 1;
    }
    return 0;
}


__device__ int
osl_strlen_is(OSL::ustringhash_pod str)
{
    //return HDSTR(str).length();
    return 0;
}


__device__ int
osl_hash_is(OSL::ustringhash_pod str)
{
    return static_cast<int>(str);
}


__device__ int
osl_getchar_isi(OSL::ustringhash_pod str, int index)
{
    //        return (str && unsigned(index) < HDSTR(str).length())
    //            ? str[index] : 0;
    return 0;
}



// Printing is handled by the host.  Copy format string's hash and
// all the arguments to our print buffer.
// Note:  the first element of 'args' is the size of the argument list
__device__ void
osl_printf(void* sg_, OSL::ustringhash_pod fmt_str_hash, void* args)
{
    uint64_t args_size = reinterpret_cast<uint64_t*>(args)[0];

    // This can be used to limit printing to one Cuda thread for debugging
    // if (launch_index.x == 0 && launch_index.y == 0)

    CUdeviceptr copy_start = atomicAdd(&OSL::pvt::osl_printf_buffer_start,
                                       args_size + sizeof(args_size)
                                           + sizeof(fmt_str_hash));

    // Only perform copy if there's enough space
    if (copy_start + args_size + sizeof(args_size) + sizeof(fmt_str_hash)
        < OSL::pvt::osl_printf_buffer_end) {
        memcpy(reinterpret_cast<void*>(copy_start), &fmt_str_hash,
               sizeof(fmt_str_hash));
        memcpy(reinterpret_cast<void*>(copy_start + sizeof(fmt_str_hash)),
               &args_size, sizeof(args_size));
        memcpy(reinterpret_cast<void*>(copy_start + sizeof(fmt_str_hash)
                                       + sizeof(args_size)),
               reinterpret_cast<char*>(args) + sizeof(args_size), args_size);
    }
}


#define OSL_TEXTURE_SET_HOSTDEVICE OSL_DEVICE
#define OSL_SHADEOP
using OSL::TextureOpt;
using OSL::TraceOpt;
using OSL::ustring;
using OSL::ustringhash;
using OSL::ustringhash_from;
using OSL::ustringhash_pod;


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_firstchannel(void* opt, int x)
{
    ((TextureOpt*)opt)->firstchannel = x;
}

OSL_TEXTURE_SET_HOSTDEVICE inline TextureOpt::Wrap
decode_wrapmode(ustringhash_pod name_)
{
    // TODO: Enable when decode_wrapmode has __device__ marker.
#ifndef __CUDA_ARCH__
    ustringhash name_hash = ustringhash_from(name_);
#    ifdef OIIO_TEXTURESYSTEM_SUPPORTS_DECODE_BY_USTRINGHASH
    return TextureOpt::decode_wrapmode(name_hash);
#    else
    ustring name = ustring_from(name_hash);
    return TextureOpt::decode_wrapmode(name);
#    endif
#else
    return TextureOpt::WrapDefault;
#endif
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE int
osl_texture_decode_wrapmode(ustringhash_pod name_)
{
    return (int)decode_wrapmode(name_);
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_swrap(void* opt, ustringhash_pod x_)
{
    ((TextureOpt*)opt)->swrap = decode_wrapmode(x_);
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_twrap(void* opt, ustringhash_pod x_)
{
    ((TextureOpt*)opt)->twrap = decode_wrapmode(x_);
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_rwrap(void* opt, ustringhash_pod x_)
{
    ((TextureOpt*)opt)->rwrap = decode_wrapmode(x_);
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_stwrap(void* opt, ustringhash_pod x_)
{
    TextureOpt::Wrap code     = decode_wrapmode(x_);
    ((TextureOpt*)opt)->swrap = code;
    ((TextureOpt*)opt)->twrap = code;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_swrap_code(void* opt, int mode)
{
    ((TextureOpt*)opt)->swrap = (TextureOpt::Wrap)mode;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_twrap_code(void* opt, int mode)
{
    ((TextureOpt*)opt)->twrap = (TextureOpt::Wrap)mode;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_rwrap_code(void* opt, int mode)
{
    ((TextureOpt*)opt)->rwrap = (TextureOpt::Wrap)mode;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_stwrap_code(void* opt, int mode)
{
    ((TextureOpt*)opt)->swrap = (TextureOpt::Wrap)mode;
    ((TextureOpt*)opt)->twrap = (TextureOpt::Wrap)mode;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_sblur(void* opt, float x)
{
    ((TextureOpt*)opt)->sblur = x;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_tblur(void* opt, float x)
{
    ((TextureOpt*)opt)->tblur = x;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_rblur(void* opt, float x)
{
    ((TextureOpt*)opt)->rblur = x;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_stblur(void* opt, float x)
{
    ((TextureOpt*)opt)->sblur = x;
    ((TextureOpt*)opt)->tblur = x;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_swidth(void* opt, float x)
{
    ((TextureOpt*)opt)->swidth = x;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_twidth(void* opt, float x)
{
    ((TextureOpt*)opt)->twidth = x;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_rwidth(void* opt, float x)
{
    ((TextureOpt*)opt)->rwidth = x;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_stwidth(void* opt, float x)
{
    ((TextureOpt*)opt)->swidth = x;
    ((TextureOpt*)opt)->twidth = x;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_fill(void* opt, float x)
{
    ((TextureOpt*)opt)->fill = x;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_time(void* opt, float x)
{
    // Not used by the texture system
    // ((TextureOpt*)opt)->time = x;
}

#if 0
OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE int
osl_texture_decode_interpmode(ustringhash_pod name_)
{
    ustringhash name_hash = ustringhash_from(name_);
    return tex_interp_to_code(name_hash);
}
#endif

#if 0
/* Not needed on GPU */
OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_interp(void* opt, ustringhash_pod modename_)
{
    using namespace OSL;
    ustringhash modename_hash = ustringhash_from(modename_);
    int mode                  = OSL::tex_interp_to_code(modename_hash);
    if (mode >= 0)
        ((TextureOpt*)opt)->interpmode = (TextureOpt::InterpMode)mode;
}
#endif

OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_interp_code(void* opt, int mode)
{
    ((TextureOpt*)opt)->interpmode = (TextureOpt::InterpMode)mode;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_subimage(void* opt, int subimage)
{
    ((TextureOpt*)opt)->subimage = subimage;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_subimagename(void* opt, ustringhash_pod subimagename_)
{
    ustringhash subimagename_hash = ustringhash_from(subimagename_);
#ifndef __CUDA_ARCH__
    // TODO: Enable when subimagename is ustringhash.
    ustring subimagename             = ustring_from(subimagename_hash);
    ((TextureOpt*)opt)->subimagename = subimagename;
#else
    // TODO: HACK to get this data through in some form, it won't be a valid
    // ustring but the GPU clients can at least convert it back to a hash.
    static_assert(sizeof(ustring) == sizeof(ustringhash), "Sizes must match");
    memcpy((void*)(&((TextureOpt*)opt)->subimagename), &subimagename_hash,
           sizeof(subimagename_hash));
#endif
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_missingcolor_arena(void* opt, const void* missing)
{
    ((TextureOpt*)opt)->missingcolor = (const float*)missing;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_texture_set_missingcolor_alpha(void* opt, int alphaindex,
                                   float missingalpha)
{
    float* m = (float*)((TextureOpt*)opt)->missingcolor;
    if (m)
        m[alphaindex] = missingalpha;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_init_trace_options(OSL::OpaqueExecContextPtr oec, void* opt)
{
    new (opt) TraceOpt;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_trace_set_mindist(void* opt, float x)
{
    ((TraceOpt*)opt)->mindist = x;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_trace_set_maxdist(void* opt, float x)
{
    ((TraceOpt*)opt)->maxdist = x;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_trace_set_shade(void* opt, int x)
{
    ((TraceOpt*)opt)->shade = x;
}


OSL_SHADEOP OSL_TEXTURE_SET_HOSTDEVICE void
osl_trace_set_traceset(void* opt, const ustringhash_pod x)
{
    ((TraceOpt*)opt)->traceset = ustringhash_from(x);
}


__forceinline__ __device__ float3
make_float3(const float4& a)
{
    return make_float3(a.x, a.y, a.z);
}



__device__ int
osl_range_check_err(int indexvalue, int length, OSL::ustringhash_pod symname,
                    void* sg, OSL::ustringhash_pod sourcefile, int sourceline,
                    OSL::ustringhash_pod groupname, int layer,
                    OSL::ustringhash_pod layername,
                    OSL::ustringhash_pod shadername)
{
    if (indexvalue < 0 || indexvalue >= length) {
        return indexvalue < 0 ? 0 : length - 1;
    }
    return indexvalue;
}


#define MAT(m) (*(OSL::Matrix44*)__builtin_assume_aligned(m, alignof(float)))

__device__ int
osl_get_matrix(void* sg_, void* r, OSL::ustringhash_pod from_)
{
    r                           = __builtin_assume_aligned(r, alignof(float));
    OSL::ustringhash from       = OSL::ustringhash_from(from_);
    OSL_CUDA::ShaderGlobals* sg = (OSL_CUDA::ShaderGlobals*)sg_;
    if (from == OSL::Hashes::common) {
        MAT(r).makeIdentity();
        return true;
    }
    if (from == OSL::Hashes::object) {
        MAT(r) = MAT(sg->object2common);
        return true;
    }
    if (from == OSL::Hashes::shader) {
        MAT(r) = MAT(sg->shader2common);
        return true;
    }

    // Find the index of the named transform in the transform list
    int match_idx = -1;
    for (size_t idx = 0; idx < OSL::pvt::num_named_xforms; ++idx) {
        if (from == HDSTR(((uint64_t*)OSL::pvt::xform_name_buffer)[idx])) {
            match_idx = static_cast<int>(idx);
            break;
        }
    }

    // Return the transform if there is a match
    if (match_idx >= 0) {
        MAT(r) = reinterpret_cast<OSL::Matrix44*>(
            OSL::pvt::xform_buffer)[match_idx];
        return true;
    }

    int ok = false;  // TODO: Implement transform
    if (!ok) {
        MAT(r).makeIdentity();
        // TBR: OSL would throw an error here, what should we do?
    }
    return ok;
}



__device__ int
osl_get_inverse_matrix(void* sg_, void* r, OSL::ustringhash_pod to_)
{
    r                           = __builtin_assume_aligned(r, alignof(float));
    OSL::ustringhash to         = OSL::ustringhash_from(to_);
    OSL_CUDA::ShaderGlobals* sg = (OSL_CUDA::ShaderGlobals*)sg_;
    if (to == OSL::Hashes::common) {
        MAT(r).makeIdentity();
        return true;
    }
    if (to == OSL::Hashes::object) {
        MAT(r) = MAT(sg->object2common);
        MAT(r).invert();
        return true;
    }
    if (to == OSL::Hashes::shader) {
        MAT(r) = MAT(sg->shader2common);
        MAT(r).invert();
        return true;
    }

    // Find the index of the named transform in the transform list
    int match_idx = -1;
    for (size_t idx = 0; idx < OSL::pvt::num_named_xforms; ++idx) {
        if (to == HDSTR(((uint64_t*)OSL::pvt::xform_name_buffer)[idx])) {
            match_idx = static_cast<int>(idx);
            break;
        }
    }
    // Return the transform if there is a match
    if (match_idx >= 0) {
        MAT(r) = reinterpret_cast<OSL::Matrix44*>(
            OSL::pvt::xform_buffer)[match_idx];
        MAT(r).invert();
        return true;
    }

    int ok = false;  // TODO: Implement transform
    if (!ok) {
        MAT(r).makeIdentity();
        // TBR: OSL would throw an error here, what should we do?
    }
    return ok;
}

#undef MAT
}
