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

OSL_NAMESPACE_ENTER
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
OSL_NAMESPACE_EXIT


// Taken from the SimplePool class
__device__ static inline size_t
alignment_offset_calc(void* ptr, size_t alignment)
{
    uintptr_t ptrbits = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t offset  = ((ptrbits + alignment - 1) & -alignment) - ptrbits;
    return offset;
}


// These functions are declared extern to prevent name mangling.
extern "C" {

// add OptiX entry point to prevent OptiX from discarding the module
__global__ void
__direct_callable__dummy_rend_lib()
{
}


__device__ void*
closure_component_allot(void* pool, int id, size_t prim_size,
                        const OSL::Color3& w)
{
    ((OSL::ClosureComponent*)pool)->id = id;
    ((OSL::ClosureComponent*)pool)->w  = w;

    size_t needed = (sizeof(OSL::ClosureComponent) - sizeof(void*) + prim_size
                     + (alignof(OSL::ClosureComponent) - 1))
                    & ~(alignof(OSL::ClosureComponent) - 1);
    char* char_ptr = (char*)pool;

    return (void*)&char_ptr[needed];
}


__device__ void*
closure_mul_allot(void* pool, const OSL::Color3& w, OSL::ClosureColor* c)
{
    ((OSL::ClosureMul*)pool)->id      = OSL::ClosureColor::MUL;
    ((OSL::ClosureMul*)pool)->weight  = w;
    ((OSL::ClosureMul*)pool)->closure = c;

    size_t needed = (sizeof(OSL::ClosureMul)
                     + (alignof(OSL::ClosureComponent) - 1))
                    & ~(alignof(OSL::ClosureComponent) - 1);
    char* char_ptr = (char*)pool;

    return &char_ptr[needed];
}


__device__ void*
closure_mul_float_allot(void* pool, const float& w, OSL::ClosureColor* c)
{
    ((OSL::ClosureMul*)pool)->id       = OSL::ClosureColor::MUL;
    ((OSL::ClosureMul*)pool)->weight.x = w;
    ((OSL::ClosureMul*)pool)->weight.y = w;
    ((OSL::ClosureMul*)pool)->weight.z = w;
    ((OSL::ClosureMul*)pool)->closure  = c;

    size_t needed = (sizeof(OSL::ClosureMul)
                     + (alignof(OSL::ClosureComponent) - 1))
                    & ~(alignof(OSL::ClosureComponent) - 1);
    char* char_ptr = (char*)pool;

    return &char_ptr[needed];
}


__device__ void*
closure_add_allot(void* pool, OSL::ClosureColor* a, OSL::ClosureColor* b)
{
    ((OSL::ClosureAdd*)pool)->id       = OSL::ClosureColor::ADD;
    ((OSL::ClosureAdd*)pool)->closureA = a;
    ((OSL::ClosureAdd*)pool)->closureB = b;

    size_t needed = (sizeof(OSL::ClosureAdd)
                     + (alignof(OSL::ClosureComponent) - 1))
                    & ~(alignof(OSL::ClosureComponent) - 1);
    char* char_ptr = (char*)pool;

    return &char_ptr[needed];
}


__device__ void*
osl_allocate_closure_component(void* sg_, int id, int size)
{
    ShaderGlobals* sg_ptr = (ShaderGlobals*)sg_;

    OSL::Color3 w = OSL::Color3(1, 1, 1);
    // Fix up the alignment
    void* ret = ((char*)sg_ptr->renderstate)
                + alignment_offset_calc(sg_ptr->renderstate,
                                        alignof(OSL::ClosureComponent));

    size = max(4, size);

    sg_ptr->renderstate = closure_component_allot(ret, id, size, w);

    return ret;
}



__device__ void*
osl_allocate_weighted_closure_component(void* sg_, int id, int size,
                                        const void* w)
{
    ShaderGlobals* sg_ptr = (ShaderGlobals*)sg_;

    const OSL::Color3* wc
        = (const OSL::Color3*)__builtin_assume_aligned(w, alignof(float));

    if (wc->x == 0.0f && wc->y == 0.0f && wc->z == 0.0f) {
        return NULL;
    }

    size = max(4, size);

    // Fix up the alignment
    void* ret = ((char*)sg_ptr->renderstate)
                + alignment_offset_calc(sg_ptr->renderstate,
                                        alignof(OSL::ClosureComponent));
    sg_ptr->renderstate = closure_component_allot(ret, id, size, *wc);

    return ret;
}



__device__ void*
osl_mul_closure_color(void* sg_, void* a, const void* w)
{
    ShaderGlobals* sg_ptr = (ShaderGlobals*)sg_;
    const OSL::Color3* wc
        = (const OSL::Color3*)__builtin_assume_aligned(w, alignof(float));

    if (a == NULL) {
        return NULL;
    }

    if (wc->x == 0.0f && wc->y == 0.0f && wc->z == 0.0f) {
        return NULL;
    }

    if (wc->x == 1.0f && wc->y == 1.0f && wc->z == 1.0f) {
        return a;
    }

    // Fix up the alignment
    void* ret = ((char*)sg_ptr->renderstate)
                + alignment_offset_calc(sg_ptr->renderstate,
                                        alignof(OSL::ClosureComponent));
    sg_ptr->renderstate = closure_mul_allot(ret, *wc, (OSL::ClosureColor*)a);

    return ret;
}



__device__ void*
osl_mul_closure_float(void* sg_, void* a, float w)
{
    a = __builtin_assume_aligned(a, alignof(float));

    ShaderGlobals* sg_ptr = (ShaderGlobals*)sg_;

    if (a == NULL || w == 0.0f) {
        return NULL;
    }

    if (w == 1.0f) {
        return a;
    }

    // Fix up the alignment
    void* ret = ((char*)sg_ptr->renderstate)
                + alignment_offset_calc(sg_ptr->renderstate,
                                        alignof(OSL::ClosureComponent));
    sg_ptr->renderstate = closure_mul_float_allot(ret, w,
                                                  (OSL::ClosureColor*)a);

    return ret;
}



__device__ void*
osl_add_closure_closure(void* sg_, void* a, void* b)
{
    a = __builtin_assume_aligned(a, alignof(float));
    b = __builtin_assume_aligned(b, alignof(float));

    ShaderGlobals* sg_ptr = (ShaderGlobals*)sg_;

    if (a == NULL) {
        return b;
    }

    if (b == NULL) {
        return a;
    }

    // Fix up the alignment
    void* ret = ((char*)sg_ptr->renderstate)
                + alignment_offset_calc(sg_ptr->renderstate,
                                        alignof(OSL::ClosureComponent));
    sg_ptr->renderstate = closure_add_allot(ret, (OSL::ClosureColor*)a,
                                            (OSL::ClosureColor*)b);

    return ret;
}


#define IS_STRING(type) (type.basetype == OSL::TypeDesc::STRING)
#define IS_PTR(type)    (type.basetype == OSL::TypeDesc::PTR)
#define IS_COLOR(type)  (type.vecsemantics == OSL::TypeDesc::COLOR)


__device__ bool
rend_get_userdata(OSL::StringParam name, void* data, int data_size,
                  const OSL::TypeDesc& type, int index)
{
    // Perform a userdata lookup using the parameter name, type, and
    // userdata index. If there is a match, memcpy the value into data and
    // return 1.

    if (IS_PTR(type) && name.hash() == STRING_PARAMS(colorsystem)) {
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
osl_bind_interpolated_param(void* sg_, const char* name, long long type,
                            int userdata_has_derivs, void* userdata_data,
                            int symbol_has_derivs, void* symbol_data,
                            int symbol_data_size, char* userdata_initialized,
                            int userdata_index)
{
    char status = *userdata_initialized;
    if (status == 0) {
        bool ok               = rend_get_userdata(HDSTR(name), userdata_data,
                                                  symbol_data_size, (*(OSL::TypeDesc*)&type),
                                                  userdata_index);
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
osl_strlen_is(const char* str)
{
    //return HDSTR(str).length();
    return 0;
}


__device__ int
osl_hash_is(const char* str)
{
    return HDSTR(str);
}


__device__ int
osl_getchar_isi(const char* str, int index)
{
    //        return (str && unsigned(index) < HDSTR(str).length())
    //            ? str[index] : 0;
    return 0;
}



// Printing is handled by the host.  Copy format string's hash and
// all the arguments to our print buffer.
// Note:  the first element of 'args' is the size of the argument list
__device__ void
osl_printf(void* sg_, char* fmt_str, void* args)
{
    uint64_t fmt_str_hash = HDSTR(fmt_str).hash();
    uint64_t args_size    = reinterpret_cast<uint64_t*>(args)[0];

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



__device__ void*
osl_get_noise_options(void* sg_)
{
    ShaderGlobals* sg = ((ShaderGlobals*)sg_);
    NoiseOptCUDA* opt
        = (NoiseOptCUDA*)((ShadingContextCUDA*)sg->context)->noise_options_ptr();
    new (opt) NoiseOptCUDA;
    return opt;
}


__device__ void*
osl_get_texture_options(void* sg_)
{
    return 0;
}

__device__ void
osl_texture_set_interp_code(void* opt, int mode)
{
    // ((TextureOpt *)opt)->interpmode = (TextureOpt::InterpMode)mode;
}


__device__ void
osl_texture_set_stwrap_code(void* opt, int mode)
{
    //((TextureOpt *)opt)->swrap = (TextureOpt::Wrap)mode;
    //((TextureOpt *)opt)->twrap = (TextureOpt::Wrap)mode;
}


__forceinline__ __device__ float3
make_float3(const float4& a)
{
    return make_float3(a.x, a.y, a.z);
}



// FIXME:
// clang++ 9.0 seems to have trouble with tex2d<float4>() look-ups,
// so we'll declare this external and implement texture lookups in
// CUDA files compiled by nvcc (optix_grid_renderer.cu and
// optix_raytrace.cu).
// (clang++ 9.0 error 'undefined __nv_tex_surf_handler')
extern __device__ float4
osl_tex2DLookup(void* handle, float s, float t);

__device__ int
osl_texture(void* sg_, const char* name, void* handle, void* opt_, float s,
            float t, float dsdx, float dtdx, float dsdy, float dtdy, int chans,
            void* result, void* dresultdx, void* dresultdy, void* alpha,
            void* dalphadx, void* dalphady, void* ustring_errormessage)
{
    if (!handle)
        return 0;
    // cudaTextureObject_t texID = cudaTextureObject_t(handle);
    float4 fromTexture = osl_tex2DLookup(handle, s, t);
    // see note above
    // float4 fromTexture = tex2D<float4>(texID, s, t);
    *((float3*)result) = make_float3(fromTexture.x, fromTexture.y,
                                     fromTexture.z);
    return 1;
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
osl_get_matrix(void* sg_, void* r, const char* from)
{
    r                 = __builtin_assume_aligned(r, alignof(float));
    ShaderGlobals* sg = (ShaderGlobals*)sg_;
    if (HDSTR(from) == STRING_PARAMS(common)) {
        MAT(r).makeIdentity();
        return true;
    }
    if (HDSTR(from) == STRING_PARAMS(object)) {
        MAT(r) = MAT(sg->object2common);
        return true;
    }
    if (HDSTR(from) == STRING_PARAMS(shader)) {
        MAT(r) = MAT(sg->shader2common);
        return true;
    }

    // Find the index of the named transform in the transform list
    int match_idx = -1;
    for (size_t idx = 0; idx < OSL::pvt::num_named_xforms; ++idx) {
        if (HDSTR(from)
            == HDSTR(((uint64_t*)OSL::pvt::xform_name_buffer)[idx])) {
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
osl_get_inverse_matrix(void* sg_, void* r, const char* to)
{
    r                 = __builtin_assume_aligned(r, alignof(float));
    ShaderGlobals* sg = (ShaderGlobals*)sg_;
    if (HDSTR(to) == STRING_PARAMS(common)) {
        MAT(r).makeIdentity();
        return true;
    }
    if (HDSTR(to) == STRING_PARAMS(object)) {
        MAT(r) = MAT(sg->object2common);
        MAT(r).invert();
        return true;
    }
    if (HDSTR(to) == STRING_PARAMS(shader)) {
        MAT(r) = MAT(sg->shader2common);
        MAT(r).invert();
        return true;
    }

    // Find the index of the named transform in the transform list
    int match_idx = -1;
    for (size_t idx = 0; idx < OSL::pvt::num_named_xforms; ++idx) {
        if (HDSTR(to) == HDSTR(((uint64_t*)OSL::pvt::xform_name_buffer)[idx])) {
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
