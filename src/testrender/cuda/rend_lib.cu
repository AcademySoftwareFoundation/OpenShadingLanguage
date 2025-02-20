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

    size_t needed = (sizeof(OSL::ClosureComponent) + prim_size
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
    OSL_CUDA::ShaderGlobals* sg_ptr = (OSL_CUDA::ShaderGlobals*)sg_;

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
    OSL_CUDA::ShaderGlobals* sg_ptr = (OSL_CUDA::ShaderGlobals*)sg_;

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
    OSL_CUDA::ShaderGlobals* sg_ptr = (OSL_CUDA::ShaderGlobals*)sg_;
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

    OSL_CUDA::ShaderGlobals* sg_ptr = (OSL_CUDA::ShaderGlobals*)sg_;

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

    OSL_CUDA::ShaderGlobals* sg_ptr = (OSL_CUDA::ShaderGlobals*)sg_;

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
