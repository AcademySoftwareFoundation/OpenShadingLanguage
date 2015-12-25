// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#define OPTIX_COMPATIBILITY 7
#include <OSL/oslclosure.h>
#include <cuda_runtime.h>

#include "rend_lib.h"

OSL_NAMESPACE_ENTER
namespace pvt {
extern __device__ char* s_color_system;
}
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
                                        const OSL::Color3* w)
{
    ShaderGlobals* sg_ptr = (ShaderGlobals*)sg_;

    if (w->x == 0.0f && w->y == 0.0f && w->z == 0.0f) {
        return NULL;
    }

    size = max(4, size);

    // Fix up the alignment
    void* ret = ((char*)sg_ptr->renderstate)
                + alignment_offset_calc(sg_ptr->renderstate,
                                        alignof(OSL::ClosureComponent));
    sg_ptr->renderstate = closure_component_allot(ret, id, size, *w);

    return ret;
}


__device__ void*
osl_mul_closure_color(void* sg_, OSL::ClosureColor* a, const OSL::Color3* w)
{
    ShaderGlobals* sg_ptr = (ShaderGlobals*)sg_;

    if (a == NULL) {
        return NULL;
    }

    if (w->x == 0.0f && w->y == 0.0f && w->z == 0.0f) {
        return NULL;
    }

    if (w->x == 1.0f && w->y == 1.0f && w->z == 1.0f) {
        return a;
    }

    // Fix up the alignment
    void* ret = ((char*)sg_ptr->renderstate)
                + alignment_offset_calc(sg_ptr->renderstate,
                                        alignof(OSL::ClosureComponent));
    sg_ptr->renderstate = closure_mul_allot(ret, *w, a);

    return ret;
}


__device__ void*
osl_mul_closure_float(void* sg_, OSL::ClosureColor* a, float w)
{
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
    sg_ptr->renderstate = closure_mul_float_allot(ret, w, a);

    return ret;
}


__device__ void*
osl_add_closure_closure(void* sg_, OSL::ClosureColor* a, OSL::ClosureColor* b)
{
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
    sg_ptr->renderstate = closure_add_allot(ret, a, b);

    return ret;
}

#define IS_STRING(type) (type.basetype == OSL::TypeDesc::STRING)
#define IS_PTR(type)    (type.basetype == OSL::TypeDesc::PTR)
#define IS_COLOR(type)  (type.vecsemantics == OSL::TypeDesc::COLOR)

__device__ bool
rend_get_userdata(OSL::StringParam name, void* data, int data_size,
                  const OSL::TypeDesc& type, int index)
{
    return false;
}

#undef IS_COLOR
#undef IS_STRING
#undef IS_PTR

__device__ int
osl_bind_interpolated_param(void* sg_, const void* name, long long type,
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
        memcpy(symbol_data, userdata_data, symbol_data_size);
        return 1;
    }
    return 0;
}


__device__ int
osl_strlen_is(const char* str)
{
    return HDSTR(str).length();
}


__device__ int
osl_hash_is(const char* str)
{
    return HDSTR(str).hash();
}


__device__ int
osl_getchar_isi(const char* str, int index)
{
    return (str && unsigned(index) < HDSTR(str).length()) ? str[index] : 0;
}


__device__ void
osl_printf(void* sg_, char* fmt_str, void* args)
{
    // This can be used to limit printing to one Cuda thread for debugging
    // if (launch_index.x == 0 && launch_index.y == 0)
    //
    // vprintf(fmt_str, (const char*)args);
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
    cudaTextureObject_t texID = cudaTextureObject_t(handle);
    float4 fromTexture        = osl_tex2DLookup(handle, s, t);
    // see note above
    // float4 fromTexture = tex2D<float4>(texID, s, t);
    *((float3*)result) = make_float3(fromTexture.x, fromTexture.y,
                                     fromTexture.z);
    return 1;
}

__device__ int
osl_range_check_err(int indexvalue, int length, const char* symname, void* sg,
                    const void* sourcefile, int sourceline,
                    const char* groupname, int layer, const char* layername,
                    const char* shadername)
{
    if (indexvalue < 0 || indexvalue >= length) {
        return indexvalue < 0 ? 0 : length - 1;
    }
    return indexvalue;
}

__device__ int
osl_range_check(int indexvalue, int length, const char* symname, void* sg,
                const void* sourcefile, int sourceline, const char* groupname,
                int layer, const char* layername, const char* shadername)
{
    if (indexvalue < 0 || indexvalue >= length) {
        indexvalue = osl_range_check_err(indexvalue, length, symname, sg,
                                         sourcefile, sourceline, groupname,
                                         layer, layername, shadername);
    }
    return indexvalue;
}

#define MAT(m) (*(OSL::Matrix44*)m)
__device__ int
osl_get_matrix(void* sg_, void* r, const char* from)
{
    ShaderGlobals* sg = (ShaderGlobals*)sg_;
    //ShadingContext *ctx = (ShadingContext *)sg->context;
    if (HDSTR(from) == StringParams::common ||
        //HDSTR(from) == ctx->shadingsys().commonspace_synonym() ||
        HDSTR(from) == StringParams::shader) {
        MAT(r).makeIdentity();
        return true;
    }
    if (HDSTR(from) == StringParams::object) {
        // TODO: Implement transform
        return false;
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
    ShaderGlobals* sg = (ShaderGlobals*)sg_;
    if (HDSTR(to) == StringParams::common ||
        //HDSTR(to) == ctx->shadingsys().commonspace_synonym() ||
        HDSTR(to) == StringParams::shader) {
        MAT(r).makeIdentity();
        return true;
    }
    if (HDSTR(to) == StringParams::object) {
        // TODO: Implement transform
        return false;
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
