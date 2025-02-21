// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#define OPTIX_COMPATIBILITY 7
#include <OSL/oslclosure.h>
#include <cuda_runtime.h>

#include "rend_lib.h"

OSL_NAMESPACE_BEGIN
namespace pvt {
extern __device__ char* s_color_system;
}
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


__device__ void*
osl_add_closure_closure(void* sg_, const void* a_, const void* b_)
{
    a_                         = __builtin_assume_aligned(a_, alignof(float));
    b_                         = __builtin_assume_aligned(b_, alignof(float));
    ShaderGlobals* sg          = (ShaderGlobals*)sg_;
    const OSL::ClosureColor* a = (const OSL::ClosureColor*)a_;
    const OSL::ClosureColor* b = (const OSL::ClosureColor*)b_;
    if (a == NULL)
        return b;
    if (b == NULL)
        return a;
    auto* closure_pool = ((RenderState*)sg->renderstate)->closure_pool;
    OSL::ClosureAdd* add
        = (OSL::ClosureAdd*)closure_pool->allocate(sizeof(OSL::ClosureAdd),
                                                   alignof(OSL::ClosureAdd));
    if (add) {
        add->id       = OSL::ClosureColor::ADD;
        add->closureA = a;
        add->closureB = b;
    }
    return add;
}

__device__ void*
osl_mul_closure_color(void* sg_, const void* a_, const void* w_)
{
    a_ = __builtin_assume_aligned(a_, alignof(float));
    w_ = __builtin_assume_aligned(w_, alignof(float));

    ShaderGlobals* sg          = (ShaderGlobals*)sg_;
    const OSL::ClosureColor* a = (const OSL::ClosureColor*)a_;
    const OSL::Color3* w       = (const OSL::Color3*)w_;
    if (a == NULL)
        return NULL;
    if (w->x == 0.0f && w->y == 0.0f && w->z == 0.0f)
        return NULL;
    if (w->x == 1.0f && w->y == 1.0f && w->z == 1.0f)
        return a;
    auto* closure_pool = ((RenderState*)sg->renderstate)->closure_pool;
    OSL::ClosureMul* mul
        = (OSL::ClosureMul*)closure_pool->allocate(sizeof(OSL::ClosureMul),
                                                   alignof(OSL::ClosureMul));
    if (mul) {
        mul->id      = OSL::ClosureColor::MUL;
        mul->weight  = *w;
        mul->closure = a;
    }
    return mul;
}

__device__ void*
osl_mul_closure_float(void* sg_, const void* a_, float w)
{
    a_ = __builtin_assume_aligned(a_, alignof(float));

    ShaderGlobals* sg          = (ShaderGlobals*)sg_;
    const OSL::ClosureColor* a = (const OSL::ClosureColor*)a_;
    if (a == NULL)
        return NULL;
    if (w == 0.0f)
        return NULL;
    if (w == 1.0f)
        return a;
    auto* closure_pool = ((RenderState*)sg->renderstate)->closure_pool;
    OSL::ClosureMul* mul
        = (OSL::ClosureMul*)closure_pool->allocate(sizeof(OSL::ClosureMul),
                                                   alignof(OSL::ClosureMul));
    if (mul) {
        mul->id      = OSL::ClosureColor::MUL;
        mul->weight  = OSL::Color3(w);
        mul->closure = a;
    }
    return mul;
}

__device__ void*
osl_allocate_closure_component(void* sg_, int id, int size)
{
    ShaderGlobals* sg  = (ShaderGlobals*)sg_;
    auto* closure_pool = ((RenderState*)sg->renderstate)->closure_pool;
    // Allocate the component and the mul back to back
    const size_t needed = sizeof(OSL::ClosureComponent) + size;
    OSL::ClosureComponent* comp
        = (OSL::ClosureComponent*)
              closure_pool->allocate(needed, alignof(OSL::ClosureComponent));
    if (comp) {
        comp->id = id;
        comp->w  = OSL::Color3(1.0f);
    }
    return comp;
}

__device__ void*
osl_allocate_weighted_closure_component(void* sg_, int id, int size,
                                        const void* w_)
{
    w_ = __builtin_assume_aligned(w_, alignof(float));

    ShaderGlobals* sg    = (ShaderGlobals*)sg_;
    const OSL::Color3* w = (const OSL::Color3*)w_;
    if (w->x == 0.0f && w->y == 0.0f && w->z == 0.0f)
        return NULL;
    auto* closure_pool = ((RenderState*)sg->renderstate)->closure_pool;
    // Allocate the component and the mul back to back
    const size_t needed = sizeof(OSL::ClosureComponent) + size;
    OSL::ClosureComponent* comp
        = (OSL::ClosureComponent*)
              closure_pool->allocate(needed, alignof(OSL::ClosureComponent));
    if (comp) {
        comp->id = id;
        comp->w  = *w;
    }
    return comp;
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
osl_bind_interpolated_param(void* sg_, OSL::ustring_pod name, long long type,
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
osl_range_check_err(int indexvalue, int length, OSL::ustring_pod symname,
                    void* sg, OSL::ustring_pod sourcefile, int sourceline,
                    OSL::ustring_pod groupname, int layer,
                    OSL::ustring_pod layername, OSL::ustring_pod shadername)
{
    if (indexvalue < 0 || indexvalue >= length) {
        return indexvalue < 0 ? 0 : length - 1;
    }
    return indexvalue;
}

__device__ int
osl_range_check(int indexvalue, int length, OSL::ustring_pod symname, void* sg,
                OSL::ustring_pod sourcefile, int sourceline,
                OSL::ustring_pod groupname, int layer,
                OSL::ustring_pod layername, OSL::ustring_pod shadername)
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
