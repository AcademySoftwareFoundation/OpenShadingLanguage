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

// These functions are declared extern to prevent name mangling.
extern "C" {

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
