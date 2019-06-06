#include <optix.h>
#include <optix_device.h>
#include <optix_math.h>
#include <OSL/oslclosure.h>

#include "rend_lib.h"


rtDeclareVariable (uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable (uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable (char*, test_str_1, , );
rtDeclareVariable (char*, test_str_2, , );

OSL_NAMESPACE_ENTER
namespace pvt {
    rtBuffer<char,1> s_color_system;
}
OSL_NAMESPACE_EXIT

// These functions are declared extern to prevent name mangling.
extern "C" {

    __device__
    void* closure_component_allot (void* pool, int id, size_t prim_size, const OSL::Color3& w)
    {
        ((OSL::ClosureComponent*) pool)->id = id;
        ((OSL::ClosureComponent*) pool)->w  = w;

        size_t needed   = (sizeof(OSL::ClosureComponent) - sizeof(void*) + prim_size + 0x7) & ~0x7;
        char*  char_ptr = (char*) pool;

        return (void*) &char_ptr[needed];
    }


    __device__
    void* closure_mul_allot (void* pool, const OSL::Color3& w, OSL::ClosureColor* c)
    {
        ((OSL::ClosureMul*) pool)->id      = OSL::ClosureColor::MUL;
        ((OSL::ClosureMul*) pool)->weight  = w;
        ((OSL::ClosureMul*) pool)->closure = c;

        size_t needed   = (sizeof(OSL::ClosureMul) + 0x7) & ~0x7;
        char*  char_ptr = (char*) pool;

        return &char_ptr[needed];
    }


    __device__
    void* closure_mul_float_allot (void* pool, const float& w, OSL::ClosureColor* c)
    {
        ((OSL::ClosureMul*) pool)->id       = OSL::ClosureColor::MUL;
        ((OSL::ClosureMul*) pool)->weight.x = w;
        ((OSL::ClosureMul*) pool)->weight.y = w;
        ((OSL::ClosureMul*) pool)->weight.z = w;
        ((OSL::ClosureMul*) pool)->closure  = c;

        size_t needed   = (sizeof(OSL::ClosureMul) + 0x7) & ~0x7;
        char*  char_ptr = (char*) pool;

        return &char_ptr[needed];
    }


    __device__
    void* closure_add_allot (void* pool, OSL::ClosureColor* a, OSL::ClosureColor* b)
    {
        ((OSL::ClosureAdd*) pool)->id       = OSL::ClosureColor::ADD;
        ((OSL::ClosureAdd*) pool)->closureA = a;
        ((OSL::ClosureAdd*) pool)->closureB = b;

        size_t needed   = (sizeof(OSL::ClosureAdd) + 0x7) & ~0x7;
        char*  char_ptr = (char*) pool;

        return &char_ptr[needed];
    }


    __device__
    void* osl_allocate_closure_component (void* sg_, int id, int size)
    {
        ShaderGlobals* sg_ptr = (ShaderGlobals*) sg_;

        OSL::Color3 w   = OSL::Color3 (1, 1, 1);
        void*  ret = sg_ptr->renderstate;

        size = max (4, size);

        sg_ptr->renderstate = closure_component_allot (sg_ptr->renderstate, id, size, w);

        return ret;
    }


    __device__
    void* osl_allocate_weighted_closure_component (void* sg_, int id, int size, const OSL::Color3* w)
    {
        ShaderGlobals* sg_ptr = (ShaderGlobals*) sg_;

        if (w->x == 0.0f && w->y == 0.0f && w->z == 0.0f) {
            return NULL;
        }

        size = max (4, size);

        void* ret = sg_ptr->renderstate;
        sg_ptr->renderstate = closure_component_allot (sg_ptr->renderstate, id, size, *w);

        return ret;
    }


    __device__
    void* osl_mul_closure_color (void* sg_, OSL::ClosureColor* a, const OSL::Color3* w)
    {
        ShaderGlobals* sg_ptr = (ShaderGlobals*) sg_;

        if (a == NULL) {
            return NULL;
        }

        if (w->x == 0.0f && w->y == 0.0f && w->z == 0.0f) {
            return NULL;
        }

        if (w->x == 1.0f && w->y == 1.0f && w->z == 1.0f) {
            return a;
        }

        void* ret = sg_ptr->renderstate;
        sg_ptr->renderstate = closure_mul_allot (sg_ptr->renderstate, *w, a);

        return ret;
    }


    __device__
    void* osl_mul_closure_float (void* sg_, OSL::ClosureColor* a, float w)
    {
        ShaderGlobals* sg_ptr = (ShaderGlobals*) sg_;

        if (a == NULL || w == 0.0f) {
            return NULL;
        }

        if (w == 1.0f) {
            return a;
        }

        void* ret = sg_ptr->renderstate;
        sg_ptr->renderstate = closure_mul_float_allot (sg_ptr->renderstate, w, a);

        return ret;
    }


    __device__
    void* osl_add_closure_closure (void* sg_, OSL::ClosureColor* a, OSL::ClosureColor* b)
    {
        ShaderGlobals* sg_ptr = (ShaderGlobals*) sg_;

        if (a == NULL) {
            return b;
        }

        if (b == NULL) {
            return a;
        }

        void* ret = sg_ptr->renderstate;
        sg_ptr->renderstate = closure_add_allot (sg_ptr->renderstate, a, b);

        return ret;
    }

#define IS_STRING(type) (type.basetype == OSL::TypeDesc::STRING)
#define IS_PTR(type)    (type.basetype == OSL::TypeDesc::PTR)

    __device__
    int rend_get_userdata (OSL::StringParam name, void* data, int data_size,
                           const OSL::TypeDesc& type, int index)
    {
        // Perform a userdata lookup using the parameter name, type, and
        // userdata index. If there is a match, memcpy the value into data and
        // return 1.

        if (IS_PTR(type) && name == StringParams::colorsystem) {
            *(void**)data = &OSL::pvt::s_color_system[0];
            return 1;
        }

        // TODO: This is temporary code for initial testing and demonstration.
        if (IS_STRING(type) && name == HDSTR(test_str_1)) {
            memcpy (data, &test_str_2, 8);
            return 1;
        }

        return 0;
    }

#undef IS_STRING
#undef IS_PTR

    __device__
    int osl_bind_interpolated_param (void *sg_, const void *name, long long type,
                                     int userdata_has_derivs, void *userdata_data,
                                     int symbol_has_derivs, void *symbol_data,
                                     int symbol_data_size,
                                     char *userdata_initialized, int userdata_index)
    {
        int status = rend_get_userdata (HDSTR(name), userdata_data, symbol_data_size,
                                        (*(OSL::TypeDesc*)&type), userdata_index);
        return status;
    }


    __device__
    int osl_strlen_is (const char *str)
    {
        return HDSTR(str).length();
    }


    __device__
    int osl_hash_is (const char *str)
    {
        return HDSTR(str).hash();
    }


    __device__
    int osl_getchar_isi (const char *str, int index)
    {
        return (str && unsigned(index) < HDSTR(str).length())
            ? str[index] : 0;
    }


    __device__
    void osl_printf (void* sg_, char* fmt_str, void* args)
    {
        // This can be used to limit printing to one Cuda thread for debugging
        // if (launch_index.x == 0 && launch_index.y == 0)
        //
        vprintf(fmt_str, (const char*) args);
    }


    __device__
    void* osl_get_noise_options (void *sg_)
    {
        ShaderGlobals* sg = ((ShaderGlobals*)sg_);
        NoiseOptCUDA* opt = (NoiseOptCUDA*)((ShadingContextCUDA*)sg->context)->noise_options_ptr();
        new (opt) NoiseOptCUDA;
        return opt;
    }


    __device__
    void* osl_get_texture_options (void *sg_)
    {
        return 0;
    }

    __device__
    void osl_texture_set_interp_code(void *opt, int mode)
    {
        // ((TextureOpt *)opt)->interpmode = (TextureOpt::InterpMode)mode;
    }

    __device__
    void osl_texture_set_stwrap_code (void *opt, int mode)
    {
        //((TextureOpt *)opt)->swrap = (TextureOpt::Wrap)mode;
        //((TextureOpt *)opt)->twrap = (TextureOpt::Wrap)mode;
    }

    __device__
    int osl_texture (void *sg_, const char *name, void *handle,
             void *opt_, float s, float t,
             float dsdx, float dtdx, float dsdy, float dtdy,
             int chans, void *result, void *dresultdx, void *dresultdy,
             void *alpha, void *dalphadx, void *dalphady,
             void *ustring_errormessage)
    {
        if (!handle)
            return 0;
        int64_t texID = int64_t(handle);
        *((float3*)result) = make_float3(optix::rtTex2D<float4>(texID, s, t));
        return 1;
    }

    __device__
    int osl_range_check_err (int indexvalue, int length,
                         const char *symname, void *sg,
                         const void *sourcefile, int sourceline,
                         const char *groupname, int layer,
                         const char *layername, const char *shadername) {
        if (indexvalue < 0 || indexvalue >= length) {
            return indexvalue < 0 ? 0 : length-1;
        }
        return indexvalue;
    }

    __device__
    int osl_range_check (int indexvalue, int length, const char *symname,
                         void *sg, const void *sourcefile, int sourceline,
                         const char *groupname, int layer, const char *layername,
                         const char *shadername)
    {
        if (indexvalue < 0 || indexvalue >= length) {
            indexvalue = osl_range_check_err (indexvalue, length, symname, sg,
                                              sourcefile, sourceline, groupname,
                                              layer, layername, shadername);
        }
        return indexvalue;
    }
}
