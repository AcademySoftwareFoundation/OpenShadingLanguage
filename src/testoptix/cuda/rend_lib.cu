#include <optix.h>
#include <optix_device.h>
#include <optix_math.h>

#include "rend_lib.h"


rtDeclareVariable (uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable (uint2, launch_dim,   rtLaunchDim, );


// These functions are declared extern to prevent name mangling.
extern "C" {

    __device__
    void* closure_component_allot (void* pool, int id, size_t prim_size, const float3& w)
    {
        ((ClosureComponent*) pool)->id = id;
        ((ClosureComponent*) pool)->w  = w;

        size_t needed   = (sizeof(ClosureComponent) - sizeof(void*) + prim_size + 0x7) & ~0x7;
        char*  char_ptr = (char*) pool;

        return (void*) &char_ptr[needed];
    }


    __device__
    void* closure_mul_allot (void* pool, const float3& w, ClosureColor* c)
    {
        ((ClosureMul*) pool)->id      = ClosureColor::MUL;
        ((ClosureMul*) pool)->weight  = w;
        ((ClosureMul*) pool)->closure = c;

        size_t needed   = (sizeof(ClosureMul) + 0x7) & ~0x7;
        char*  char_ptr = (char*) pool;

        return &char_ptr[needed];
    }


    __device__
    void* closure_mul_float_allot (void* pool, const float& w, ClosureColor* c)
    {
        ((ClosureMul*) pool)->id       = ClosureColor::MUL;
        ((ClosureMul*) pool)->weight.x = w;
        ((ClosureMul*) pool)->weight.y = w;
        ((ClosureMul*) pool)->weight.z = w;
        ((ClosureMul*) pool)->closure  = c;

        size_t needed   = (sizeof(ClosureMul) + 0x7) & ~0x7;
        char*  char_ptr = (char*) pool;

        return &char_ptr[needed];
    }


    __device__
    void* closure_add_allot (void* pool, ClosureColor* a, ClosureColor* b)
    {
        ((ClosureAdd*) pool)->id       = ClosureColor::ADD;
        ((ClosureAdd*) pool)->closureA = a;
        ((ClosureAdd*) pool)->closureB = b;

        size_t needed   = (sizeof(ClosureAdd) + 0x7) & ~0x7;
        char*  char_ptr = (char*) pool;

        return &char_ptr[needed];
    }


    __device__
    void* osl_allocate_closure_component (void* sg_, int id, int size)
    {
        ShaderGlobals* sg_ptr = (ShaderGlobals*) sg_;

        float3 w   = make_float3 (1.0f);
        void*  ret = sg_ptr->renderstate;

        size = max (4, size);

        sg_ptr->renderstate = closure_component_allot (sg_ptr->renderstate, id, size, w);

        return ret;
    }


    __device__
    void* osl_allocate_weighted_closure_component (void* sg_, int id, int size, const float3* w)
    {
        ShaderGlobals* sg_ptr = (ShaderGlobals*) sg_;

        if  (w->x == 0.0f && w->y == 0.0f && w->z == 0.0f) {
            return NULL;
        }

        size = max (4, size);

        void* ret = sg_ptr->renderstate;
        sg_ptr->renderstate = closure_component_allot (sg_ptr->renderstate, id, size, *w);

        return ret;
    }


    __device__
    void* osl_mul_closure_color (void* sg_, ClosureColor* a, float3* w)
    {
        ShaderGlobals* sg_ptr = (ShaderGlobals*) sg_;

        if  (a == NULL) {
            return NULL;
        }

        if  (w->x == 0.0f && w->y == 0.0f && w->z == 0.0f) {
            return NULL;
        }

        if  (w->x == 1.0f && w->y == 1.0f && w->z == 1.0f) {
            return a;
        }

        void* ret = sg_ptr->renderstate;
        sg_ptr->renderstate = closure_mul_allot (sg_ptr->renderstate, *w, a);

        return ret;
    }


    __device__
    void* osl_mul_closure_float (void* sg_, ClosureColor* a, float w)
    {
        ShaderGlobals* sg_ptr = (ShaderGlobals*) sg_;

        if  (a == NULL || w == 0.0f) {
            return NULL;
        }

        if  (w == 1.0f) {
            return a;
        }

        void* ret = sg_ptr->renderstate;
        sg_ptr->renderstate = closure_mul_float_allot (sg_ptr->renderstate, w, a);

        return ret;
    }


    __device__
    void* osl_add_closure_closure (void* sg_, ClosureColor* a, ClosureColor* b)
    {
        ShaderGlobals* sg_ptr = (ShaderGlobals*) sg_;

        if  (a == NULL) {
            return b;
        }

        if  (b == NULL) {
            return a;
        }

        void* ret = sg_ptr->renderstate;
        sg_ptr->renderstate = closure_add_allot (sg_ptr->renderstate, a, b);

        return ret;
    }


    __device__
    int rend_get_userdata (char* name, void* data, int data_size)
    {
        return 0;
    }


    __device__
    int osl_bind_interpolated_param (void *sg_, const void *name, long long type,
                                     int userdata_has_derivs, void *userdata_data,
                                     int symbol_has_derivs, void *symbol_data,
                                     int symbol_data_size,
                                     char *userdata_initialized, int userdata_index)
    {
        int layer = 0;
        return rend_get_userdata ((char*)name, symbol_data, symbol_data_size);
    }
}
