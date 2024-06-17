// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <optix.h>
#include <optix_device.h>

#include <OSL/hashes.h>

#include "optix_raytracer.h"
#include "rend_lib.h"
#include "vec_math.h"

#include "../background.h"
#include "../raytracer.h"
#include "../render_params.h"
#include "../sampling.h"

// clang-format off
// These files must be included in this specific order
#include "../shading.h"
#include "../shading.cpp"
#include "../shading_cuda.cpp"
// clang-format on

#include <cstdint>


OSL_NAMESPACE_ENTER
namespace pvt {
__device__ CUdeviceptr s_color_system          = 0;
__device__ CUdeviceptr osl_printf_buffer_start = 0;
__device__ CUdeviceptr osl_printf_buffer_end   = 0;
__device__ uint64_t test_str_1                 = 0;
__device__ uint64_t test_str_2                 = 0;
__device__ uint64_t num_named_xforms           = 0;
__device__ CUdeviceptr xform_name_buffer       = 0;
__device__ CUdeviceptr xform_buffer            = 0;
}  // namespace pvt
OSL_NAMESPACE_EXIT


extern "C" {
__device__ __constant__ RenderParams render_params;
}


static __device__ void
globals_from_hit(ShaderGlobalsType& sg, float radius = 0.0f, float spread = 0.0f,
                 Ray::RayType raytype = Ray::RayType::CAMERA)
{
    ShaderGlobalsType local_sg;
    // hit-kind 0: quad hit
    //          1: sphere hit
    optixDirectCall<void, unsigned int, float, float3, float3, ShaderGlobalsType*>(
        optixGetHitKind(), optixGetPrimitiveIndex(), optixGetRayTmax(),
        optixGetWorldRayOrigin(), optixGetWorldRayDirection(), &local_sg);
    // Setup the ShaderGlobals
    const float3 ray_direction = optixGetWorldRayDirection();
    const float3 ray_origin    = optixGetWorldRayOrigin();
    const float t_hit          = optixGetRayTmax();

    // Construct a Ray in order to calculate P and its derivatives
    Ray ray(F3_TO_V3(ray_origin), F3_TO_V3(ray_direction), radius, spread,
            Ray::RayType::CAMERA);
    Dual2<float> t(t_hit);
    Dual2<Vec3> P = ray.point(t);

    sg.I  = ray_direction;
    sg.N  = normalize(optixTransformNormalFromObjectToWorldSpace(V3_TO_F3(local_sg.N)));
    sg.Ng = normalize(optixTransformNormalFromObjectToWorldSpace(V3_TO_F3(local_sg.Ng)));
    sg.P  = V3_TO_F3(P.val());
    sg.dPdx        = V3_TO_F3(P.dx());
    sg.dPdy        = V3_TO_F3(P.dy());
    sg.dPdu        = local_sg.dPdu;
    sg.dPdv        = local_sg.dPdv;
    sg.u           = local_sg.u;
    sg.v           = local_sg.v;
    sg.Ci          = nullptr;
    sg.surfacearea = local_sg.surfacearea;
    sg.backfacing  = dot(V3_TO_F3(sg.N), V3_TO_F3(sg.I)) > 0.0f;
    sg.shaderID    = local_sg.shaderID;

    if (sg.backfacing) {
        sg.N  = -sg.N;
        sg.Ng = -sg.Ng;
    }

    sg.raytype        = raytype;
    sg.flipHandedness = dot(V3_TO_F3(sg.N), cross(V3_TO_F3(sg.dPdx), V3_TO_F3(sg.dPdy))) < 0.0f;
}


static inline __device__ void
execute_shader(ShaderGlobalsType& sg, char* closure_pool)
{
    if (sg.shaderID < 0) {
        // TODO: should probably never get here ...
        return;
    }

    // Pack the "closure pool" into one of the ShaderGlobals pointers
    *(int*)&closure_pool[0] = 0;
    sg.renderstate          = &closure_pool[0];

    // Create some run-time options structs. The OSL shader fills in the structs
    // as it executes, based on the options specified in the shader source.
    NoiseOptCUDA noiseopt;
    TextureOptCUDA textureopt;
    TraceOptCUDA traceopt;

    // Pack the pointers to the options structs in a faux "context",
    // which is a rough stand-in for the host ShadingContext.
    ShadingContextCUDA shading_context = { &noiseopt, &textureopt, &traceopt };
    sg.context                         = &shading_context;

    // Run the OSL callable
    void* interactive_ptr = reinterpret_cast<void**>(
        render_params.interactive_params)[sg.shaderID];
    const unsigned int shaderIdx = 2u + sg.shaderID + 0u;
    optixDirectCall<void, ShaderGlobalsType*, void*, void*, void*, int, void*>(
        shaderIdx, &sg /*shaderglobals_ptr*/,
        nullptr /*groupdata_ptr*/,
        nullptr /*userdata_base_ptr*/,
        nullptr /*output_base_ptr*/,
        0 /*shadeindex - unused*/,
        interactive_ptr /*interactive_params_ptr*/
    );
}


//
// Closure evaluation functions
//


#if 0
static __device__ Color3
evaluate_layer_opacity(const ShaderGlobalsType& sg, const ClosureColor* closure)
{
    // Null closure, the layer is fully transparent
    if (closure == nullptr)
        return Color3(0);

    // The depth of the closure tree must not exceed the stack size.
    // A stack size of 8 is probably quite generous for relatively
    // balanced trees.
    const int STACK_SIZE = 8;

    // Non-recursive traversal stack
    int stack_idx = 0;
    const ClosureColor* ptr_stack[STACK_SIZE];
    Color3 weight_stack[STACK_SIZE];

    // Shading accumulator
    Color3 weight = Color3(1.0f);

    while (closure) {
        switch (closure->id) {
        case MUL: {
            weight *= ((ClosureMul*)closure)->weight;
            closure = ((ClosureMul*)closure)->closure;
            break;
        }
        case ADD: {
            ptr_stack[stack_idx]      = ((ClosureAdd*)closure)->closureB;
            weight_stack[stack_idx++] = weight;
            closure                   = ((ClosureAdd*)closure)->closureA;
            break;
        }
        default: {
            const ClosureComponent* comp = closure->as_comp();
            Color3 w                     = comp->w;
            switch (comp->id) {
            case MX_LAYER_ID: {
                const MxLayerParams* srcparams = comp->as<MxLayerParams>();
                closure                        = srcparams->top;
                ptr_stack[stack_idx]           = srcparams->base;
                weight_stack[stack_idx++]      = weight * w;
                break;
            }
            case REFLECTION_ID:
            case FRESNEL_REFLECTION_ID: {
                const ReflectionParams* params = comp->as<ReflectionParams>();
                Reflection bsdf(*params);
                weight *= w * bsdf.get_albedo(-F3_TO_V3(sg.I));
                closure = nullptr;
                break;
            }
            case MX_DIELECTRIC_ID: {
                const MxDielectricParams* params
                    = comp->as<MxDielectricParams>();
                // Transmissive dielectrics are opaque
                if (!is_black(params->transmission_tint)) {
                    closure = nullptr;
                    break;
                }
                MxDielectricOpaque bsdf(*params, 1.0f);
                weight *= w * bsdf.get_albedo(-F3_TO_V3(sg.I));
                closure = nullptr;
                break;
            }
            case MX_GENERALIZED_SCHLICK_ID: {
                const MxGeneralizedSchlickParams* params
                    = comp->as<MxGeneralizedSchlickParams>();
                // Transmissive dielectrics are opaque
                if (!is_black(params->transmission_tint)) {
                    closure = nullptr;
                    break;
                }
                MxGeneralizedSchlickOpaque bsdf(*params, 1.0f);
                weight *= w * bsdf.get_albedo(-F3_TO_V3(sg.I));
                closure = nullptr;
                break;
            }
            case MX_SHEEN_ID: {
                const MxSheenParams* params = comp->as<MxSheenParams>();
                MxSheen bsdf(*params);
                weight *= w * bsdf.get_albedo(-F3_TO_V3(sg.I));
                closure = nullptr;
                break;
            }
            default:  // Assume unhandled BSDFs are opaque
                closure = nullptr;
                break;
            }
        }
        }
        if (closure == nullptr && stack_idx > 0) {
            closure = ptr_stack[--stack_idx];
            weight  = weight_stack[stack_idx];
        }
    }
    return weight;
}


static __device__ Color3
process_medium_closure(const ShaderGlobalsType& sg, ShadingResult& result,
                       const ClosureColor* closure, const Color3& w)
{
    Color3 color_result = Color3(0.0f);
    if (!closure) {
        return color_result;
    }

    // The depth of the closure tree must not exceed the stack size.
    // A stack size of 8 is probably quite generous for relatively
    // balanced trees.
    const int STACK_SIZE = 8;

    // Non-recursive traversal stack
    int stack_idx = 0;
    const ClosureColor* ptr_stack[STACK_SIZE];
    Color3 weight_stack[STACK_SIZE];

    // Shading accumulator
    Color3 weight = w; // Color3(1.0f);
    while (closure) {
        ClosureIDs id = static_cast<ClosureIDs>(closure->id);
        switch (id) {
        case ADD: {
            ptr_stack[stack_idx]      = ((ClosureAdd*)closure)->closureB;
            weight_stack[stack_idx++] = weight;
            closure                   = ((ClosureAdd*)closure)->closureA;
            break;
        }
        case MUL: {
            weight *= ((ClosureMul*)closure)->weight;
            closure = ((ClosureMul*)closure)->closure;
            break;
        }
        case MX_LAYER_ID: {
            const ClosureComponent* comp = closure->as_comp();
            const MxLayerParams* params  = comp->as<MxLayerParams>();
            Color3 base_w
                = w
                  * (Color3(1)
                     - clamp(evaluate_layer_opacity(sg, params->top), 0.f, 1.f));
            closure                   = params->top;
            ptr_stack[stack_idx]      = params->base;
            weight_stack[stack_idx++] = weight * w;
            break;
        }
        case MX_ANISOTROPIC_VDF_ID: {
            const ClosureComponent* comp = closure->as_comp();
            Color3 cw                    = w * comp->w;
            const auto& params           = *comp->as<MxAnisotropicVdfParams>();
            result.sigma_t               = cw * params.extinction;
            result.sigma_s               = params.albedo * result.sigma_t;
            result.medium_g              = params.anisotropy;
            result.refraction_ior        = 1.0f;
            result.priority = 0;
            closure = nullptr;
            break;
        }
        case MX_MEDIUM_VDF_ID: {
            const ClosureComponent* comp = closure->as_comp();
            Color3 cw                    = w * comp->w;
            const auto& params           = *comp->as<MxMediumVdfParams>();
            result.sigma_t = { -OIIO::fast_log(params.transmission_color.x),
                               -OIIO::fast_log(params.transmission_color.y),
                               -OIIO::fast_log(params.transmission_color.z) };
            // NOTE: closure weight scales the extinction parameter
            result.sigma_t *= cw / params.transmission_depth;
            result.sigma_s  = params.albedo * result.sigma_t;
            result.medium_g = params.anisotropy;
            // TODO: properly track a medium stack here ...
            result.refraction_ior = sg.backfacing ? 1.0f / params.ior : params.ior;
            result.priority       = params.priority;
            closure = nullptr;
            break;
        }
        case MX_DIELECTRIC_ID: {
            const ClosureComponent* comp = closure->as_comp();
            const auto& params           = *comp->as<MxDielectricParams>();
            if (!is_black(w * comp->w * params.transmission_tint)) {
                // TODO: properly track a medium stack here ...
                result.refraction_ior = sg.backfacing ? 1.0f / params.ior
                                                      : params.ior;
            }
            closure = nullptr;
            break;
        }
        case MX_GENERALIZED_SCHLICK_ID: {
            const ClosureComponent* comp = closure->as_comp();
            const auto& params           = *comp->as<MxGeneralizedSchlickParams>();
            if (!is_black(w * comp->w * params.transmission_tint)) {
                // TODO: properly track a medium stack here ...
                float avg_F0  = clamp((params.f0.x + params.f0.y + params.f0.z)
                                          / 3.0f,
                                      0.0f, 0.99f);
                float sqrt_F0 = sqrtf(avg_F0);
                float ior     = (1 + sqrt_F0) / (1 - sqrt_F0);
                result.refraction_ior = sg.backfacing ? 1.0f / ior : ior;
            }
            closure = nullptr;
            break;
        }
        default:
            closure = nullptr;
            break;
        }
        if (closure == nullptr && stack_idx > 0) {
            closure = ptr_stack[--stack_idx];
            weight  = weight_stack[stack_idx];
        }
    }
    return weight;
}


static __device__ void
process_closure(const ShaderGlobalsType& sg, const ClosureColor* closure,
                ShadingResult& result, bool light_only)
{
    if (!closure) {
        return;
    }

    static const ustringhash uh_ggx("ggx");
    static const ustringhash uh_beckmann("beckmann");
    static const ustringhash uh_default("default");

    // The depth of the closure tree must not exceed the stack size.
    // A stack size of 8 is probably quite generous for relatively
    // balanced trees.
    const int STACK_SIZE = 8;

    // Non-recursive traversal stack
    int stack_idx = 0;
    const ClosureColor* ptr_stack[STACK_SIZE];
    Color3 weight_stack[STACK_SIZE];

    // Shading accumulator
    Color3 weight = Color3(1.0f);
    while (closure) {
        ClosureIDs id = static_cast<ClosureIDs>(closure->id);
        switch (id) {
        case ADD: {
            ptr_stack[stack_idx]      = ((ClosureAdd*)closure)->closureB;
            weight_stack[stack_idx++] = weight;
            closure                   = ((ClosureAdd*)closure)->closureA;
            break;
        }
        case MUL: {
            weight *= ((ClosureMul*)closure)->weight;
            closure = ((ClosureMul*)closure)->closure;
            break;
        }
        default: {
            bool ok                      = false;
            const ClosureComponent* comp = closure->as_comp();
            Color3 cw                    = weight * comp->w;
            switch (id) {
            case EMISSION_ID:
                result.Le += cw;
                closure = nullptr;
                ok      = true;
                break;
            case DIFFUSE_ID:
                ok = result.bsdf.add_bsdf<Diffuse<0>>(
                    cw, *comp->as<DiffuseParams>());
                closure = nullptr;
                break;
            case OREN_NAYAR_ID:
                ok = result.bsdf.add_bsdf<OrenNayar>(
                    cw, *comp->as<OrenNayarParams>());
                closure = nullptr;
                break;
            case TRANSLUCENT_ID:
                ok = result.bsdf.add_bsdf<Diffuse<1>>(
                    cw, *comp->as<DiffuseParams>());
                closure = nullptr;
                break;
            case PHONG_ID:
                ok = result.bsdf.add_bsdf<Phong>(cw, *comp->as<PhongParams>());
                closure = nullptr;
                break;
            case WARD_ID:
                ok = result.bsdf.add_bsdf<Ward>(cw, *comp->as<WardParams>());
                closure = nullptr;
                break;
            case MICROFACET_ID: {
                closure                    = nullptr;
                const MicrofacetParams* mp = comp->as<MicrofacetParams>();
                if (mp->dist == uh_ggx) {
                    switch (mp->refract) {
                    case 0:
                        ok = result.bsdf.add_bsdf<MicrofacetGGXRefl>(cw, *mp);
                        break;
                    case 1:
                        ok = result.bsdf.add_bsdf<MicrofacetGGXRefr>(cw, *mp);
                        break;
                    case 2:
                        ok = result.bsdf.add_bsdf<MicrofacetGGXBoth>(cw, *mp);
                        break;
                    }
                } else if (mp->dist == uh_beckmann || mp->dist == uh_default) {
                    switch (mp->refract) {
                    case 0:
                        ok = result.bsdf.add_bsdf<MicrofacetBeckmannRefl>(cw,
                                                                          *mp);
                        break;
                    case 1:
                        ok = result.bsdf.add_bsdf<MicrofacetBeckmannRefr>(cw,
                                                                          *mp);
                        break;
                    case 2:
                        ok = result.bsdf.add_bsdf<MicrofacetBeckmannBoth>(cw,
                                                                          *mp);
                        break;
                    }
                }
                break;
            }
            case REFLECTION_ID:
            case FRESNEL_REFLECTION_ID:
                ok = result.bsdf.add_bsdf<Reflection>(
                    cw, *comp->as<ReflectionParams>());
                closure = nullptr;
                break;
            case REFRACTION_ID:
                ok = result.bsdf.add_bsdf<Refraction>(
                    cw, *comp->as<RefractionParams>());
                closure = nullptr;
                break;
            case TRANSPARENT_ID:
                ok      = result.bsdf.add_bsdf<Transparent>(cw);
                closure = nullptr;
                break;
            case MX_OREN_NAYAR_DIFFUSE_ID: {
                // translate MaterialX parameters into existing closure
                const MxOrenNayarDiffuseParams* srcparams
                    = comp->as<MxOrenNayarDiffuseParams>();
                OrenNayarParams params = {};
                params.N               = srcparams->N;
                params.sigma           = srcparams->roughness;
                ok = result.bsdf.add_bsdf<OrenNayar>(cw * srcparams->albedo,
                                                     params);
                closure = nullptr;
                break;
            }
            case MX_BURLEY_DIFFUSE_ID: {
                const MxBurleyDiffuseParams& params
                    = *comp->as<MxBurleyDiffuseParams>();
                ok      = result.bsdf.add_bsdf<MxBurleyDiffuse>(cw, params);
                closure = nullptr;
                break;
            }
            case MX_DIELECTRIC_ID: {
                const MxDielectricParams& params
                    = *comp->as<MxDielectricParams>();
                if (is_black(params.transmission_tint))
                    ok = result.bsdf.add_bsdf<MxDielectricOpaque>(cw, params,
                                                                  1.0f);
                else
                    ok = result.bsdf.add_bsdf<MxDielectric>(
                        cw, params, result.refraction_ior);
                closure = nullptr;
                break;
            }
            case MX_CONDUCTOR_ID: {
                const MxConductorParams& params = *comp->as<MxConductorParams>();
                ok      = result.bsdf.add_bsdf<MxConductor>(cw, params, 1.0f);
                closure = nullptr;
                break;
            }
            case MX_GENERALIZED_SCHLICK_ID: {
                const MxGeneralizedSchlickParams& params
                    = *comp->as<MxGeneralizedSchlickParams>();
                if (is_black(params.transmission_tint))
                    ok = result.bsdf.add_bsdf<MxGeneralizedSchlickOpaque>(
                        cw, params, 1.0f);
                else
                    ok = result.bsdf.add_bsdf<MxGeneralizedSchlick>(
                        cw, params, result.refraction_ior);
                closure = nullptr;
                break;
            }
            case MX_SHEEN_ID: {
                const MxSheenParams& params = *comp->as<MxSheenParams>();
                ok      = result.bsdf.add_bsdf<MxSheen>(cw, params);
                closure = nullptr;
                break;
            }
            case MX_LAYER_ID: {
                // TODO: The weight handling here is questionable ...
                const MxLayerParams* srcparams = comp->as<MxLayerParams>();
                Color3 base_w
                    = cw
                      * (Color3(1, 1, 1)
                         - clamp(evaluate_layer_opacity(sg, srcparams->top),
                                 0.f, 1.f));
                closure = srcparams->top;
                weight  = cw;
                if (!is_black(base_w)) {
                    ptr_stack[stack_idx]      = srcparams->base;
                    weight_stack[stack_idx++] = base_w;
                }
                ok = true;
                break;
            }
            case MX_ANISOTROPIC_VDF_ID:
            case MX_MEDIUM_VDF_ID: {
                closure = nullptr;
                ok      = true;
                break;
            }
            default:
                printf("Unhandled closure ID: %s (%d)\n", id_to_string(id),
                       int(id));
                closure = nullptr;
                ok      = true;
                break;
            }
            if (!ok) {
                printf("Unable to add BSDF: %s (%d)\n", id_to_string(id),
                       int(id));
            }
        }
        }
        if (closure == nullptr && stack_idx > 0) {
            closure = ptr_stack[--stack_idx];
            weight  = weight_stack[stack_idx];
        }
    }
}


static __device__ void
process_closure(const ShaderGlobalsType& sg, ShadingResult& result,
                const void* Ci, bool light_only)
{
    if (!light_only) {
        process_medium_closure(sg, result, (const ClosureColor*) Ci, Color3(1));
    }
    process_closure(sg, (const ClosureColor*)Ci, result, light_only);
}


static __device__ Color3
process_background_closure(const ShaderGlobalsType& sg, const ClosureColor* closure)
{
    if (!closure) {
        return Color3(0);
    }

    // The depth of the closure tree must not exceed the stack size.
    // A stack size of 8 is probably quite generous for relatively
    // balanced trees.
    const int STACK_SIZE = 8;

    // Non-recursive traversal stack
    int stack_idx = 0;
    const ClosureColor* ptr_stack[STACK_SIZE];
    Color3 weight_stack[STACK_SIZE];

    // Shading accumulator
    Color3 weight = Color3(1.0f);
    while (closure) {
        ClosureIDs id = static_cast<ClosureIDs>(closure->id);
        switch (id) {
        case ADD: {
            ptr_stack[stack_idx]      = ((ClosureAdd*)closure)->closureB;
            weight_stack[stack_idx++] = weight;
            closure                   = ((ClosureAdd*)closure)->closureA;
            break;
        }
        case MUL: {
            weight *= ((ClosureMul*)closure)->weight;
            closure = ((ClosureMul*)closure)->closure;
            break;
        }
        case BACKGROUND_ID: {
            const ClosureComponent* comp = closure->as_comp();
            weight *= comp->w;
            closure = nullptr;
            break;
        }
        default:
            // Should never get here
            assert(false);
        }
        if (closure == nullptr && stack_idx > 0) {
            closure = ptr_stack[--stack_idx];
            weight  = weight_stack[stack_idx];
        }
    }
    return weight;
}
#endif


static __device__ Color3
eval_background(const Dual2<Vec3>& dir, void* /*ctx*/, int bounce = -1)
{
    ShaderGlobalsType sg;
    memset((char*)&sg, 0, sizeof(ShaderGlobalsType));
    sg.I    = V3_TO_F3(dir.val());
    sg.dIdx = V3_TO_F3(dir.dx());
    sg.dIdy = V3_TO_F3(dir.dy());
    if (bounce >= 0)
        sg.raytype = bounce > 0 ? Ray::DIFFUSE : Ray::CAMERA;
    sg.shaderID = render_params.bg_id;

    alignas(8) char closure_pool[256];
    execute_shader(sg, closure_pool);
    return process_background_closure((const ClosureColor*)sg.Ci);
}


// Return a direction towards a point on the sphere
// Adapted from Sphere::sample in ../raytracer.h
static __device__ float3
sample_sphere(const Vec3& x, const SphereParams& sphere, float xi, float yi,
              float& pdf)
{
    const float TWOPI = float(2 * M_PI);
    float cmax2       = 1 - sphere.r2 / dot(sphere.c - V3_TO_F3(x), sphere.c - V3_TO_F3(x));
    float cmax        = cmax2 > 0 ? sqrtf(cmax2) : 0;
    float cos_a       = 1 - xi + xi * cmax;
    float sin_a       = sqrtf(1 - cos_a * cos_a);
    float phi         = TWOPI * yi;
    float sp, cp;
    OIIO::fast_sincos(phi, &sp, &cp);
    float3 sw = normalize(sphere.c - V3_TO_F3(x)), su, sv;
    ortho(sw, su, sv);
    pdf = 1 / (TWOPI * (1 - cmax));
    return normalize(su * (cp * sin_a) + sv * (sp * sin_a) + sw * cos_a);
}


// Return a direction towards a point on the quad
// Adapted from Quad::sample in ../raytracer.h
static __device__ float3
sample_quad(const Vec3& x, const QuadParams& quad, float xi, float yi,
            float& pdf)
{
    float3 l   = (quad.p + xi * quad.ex + yi * quad.ey) - V3_TO_F3(x);
    float  d2  = dot(l, l); // l.length2();
    float3 dir = normalize(l);
    pdf        = d2 / (quad.a * fabsf(dot(dir, quad.n)));
    return dir;
}


static inline __device__ void
trace_ray(OptixTraversableHandle handle, const Payload& payload, const float3& origin,
          const float3& direction)
{
    uint32_t p0 = payload.raw[0];
    uint32_t p1 = payload.raw[1];
    uint32_t p2 = __float_as_uint(payload.radius);
    uint32_t p3 = __float_as_uint(payload.spread);
    uint32_t p4 = payload.raytype;

    optixTrace(handle,                         // handle
               origin,                         // origin
               direction,                      // direction
               1e-3f,                          // tmin
               1e13f,                          // tmax
               0,                              // ray time
               OptixVisibilityMask(1),         // visibility mask
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,  // ray flags
               0,                              // SBT offset
               1,                              // SBT stride
               0,                              // miss SBT offset
               p0, p1, p2, p3, p4);
};

//
// CudaScene
//

OSL_HOSTDEVICE bool
CudaScene::intersect(const Ray& r, Dual2<float>& t, int& primID, void* sg) const
{
    Payload payload;
    payload.sg_ptr  = reinterpret_cast<ShaderGlobalsType*>(sg);
    payload.radius  = r.radius;
    payload.spread  = r.spread;
    payload.raytype = *reinterpret_cast<const Ray::RayType*>(&r.raytype);
    // The renderer uses global object IDs across primitive types, but
    // OptiX uses object IDs for each primitive type. So we need to convert
    // between the two ranges.
    // TODO: Make this less convoluted.
    {
        int* tracedata = (int*)payload.sg_ptr->tracedata;
        int hit_kind   = tracedata[3];
        primID = (hit_kind == 0) ? primID - num_spheres : primID;
        tracedata[2]   = (hit_kind == 0) ? tracedata[2] - num_spheres
                                         : tracedata[2];
        trace_ray(handle, payload, V3_TO_F3(r.origin), V3_TO_F3(r.direction));
    }
    {
        int* tracedata = (int*)payload.sg_ptr->tracedata;
        int hit_kind   = tracedata[1];
        primID = (hit_kind == 0) ? tracedata[0] + num_spheres : tracedata[0];
        tracedata[0] = primID;
    }
    return (payload.sg_ptr->shaderID >= 0);
}


OSL_HOSTDEVICE float
CudaScene::shapepdf(int primID, const Vec3& x, const Vec3& p) const
{
    SphereParams* spheres = (SphereParams*)spheres_buffer;
    QuadParams* quads     = (QuadParams*)quads_buffer;
    return (primID < num_spheres)
               ? spheres[primID].shapepdf(x, p)
               : quads[primID - num_spheres].shapepdf(x, p);
}


OSL_HOSTDEVICE bool
CudaScene::islight(int primID) const
{
    SphereParams* spheres = (SphereParams*)spheres_buffer;
    QuadParams* quads     = (QuadParams*)quads_buffer;

    if (primID < num_spheres)
        return spheres[primID].isLight;
    return quads[primID - num_spheres].isLight;
}


OSL_HOSTDEVICE Vec3
CudaScene::sample(int primID, const Vec3& x, float xi, float yi,
                  float& pdf) const
{
    SphereParams* spheres = (SphereParams*)spheres_buffer;
    QuadParams* quads     = (QuadParams*)quads_buffer;

    float3 res;
    if (primID < num_spheres)
        res = sample_sphere(x, spheres[primID], xi, yi, pdf);
    else
        res = sample_quad(x, quads[primID - num_spheres], xi, yi, pdf);
    return F3_TO_V3(res);
}


OSL_HOSTDEVICE int
CudaScene::num_prims() const
{
    return num_spheres + num_quads;
}

//------------------------------------------------------------------------------

// Because clang++ 9.0 seems to have trouble with some of the texturing "intrinsics"
// let's do the texture look-ups in this file.
extern "C" __device__ float4
osl_tex2DLookup(void* handle, float s, float t, float dsdx, float dtdx, float dsdy, float dtdy)
{
    const float2 dx = {dsdx, dtdx};
    const float2 dy = {dsdy, dtdy};
    cudaTextureObject_t texID = cudaTextureObject_t(handle);
    return tex2DGrad<float4>(texID, s, t, dx, dy);
}


//
// OptiX Programs
//


extern "C" __global__ void
__miss__()
{
    uint3 launch_dims  = optixGetLaunchDimensions();
    uint3 launch_index = optixGetLaunchIndex();

    float3* output_buffer = reinterpret_cast<float3*>(
        render_params.output_buffer);

    int pixel            = launch_index.y * launch_dims.x + launch_index.x;
    output_buffer[pixel] = make_float3(0, 0, 1);
}


extern "C" __global__ void
__raygen__setglobals()
{
    uint3 launch_dims  = optixGetLaunchDimensions();
    uint3 launch_index = optixGetLaunchIndex();

    // Set global variables
    if (launch_index.x == 0 && launch_index.y == 0) {
        OSL::pvt::osl_printf_buffer_start
            = render_params.osl_printf_buffer_start;
        OSL::pvt::osl_printf_buffer_end = render_params.osl_printf_buffer_end;
        OSL::pvt::s_color_system        = render_params.color_system;
        OSL::pvt::test_str_1            = render_params.test_str_1;
        OSL::pvt::test_str_2            = render_params.test_str_2;
    }

    Background background;
    background.set_variables((Vec3*)render_params.bg_values,
                             (float*)render_params.bg_rows,
                             (float*)render_params.bg_cols,
                             render_params.bg_res);

    if (render_params.bg_id < 0)
        return;

    auto evaler = [](const Dual2<Vec3>& dir) {
        return eval_background(dir, nullptr);
    };

    // Background::prepare_cuda must run on a single warp
    assert(launch_index.x < 32 && launch_index.y == 0);
    background.prepare_cuda(launch_dims.x, launch_index.x, evaler);
}


extern "C" __global__ void
__miss__setglobals()
{
}


extern "C" __global__ void
__closesthit__deferred()
{
    Payload payload;
    payload.get();
    ShaderGlobalsType* sg_ptr = payload.sg_ptr;
    uint32_t* trace_data      = (uint32_t*)sg_ptr->tracedata;
    const float t_hit         = optixGetRayTmax();
    trace_data[0]             = optixGetPrimitiveIndex();
    trace_data[1]             = optixGetHitKind();
    trace_data[2]             = *(uint32_t*)&t_hit;
    globals_from_hit(*sg_ptr, payload.radius, payload.spread, payload.raytype);
}


extern "C" __global__ void
__raygen__deferred()
{
    Background background;
    background.set_variables((Vec3*)render_params.bg_values,
                             (float*)render_params.bg_rows,
                             (float*)render_params.bg_cols,
                             render_params.bg_res);

    Color3 result(0, 0, 0);
    const int aa = render_params.aa;
    for (int si = 0, n = aa * aa; si < n; si++) {
        uint3 launch_index = optixGetLaunchIndex();
        Sampler sampler(launch_index.x, launch_index.y, si);
        Vec3 j = sampler.get();
        // warp distribution to approximate a tent filter [-1,+1)^2
        j.x *= 2;
        j.x = j.x < 1 ? sqrtf(j.x) - 1 : 1 - sqrtf(2 - j.x);
        j.y *= 2;
        j.y = j.y < 1 ? sqrtf(j.y) - 1 : 1 - sqrtf(2 - j.y);

        if (render_params.no_jitter) {
            j *= 0.0f;
        }

        // Compute the pixel coordinates
        const float2 d
            = make_float2(static_cast<float>(launch_index.x) + 0.5f + j.x,
                          static_cast<float>(launch_index.y) + 0.5f + j.y);

        SimpleRaytracer raytracer;
        raytracer.background           = background;
        raytracer.backgroundResolution = render_params.bg_id >= 0
                                             ? render_params.bg_res
                                             : 0;
        raytracer.backgroundShaderID   = render_params.bg_id;
        raytracer.max_bounces          = render_params.max_bounces;
        raytracer.rr_depth             = 5;
        raytracer.show_albedo_scale    = render_params.show_albedo_scale;

        const Vec3 eye  = F3_TO_V3(render_params.eye);
        const Vec3 dir  = F3_TO_V3(render_params.dir);
        const Vec3 up   = F3_TO_V3(render_params.up);
        const float fov = render_params.fov;

        uint3 launch_dims = optixGetLaunchDimensions();
        raytracer.camera.resolution(launch_dims.x, launch_dims.y);
        raytracer.camera.lookat(eye, dir, up, fov);
        raytracer.camera.finalize();

        raytracer.scene = { render_params.num_spheres, render_params.num_quads,
                            render_params.spheres_buffer,
                            render_params.quads_buffer,
                            render_params.traversal_handle };

        Color3 r = raytracer.subpixel_radiance(d.x, d.y, sampler, nullptr);

        result = OIIO::lerp(result, r, 1.0f / (si + 1));
    }

    uint3 launch_dims     = optixGetLaunchDimensions();
    uint3 launch_index    = optixGetLaunchIndex();
    float3* output_buffer = reinterpret_cast<float3*>(
        render_params.output_buffer);
    int pixel            = launch_index.y * launch_dims.x + launch_index.x;
    output_buffer[pixel] = C3_TO_F3(result);
}

//------------------------------------------------------------------------------

// We need to pull in the definition of SimpleRaytracer::subpixel_radiance(),
// which is shared between the host and CUDA renderers.
#include "../simpleraytracer.cpp"

//------------------------------------------------------------------------------
