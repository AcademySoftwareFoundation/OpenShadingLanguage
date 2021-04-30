// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#if !defined(OSL_USE_BATCHED) || (OSL_USE_BATCHED == 0)
#    error batched_shaderglobals.h should not be included unless OSL_USE_BATCHED is defined to 1
#endif

#include <OSL/wide.h>

#include <OSL/shaderglobals.h>

OSL_NAMESPACE_ENTER

namespace pvt {
class BatchedBackendLLVM;
}

struct UniformShaderGlobals {
    UniformShaderGlobals()                                  = default;
    UniformShaderGlobals(const UniformShaderGlobals& other) = delete;

    /// There are three opaque pointers that may be set by the renderer here
    /// in the ShaderGlobals before shading execution begins, and then
    /// retrieved again from the within the implementation of various
    /// RendererServices methods. Exactly what they mean and how they are
    /// used is renderer-dependent, but roughly speaking it's probably a
    /// pointer to some internal renderer state (needed for, say, figuring
    /// out how to retrieve userdata), state about the ray tree (needed to
    /// resume for a trace() call), and information about the object being
    /// shaded.
    void* renderstate;
    void* tracedata;
    void* objdata;

    /// Back-pointer to the ShadingContext (set and used by OSL itself --
    /// renderers shouldn't mess with this at all).
    ShadingContext* context;

    /// Pointer to the RendererServices object. This is how OSL finds its
    /// way back to the renderer for callbacks.
    RendererServices* renderer;

    /// The output closure will be placed here. The rendererer should
    /// initialize this to NULL before shading execution, and this is where
    /// it can retrieve the output closure from after shader execution has
    /// completed.
    /// DESIGN DECISION:  NOT CURRENTLY SUPPORTING CLOSURES IN BATCH MODE
    ClosureColor* Ci;

    /// Bit field of ray type flags.
    int raytype;

    // We want to manually pad this structure out to 64 byte boundary
    // and make it simple to duplicate in LLVM without relying on
    // compiler structure alignment rules
    int pad0;
    int pad1;
    int pad2;

    void dump()
    {
#define __OSL_DUMP(VARIABLE_NAME) \
    std::cout << #VARIABLE_NAME << " = " << VARIABLE_NAME << std::endl;
        std::cout << "UniformShaderGlobals = {" << std::endl;
        __OSL_DUMP(renderstate);
        __OSL_DUMP(tracedata);
        __OSL_DUMP(objdata);
        __OSL_DUMP(context);
        __OSL_DUMP(renderer);
        __OSL_DUMP(Ci);
        __OSL_DUMP(raytype);

        std::cout << "};" << std::endl;
#undef __OSL_DUMP
    }
};
static_assert(sizeof(UniformShaderGlobals) % 64 == 0,
              "UniformShaderGlobals must be padded to a 64byte boundary");

template<int WidthT> struct alignas(64) VaryingShaderGlobals {
    VaryingShaderGlobals()                                  = default;
    VaryingShaderGlobals(const VaryingShaderGlobals& other) = delete;

    template<typename T> using Block = OSL::Block<T, WidthT>;

    /// Surface position (and its x & y differentials).
    Block<Vec3> P, dPdx, dPdy;
    /// P's z differential, used for volume shading only.
    Block<Vec3> dPdz;

    /// Incident ray, and its x and y derivatives.
    Block<Vec3> I, dIdx, dIdy;

    /// Shading normal, already front-facing.
    Block<Vec3> N;

    /// True geometric normal.
    Block<Vec3> Ng;

    /// 2D surface parameter u, and its differentials.
    Block<float> u, dudx, dudy;
    /// 2D surface parameter v, and its differentials.
    Block<float> v, dvdx, dvdy;

    /// Surface tangents: derivative of P with respect to surface u and v.
    Block<Vec3> dPdu, dPdv;

    /// Time for this shading sample.
    Block<float> time;
    /// Time interval for the frame (or shading sample).
    Block<float> dtime;
    ///  Velocity vector: derivative of position P with respect to time.
    Block<Vec3> dPdtime;

    /// For lights or light attenuation shaders: the point being illuminated
    /// (Ps), and its differentials.
    Block<Vec3> Ps, dPsdx, dPsdy;

    /// Opaque pointers set by the renderer before shader execution, to
    /// allow later retrieval of the object->common and shader->common
    /// transformation matrices, by the RendererServices
    /// get_matrix/get_inverse_matrix methods. This doesn't need to point
    /// to the 4x4 matrix itself; rather, it's just a pointer to whatever
    /// structure the RenderServices::get_matrix() needs to (if and when
    /// requested) generate the 4x4 matrix for the right time value.
    Block<TransformationPtr> object2common;
    Block<TransformationPtr> shader2common;

    /// Surface area of the emissive object (used by light shaders for
    /// energy normalization).
    Block<float> surfacearea;

    /// If nonzero, will flip the result of calculatenormal().
    Block<int> flipHandedness;

    /// If nonzero, we are shading the back side of a surface.
    Block<int> backfacing;

    void dump()
    {
#define __OSL_DUMP(VARIABLE_NAME) VARIABLE_NAME.dump(#VARIABLE_NAME)
        std::cout << "VaryingShaderGlobals = {" << std::endl;
        __OSL_DUMP(P);
        __OSL_DUMP(dPdx);
        __OSL_DUMP(dPdy);
        __OSL_DUMP(dPdz);
        __OSL_DUMP(I);
        __OSL_DUMP(dIdx);
        __OSL_DUMP(dIdy);
        __OSL_DUMP(N);
        __OSL_DUMP(Ng);
        __OSL_DUMP(u);
        __OSL_DUMP(dudx);
        __OSL_DUMP(dudy);
        __OSL_DUMP(v);
        __OSL_DUMP(dvdx);
        __OSL_DUMP(dvdy);
        __OSL_DUMP(dPdu);
        __OSL_DUMP(dPdv);
        __OSL_DUMP(time);
        __OSL_DUMP(dtime);
        __OSL_DUMP(dPdtime);
        __OSL_DUMP(Ps);
        __OSL_DUMP(dPsdx);
        __OSL_DUMP(dPsdy);
        __OSL_DUMP(object2common);
        __OSL_DUMP(shader2common);
        __OSL_DUMP(surfacearea);
        __OSL_DUMP(flipHandedness);
        __OSL_DUMP(backfacing);
        std::cout << "};" << std::endl;
#undef __OSL_DUMP
    }
};

template<int WidthT> struct alignas(64) BatchedShaderGlobals {
    BatchedShaderGlobals() {}

    // Disallow copying
    BatchedShaderGlobals(const BatchedShaderGlobals&) = delete;

    UniformShaderGlobals uniform;
    VaryingShaderGlobals<WidthT> varying;

    void dump()
    {
        std::cout << "BatchedShaderGlobals"
                  << " = {" << std::endl;
        uniform.dump();
        varying.dump();
        std::cout << "};" << std::endl;
    }
};

#define __OSL_USING_SHADERGLOBALS(WIDTH_OF_OSL_DATA) \
    using BatchedShaderGlobals                       \
        = OSL_NAMESPACE::BatchedShaderGlobals<WIDTH_OF_OSL_DATA>;

#undef OSL_USING_DATA_WIDTH
#ifdef __OSL_USING_BATCHED_TEXTURE
#    define OSL_USING_DATA_WIDTH(WIDTH_OF_OSL_DATA)  \
        __OSL_USING_WIDE(WIDTH_OF_OSL_DATA)          \
        __OSL_USING_SHADERGLOBALS(WIDTH_OF_OSL_DATA) \
        __OSL_USING_BATCHED_TEXTURE(WIDTH_OF_OSL_DATA)
#else
#    define OSL_USING_DATA_WIDTH(WIDTH_OF_OSL_DATA) \
        __OSL_USING_WIDE(WIDTH_OF_OSL_DATA)         \
        __OSL_USING_SHADERGLOBALS(WIDTH_OF_OSL_DATA)
#endif

OSL_NAMESPACE_EXIT
