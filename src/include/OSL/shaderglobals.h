// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once


OSL_NAMESPACE_ENTER

struct ClosureColor;
class ShadingContext;
class RendererServices;



/// Type for an opaque pointer to whatever the renderer uses to represent a
/// coordinate transformation.
typedef const void* TransformationPtr;

/// Type for an opaque pointer to whatever the Shading System uses to represent
/// Uniform State.
typedef const void* OpaqueShadingStateUniformPtr;


/// The ShaderGlobals structure represents the state describing a particular
/// point to be shaded. It serves two primary purposes: (1) it holds the
/// values of the "global" variables accessible from a shader (such as P, N,
/// Ci, etc.); (2) it serves as a means of passing (via opaque pointers)
/// additional state between the renderer when it invokes the shader, and
/// the RendererServices that fields requests from OSL back to the renderer.
///
/// Except where noted, it is expected that all values are filled in by the
/// renderer before passing it to ShadingSystem::execute() to actually run
/// the shader. Not all fields will be valid in all contexts. In particular,
/// a few are only needed for lights and volumes.
///
/// All points, vectors and normals are given in "common" space.
///

/// We are working towards making ShaderGlobals a private implementation
/// detail.  Preference is given to utilizing an Opaque Execution Context
/// Pointer with accessor functions (setter/getters) to access shader global
/// values or any context for the currently executing shade.  These accessors
/// are at the bottom of this file following the form:
///     const Vec3& get_P(const OpaqueExecContextPtr oec);
///     const Vec3& get_N(const OpaqueExecContextPtr oec);
///     float get_time(const OpaqueExecContextPtr oec);
///     int get_shade_index(const OpaqueExecContextPtr oec);
/// Currently ExecutionContext is a ShaderGlobals and can be used
/// interchangably.  After inlining we don't expect worse performance.
/// Users should transition to the new accessor function API.  Once transition
/// is complete, where the values are stored and how they are populated is
/// expected to change and the new API will provide a layer of encapsulation
/// that the raw ShaderGlobals struct doesn't allow.
struct ShaderGlobals {
    /// Surface position (and its x & y differentials).
    Vec3 P, dPdx, dPdy;
    /// P's z differential, used for volume shading only.
    Vec3 dPdz;

    /// Incident ray, and its x and y derivatives.
    Vec3 I, dIdx, dIdy;

    /// Shading normal, already front-facing.
    Vec3 N;

    /// True geometric normal.
    Vec3 Ng;

    /// 2D surface parameter u, and its differentials.
    float u, dudx, dudy;
    /// 2D surface parameter v, and its differentials.
    float v, dvdx, dvdy;

    /// Surface tangents: derivative of P with respect to surface u and v.
    Vec3 dPdu, dPdv;

    /// Time for this shading sample.
    float time;
    /// Time interval for the frame (or shading sample).
    float dtime;
    ///  Velocity vector: derivative of position P with respect to time.
    Vec3 dPdtime;

    /// For lights or light attenuation shaders: the point being illuminated
    /// (Ps), and its differentials.
    Vec3 Ps, dPsdx, dPsdy;

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
    OpaqueShadingStateUniformPtr shadingStateUniform;

    /// Shading System responsible for setting
    /// 0 based thread index, required by journaling mechanism
    int thread_index;

    /// Shading System responsible for setting
    int shade_index;

    /// Pointer to the RendererServices object. This is how OSL finds its
    /// way back to the renderer for callbacks.
    RendererServices* renderer;

    /// Opaque pointers set by the renderer before shader execution, to
    /// allow later retrieval of the object->common and shader->common
    /// transformation matrices, by the RendererServices
    /// get_matrix/get_inverse_matrix methods. This doesn't need to point
    /// to the 4x4 matrix itself; rather, it's just a pointer to whatever
    /// structure the RenderServices::get_matrix() needs to (if and when
    /// requested) generate the 4x4 matrix for the right time value.
    TransformationPtr object2common;
    TransformationPtr shader2common;

    /// The output closure will be placed here. The renderer should
    /// initialize this to NULL before shading execution, and this is where
    /// it can retrieve the output closure from after shader execution has
    /// completed.
    ClosureColor* Ci;

    /// Surface area of the emissive object (used by light shaders for
    /// energy normalization).
    float surfacearea;

    /// Bit field of ray type flags.
    int raytype;

    /// If nonzero, will flip the result of calculatenormal().
    int flipHandedness;

    /// If nonzero, we are shading the back side of a surface.
    int backfacing;
};



/// Enum giving values that can be 'or'-ed together to make a bitmask
/// of which "global" variables are needed or written to by the shader.
enum class SGBits {
    None    = 0,
    P       = 1 << 0,
    I       = 1 << 1,
    N       = 1 << 2,
    Ng      = 1 << 3,
    u       = 1 << 4,
    v       = 1 << 5,
    dPdu    = 1 << 6,
    dPdv    = 1 << 7,
    time    = 1 << 8,
    dtime   = 1 << 9,
    dPdtime = 1 << 10,
    Ps      = 1 << 11,
    Ci      = 1 << 12,
    last
};

typedef void* OpaqueExecContextPtr;
namespace pvt {
// As the concrete ExecutionContext is an implementation detail,
// it is placed in the pvt namespace.
// New accessor functions are provided below in the form
//     OSL::get_???(const OpaqueExecContextPtr)
// Renderer's should use the accessor functions, and not attempt to cast or
// directly use the underlying ExecutionContext.  This will allow the
// implemenation to change and possibly have accessors generated in llvm if
// needed. OSL library function implementions may directly use the
// ExecutionContext.
typedef ShaderGlobals ExecContext;
typedef ExecContext* ExecContextPtr;

OSL_HOSTDEVICE inline ExecContextPtr
get_ec(OSL::OpaqueExecContextPtr oec)
{
    return reinterpret_cast<ExecContextPtr>(oec);
}
};  // namespace pvt



template<typename RenderStateT>
OSL_HOSTDEVICE inline RenderStateT*
get_rs(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    auto rs = reinterpret_cast<RenderStateT*>(ec->renderstate);
    return rs;
}

template<typename TraceDataT>
OSL_HOSTDEVICE inline TraceDataT*
get_tracedata(const OpaqueExecContextPtr oec)
{
    auto ec        = pvt::get_ec(oec);
    auto tracedata = reinterpret_cast<TraceDataT*>(ec->tracedata);
    return tracedata;
}

template<typename ObjectT>
OSL_HOSTDEVICE inline ObjectT*
get_objdata(const OpaqueExecContextPtr oec)
{
    auto ec      = pvt::get_ec(oec);
    auto objdata = reinterpret_cast<ObjectT*>(ec->objdata);
    return objdata;
}

// TODO: not sure ci should be exposed via a getter or not
//       its presence here maybe temporary
OSL_HOSTDEVICE inline ClosureColor*
get_Ci(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->Ci;
}

OSL_HOSTDEVICE inline TransformationPtr
get_object2common(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->object2common;
}

OSL_HOSTDEVICE inline TransformationPtr
get_shader2common(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->shader2common;
}

///Required by free function renderer services to process fmt specification calls; set by Shading System, and not by user
OSL_HOSTDEVICE inline int
get_shade_index(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->shade_index;
}

///Required for memory allocation on a per thread basis for journaling fmt specification calls; set by Shading System, and not by user
OSL_HOSTDEVICE inline int
get_thread_index(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->thread_index;
}

/// Surface position (and its x & y differentials).
OSL_HOSTDEVICE inline const Vec3&
get_P(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->P;
}

OSL_HOSTDEVICE inline const Vec3&
get_dPdx(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->dPdx;
}

OSL_HOSTDEVICE inline const Vec3&
get_dPdy(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->dPdy;
}


/// P's z differential, used for volume shading only.
OSL_HOSTDEVICE inline const Vec3&
get_dPdz(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->dPdz;
}

/// Incident ray, and its x and y derivatives.
OSL_HOSTDEVICE inline const Vec3&
get_I(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->I;
}

OSL_HOSTDEVICE inline const Vec3&
get_dIdx(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->dIdx;
}

OSL_HOSTDEVICE inline const Vec3&
get_dIdy(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->dIdy;
}


/// Shading normal, already front-facing.
OSL_HOSTDEVICE inline const Vec3&
get_N(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->N;
}

/// True geometric normal.
OSL_HOSTDEVICE inline const Vec3&
get_Ng(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->Ng;
}

/// 2D surface parameter u, and its differentials.
OSL_HOSTDEVICE inline float
get_u(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->u;
}

OSL_HOSTDEVICE inline float
get_dudx(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->dudx;
}

OSL_HOSTDEVICE inline float
get_dudy(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->dudy;
}

/// 2D surface parameter v, and its differentials.
OSL_HOSTDEVICE inline float
get_v(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->v;
}

OSL_HOSTDEVICE inline float
get_dvdx(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->dvdx;
}

OSL_HOSTDEVICE inline float
get_dvdy(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->dvdy;
}

/// Surface tangents: derivative of P with respect to surface u and v.
OSL_HOSTDEVICE inline const Vec3&
get_dPdu(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->dPdu;
}

OSL_HOSTDEVICE inline const Vec3&
get_dPdv(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->dPdv;
}

/// Time for this shading sample.
OSL_HOSTDEVICE inline float
get_time(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->time;
}

/// Time interval for the frame (or shading sample).
OSL_HOSTDEVICE inline float
get_dtime(const OpaqueExecContextPtr oec)
{
    auto ec = reinterpret_cast<OSL::ShaderGlobals*>(oec);
    return ec->dtime;
}

///  Velocity vector: derivative of position P with respect to time.
OSL_HOSTDEVICE inline const Vec3&
get_dPdtime(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->dPdtime;
}

/// For lights or light attenuation shaders: the point being illuminated
/// (Ps), and its differentials.
OSL_HOSTDEVICE inline const Vec3&
get_Ps(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->Ps;
}
OSL_HOSTDEVICE inline const Vec3&
get_dPsdx(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->dPsdx;
}
OSL_HOSTDEVICE inline const Vec3&
get_dPsdy(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->dPsdy;
}


/// If nonzero, we are shading the back side of a surface.
OSL_HOSTDEVICE inline int
get_backfacing(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->backfacing;
}

/// If nonzero, will flip the result of calculatenormal().
OSL_HOSTDEVICE inline int
get_flipHandedness(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->flipHandedness;
}

/// Bit field of ray type flags.
OSL_HOSTDEVICE inline int
get_raytype(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->raytype;
}

/// Surface area of the emissive object (used by light shaders for
/// energy normalization).
OSL_HOSTDEVICE inline float
get_surfacearea(const OpaqueExecContextPtr oec)
{
    auto ec = pvt::get_ec(oec);
    return ec->surfacearea;
}

namespace pvt {
// defined in opfmt.cpp
OSLEXECPUBLIC int
get_max_warnings_per_thread(const OSL::OpaqueExecContextPtr cptr);
}  // namespace pvt


// Useful to pass to OSL::journal::Writer::record_warningfmt
inline int
get_max_warnings_per_thread(const OpaqueExecContextPtr oec)
{
    return pvt::get_max_warnings_per_thread(oec);
}



OSL_NAMESPACE_EXIT
