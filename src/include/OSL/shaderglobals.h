/*
Copyright (c) 2009-2013 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once


OSL_NAMESPACE_ENTER

struct ClosureColor;
class ShadingContext;
class RendererServices;



/// Type for an opaque pointer to whatever the renderer uses to represent a
/// coordinate transformation.
typedef const void * TransformationPtr;




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

    /// The output closure will be placed here. The rendererer should
    /// initialize this to NULL before shading execution, and this is where
    /// it can retrieve the output closure from after shader execution has
    /// completed.
    ClosureColor *Ci;

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


OSL_NAMESPACE_EXIT
