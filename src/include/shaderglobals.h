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



/// Opaque pointer to whatever the renderer uses to represent a
/// (potentially motion-blurred) coordinate transformation.
typedef const void * TransformationPtr;




/// This struct represents the global variables accessible from a shader, note
/// that not all fields will be valid in all contexts.
///
/// All points, vectors and normals are given in "common" space.
struct ShaderGlobals {
    Vec3 P, dPdx, dPdy;              /**< Position */
    Vec3 dPdz;                       /**< z zeriv for volume shading */
    Vec3 I, dIdx, dIdy;              /**< Incident ray */
    Vec3 N;                          /**< Shading normal */
    Vec3 Ng;                         /**< True geometric normal */
    float u, dudx, dudy;             /**< Surface parameter u */
    float v, dvdx, dvdy;             /**< Surface parameter v */
    Vec3 dPdu, dPdv;                 /**< Tangents on the surface */
    float time;                      /**< Time for each sample */
    float dtime;                     /**< Time interval for each sample */
    Vec3 dPdtime;                    /**< Velocity */
    Vec3 Ps, dPsdx, dPsdy;           /**< Point being lit (valid only in light
                                          attenuation shaders */
    void* renderstate;               /**< Opaque pointer to renderer state (can
                                          be used to retrieve renderer specific
                                          details like userdata) */
    void* tracedata;                 /**< Opaque pointer to renderer state
                                          resuling from a trace() call. */
    void* objdata;                   /**< Opaque pointer to object data */
    ShadingContext* context;         /**< ShadingContext (this will be set by
                                          OSL itself) */
    RendererServices* renderer;      /**< Ptr to the RendererServices object */                                        
    TransformationPtr object2common; /**< Object->common xform */
    TransformationPtr shader2common; /**< Shader->common xform */
    ClosureColor *Ci;                /**< Output closure (should be initialized
                                          to NULL) */
    float surfacearea;               /**< Total area of the object (defined by
                                          light shaders for energy normalization) */
    int raytype;                     /**< Bit field of ray type flags */
    int flipHandedness;              /**< flips the result of calculatenormal() */
    int backfacing;                  /**< True if we want are shading the
                                          backside of the surface */
};


OSL_NAMESPACE_EXIT
