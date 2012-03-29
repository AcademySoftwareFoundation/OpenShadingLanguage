/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
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


#include "oslexec.h"
#include "simplerend.h"
using namespace OSL;

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {

static ustring u_camera("camera"), u_screen("screen");
static ustring u_NDC("NDC"), u_raster("raster");
static ustring u_perspective("perspective");



SimpleRenderer::SimpleRenderer ()
{
    Matrix44 M;  M.makeIdentity();
    camera_params (M, u_perspective, 90.0f,
                   0.1f, 1000.0f, 256, 256);
}



void
SimpleRenderer::camera_params (const Matrix44 &world_to_camera,
                               ustring projection, float hfov,
                               float hither, float yon,
                               int xres, int yres)
{
    m_world_to_camera = world_to_camera;
    m_projection = projection;
    m_fov = hfov;
    m_hither = hither;
    m_yon = yon;
    m_xres = xres;
    m_yres = yres;
}



bool
SimpleRenderer::get_matrix (Matrix44 &result, TransformationPtr xform,
                            float time)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    result = *(OSL::Matrix44 *)xform;
    return true;
}



bool
SimpleRenderer::get_matrix (Matrix44 &result, ustring from, float time)
{
    TransformMap::const_iterator found = m_named_xforms.find (from);
    if (found != m_named_xforms.end()) {
        result = *(found->second);
        return true;
    } else {
        return false;
    }
}



bool
SimpleRenderer::get_matrix (Matrix44 &result, TransformationPtr xform)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    result = *(OSL::Matrix44 *)xform;
    return true;
}



bool
SimpleRenderer::get_matrix (Matrix44 &result, ustring from)
{
    // SimpleRenderer doesn't understand motion blur, so we never fail
    // on account of time-varying transformations.
    TransformMap::const_iterator found = m_named_xforms.find (from);
    if (found != m_named_xforms.end()) {
        result = *(found->second);
        return true;
    } else {
        return false;
    }
}



bool
SimpleRenderer::get_inverse_matrix (Matrix44 &result, ustring to, float time)
{
    if (to == u_camera || to == u_screen || to == u_NDC || to == u_raster) {
        Matrix44 M = m_world_to_camera;
        if (to == u_screen || to == u_NDC || to == u_raster) {
            float depthrange = (double)m_yon-(double)m_hither;
            static ustring u_perspective("perspective");
            if (m_projection == u_perspective) {
                float tanhalffov = tanf (0.5f * m_fov * M_PI/180.0);
                Matrix44 camera_to_screen (1/tanhalffov, 0, 0, 0,
                                           0, 1/tanhalffov, 0, 0,
                                           0, 0, m_yon/depthrange, 1,
                                           0, 0, -m_yon*m_hither/depthrange, 0);
                M = M * camera_to_screen;
            } else {
                Matrix44 camera_to_screen (1, 0, 0, 0,
                                           0, 1, 0, 0,
                                           0, 0, 1/depthrange, 0,
                                           0, 0, -m_hither/depthrange, 1);
                M = M * camera_to_screen;
            }
            if (to == u_NDC || to == u_raster) {
                float screenleft = -1.0, screenwidth = 2.0;
                float screenbottom = -1.0, screenheight = 2.0;
                Matrix44 screen_to_ndc (1/screenwidth, 0, 0, 0,
                                        0, 1/screenheight, 0, 0,
                                        0, 0, 1, 0,
                                        -screenleft/screenwidth, -screenbottom/screenheight, 0, 1);
                M = M * screen_to_ndc;
                if (to == u_raster) {
                    Matrix44 ndc_to_raster (m_xres, 0, 0, 0,
                                            0, m_yres, 0, 0,
                                            0, 0, 1, 0,
                                            0, 0, 0, 1);
                    M = M * ndc_to_raster;
                }
            }
        }
        result = M;
        return true;
    }

    TransformMap::const_iterator found = m_named_xforms.find (to);
    if (found != m_named_xforms.end()) {
        result = *(found->second);
        result.invert();
        return true;
    } else {
        return false;
    }
}



void
SimpleRenderer::name_transform (const char *name, const OSL::Matrix44 &xform)
{
    shared_ptr<Transformation> M (new OSL::Matrix44 (xform));
    m_named_xforms[ustring(name)] = M;
}

bool
SimpleRenderer::get_array_attribute (void *renderstate, bool derivatives, ustring object,
                                     TypeDesc type, ustring name,
                                     int index, void *val)
{
    return false;
}

bool
SimpleRenderer::get_attribute (void *renderstate, bool derivatives, ustring object,
                               TypeDesc type, ustring name, void *val)
{
    return false;
}

bool
SimpleRenderer::get_userdata (bool derivatives, ustring name, TypeDesc type, void *renderstate, void *val)
{
    return false;
}

bool
SimpleRenderer::has_userdata (ustring name, TypeDesc type, void *renderstate)
{
    return false;
}

int
SimpleRenderer::pointcloud_search (ShaderGlobals *sg,
                                   ustring filename, const OSL::Vec3 &center,
                                   float radius, int max_points, bool sort,
                                   size_t *out_indices,
                                   float *out_distances, int derivs_offset)
{
    return 0;
}

int
SimpleRenderer::pointcloud_get (ustring filename, size_t *indices, int count,
                                ustring attr_name, TypeDesc attr_type,
                                void *out_data)
{
    return 0;
}

};  // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
