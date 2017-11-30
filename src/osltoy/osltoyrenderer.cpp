/*
Copyright (c) 2017 Sony Pictures Imageworks Inc., et al.
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


#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/timer.h>
#include <OSL/oslexec.h>
#include "osltoyrenderer.h"
using namespace OSL;



OSL_NAMESPACE_ENTER

static ustring u_camera("camera"), u_screen("screen");
static ustring u_NDC("NDC"), u_raster("raster");
static ustring u_perspective("perspective");
static ustring u_s("s"), u_t("t");
static ustring u_mouse("mouse");
static constexpr TypeDesc TypeFloatArray2 (TypeDesc::FLOAT, 2);
static constexpr TypeDesc TypeFloatArray4 (TypeDesc::FLOAT, 4);
static constexpr TypeDesc TypeIntArray2 (TypeDesc::INT, 2);





OSLToyRenderer::OSLToyRenderer ()
{
    m_shadingsys = new ShadingSystem (this);
    m_shadingsys->attribute ("allow_shader_replacement", 1);
    ustring outputs[] = { ustring("Cout") };
    m_shadingsys->attribute ("renderer_outputs", TypeDesc(TypeDesc::STRING,1), &outputs);
    // set attributes for the shadingsys

    Matrix44 M;  M.makeIdentity();
    camera_params (M, u_perspective, 90.0f,
                   0.1f, 1000.0f, 256, 256);

    // Set up getters
    m_attr_getters[ustring("osl:version")] = &OSLToyRenderer::get_osl_version;
    m_attr_getters[ustring("camera:resolution")] = &OSLToyRenderer::get_camera_resolution;
    m_attr_getters[ustring("camera:projection")] = &OSLToyRenderer::get_camera_projection;
    m_attr_getters[ustring("camera:pixelaspect")] = &OSLToyRenderer::get_camera_pixelaspect;
    m_attr_getters[ustring("camera:screen_window")] = &OSLToyRenderer::get_camera_screen_window;
    m_attr_getters[ustring("camera:fov")] = &OSLToyRenderer::get_camera_fov;
    m_attr_getters[ustring("camera:clip")] = &OSLToyRenderer::get_camera_clip;
    m_attr_getters[ustring("camera:clip_near")] = &OSLToyRenderer::get_camera_clip_near;
    m_attr_getters[ustring("camera:clip_far")] = &OSLToyRenderer::get_camera_clip_far;
    m_attr_getters[ustring("camera:shutter")] = &OSLToyRenderer::get_camera_shutter;
    m_attr_getters[ustring("camera:shutter_open")] = &OSLToyRenderer::get_camera_shutter_open;
    m_attr_getters[ustring("camera:shutter_close")] = &OSLToyRenderer::get_camera_shutter_close;

    // Set up default shaderglobals
    ShaderGlobals &sg (m_shaderglobals_template);
    memset (&sg, 0, sizeof(ShaderGlobals));
    Matrix44 Mshad, Mobj;  // just let these be identity for now
    // Set "shader" space to be Mshad.  In a real renderer, this may be
    // different for each shader group.
    sg.shader2common = OSL::TransformationPtr (&Mshad);
    // Set "object" space to be Mobj.  In a real renderer, this may be
    // different for each object.
    sg.object2common = OSL::TransformationPtr (&Mobj);
    // Just make it look like all shades are the result of 'raytype' rays.
    sg.raytype = 0;  // default ray type
    // Set the surface area of the patch to 1 (which it is).  This is
    // only used for light shaders that call the surfacearea() function.
    sg.surfacearea = 1;
    // Derivs are constant across the image
    // if (shadelocations == ShadePixelCenters) {
        sg.dudx  = 1.0f / m_xres;    // sg.dudy is already 0
        sg.dvdy  = 1.0f / m_yres;    // sg.dvdx is already 0
    // } else {
    //     sg.dudx  = 1.0f / std::max(1,(m_xres-1));
    //     sg.dvdy  = 1.0f / std::max(1,(m_yres-1));
    // }
    // Derivatives with respect to x,y
    sg.dPdx = Vec3 (1.0f, 0.0f, 0.0f);
    sg.dPdy = Vec3 (0.0f, 1.0f, 0.0f);
    sg.dPdz = Vec3 (0.0f, 0.0f, 1.0f);
    // Tangents of P with respect to surface u,v
    sg.dPdu = Vec3 (m_xres, 0.0f, 0.0f);
    sg.dPdv = Vec3 (0.0f, m_yres, 0.0f);
    sg.dPdz = Vec3 (0.0f, 0.0f, 0);
    // That also implies that our normal points to (0,0,1)
    sg.N    = Vec3 (0, 0, 1);
    sg.Ng   = Vec3 (0, 0, 1);
    // In our SimpleRenderer, the "renderstate" itself just a pointer to
    // the ShaderGlobals.
    // sg.renderstate = &sg;
}



void
OSLToyRenderer::render_image ()
{
    if (! m_framebuffer.initialized())
        m_framebuffer.reset (OIIO::ImageSpec (m_xres, m_yres, 3, TypeDesc::FLOAT));

    static ustring outputs[] = { ustring("Cout") };
//    OIIO::Timer timer;
    OIIO::ImageBufAlgo::parallel_image_options popt;
#if OPENIMAGEIO_VERSION > 10902
    popt.minitems = 4096;
    popt.splitdir = OIIO::Split_Tile;
    popt.recursive = true;
#endif
    shade_image (*shadingsys(), *shadergroup(), &m_shaderglobals_template,
                 m_framebuffer, outputs, ShadePixelCenters, OIIO::ROI(), popt);
//    std::cout << timer() << "\n";
}



int
OSLToyRenderer::supports (string_view feature) const
{
    return false;
}



void
OSLToyRenderer::camera_params (const Matrix44 &world_to_camera,
                               ustring projection, float hfov,
                               float hither, float yon,
                               int xres, int yres)
{
    m_world_to_camera = world_to_camera;
    m_projection = projection;
    m_fov = hfov;
    m_pixelaspect = 1.0f; // hard-coded
    m_hither = hither;
    m_yon = yon;
    m_shutter[0] = 0.0f; m_shutter[1] = 1.0f;  // hard-coded
    float frame_aspect = float(xres)/float(yres) * m_pixelaspect;
    m_screen_window[0] = -frame_aspect;
    m_screen_window[1] = -1.0f;
    m_screen_window[2] =  frame_aspect;
    m_screen_window[3] =  1.0f;
    m_xres = xres;
    m_yres = yres;
}



bool
OSLToyRenderer::get_matrix (ShaderGlobals *sg, Matrix44 &result,
                            TransformationPtr xform,
                            float time)
{
    // OSLToyRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    result = *reinterpret_cast<const Matrix44*>(xform);
    return true;
}



bool
OSLToyRenderer::get_matrix (ShaderGlobals *sg, Matrix44 &result,
                            ustring from, float time)
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
OSLToyRenderer::get_matrix (ShaderGlobals *sg, Matrix44 &result,
                            TransformationPtr xform)
{
    // OSLToyRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    result = *(OSL::Matrix44 *)xform;
    return true;
}



bool
OSLToyRenderer::get_matrix (ShaderGlobals *sg, Matrix44 &result,
                            ustring from)
{
    // OSLToyRenderer doesn't understand motion blur, so we never fail
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
OSLToyRenderer::get_inverse_matrix (ShaderGlobals *sg, Matrix44 &result,
                                    ustring to, float time)
{
    if (to == u_camera || to == u_screen || to == u_NDC || to == u_raster) {
        Matrix44 M = m_world_to_camera;
        if (to == u_screen || to == u_NDC || to == u_raster) {
            float depthrange = (double)m_yon-(double)m_hither;
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
OSLToyRenderer::name_transform (const char *name, const OSL::Matrix44 &xform)
    {
    std::shared_ptr<Transformation> M (new OSL::Matrix44 (xform));
    m_named_xforms[ustring(name)] = M;
}



bool
OSLToyRenderer::get_array_attribute (ShaderGlobals *sg, bool derivatives, ustring object,
                                     TypeDesc type, ustring name,
                                     int index, void *val)
{
    AttrGetterMap::const_iterator g = m_attr_getters.find (name);
    if (g != m_attr_getters.end()) {
        AttrGetter getter = g->second;
        return (this->*(getter)) (sg, derivatives, object, type, name, val);
    }

    if (object == u_mouse) {
        if (name == u_s && type == TypeDesc::FLOAT && m_mouse_x >= 0) {
            *(float *)val = (m_mouse_x + 0.5f) / float(m_xres);
            return true;
        }
        if (name == u_t && type == TypeDesc::FLOAT && m_mouse_y >= 0) {
            *(float *)val = (m_mouse_y + 0.5f) / float(m_yres);
            return true;
        }
    }

    // In order to test getattribute(), respond positively to
    // "options"/"blahblah"
    if (object == "options" && name == "blahblah" &&
        type == TypeDesc::TypeFloat) {
        *(float *)val = 3.14159;
        return true;
    }

    // If no named attribute was found, allow userdata to bind to the
    // attribute request.
    if (object.empty() && index == -1)
        return get_userdata (derivatives, name, type, sg, val);

    return false;
}



bool
OSLToyRenderer::get_attribute (ShaderGlobals *sg, bool derivatives, ustring object,
                               TypeDesc type, ustring name, void *val)
{
    return get_array_attribute (sg, derivatives, object,
                                type, name, -1, val);
}



bool
OSLToyRenderer::get_userdata (bool derivatives, ustring name, TypeDesc type,
                              ShaderGlobals *sg, void *val)
{
    // Just to illustrate how this works, respect s and t userdata, filled
    // in with the uv coordinates.  In a real renderer, it would probably
    // look up something specific to the primitive, rather than have hard-
    // coded names.

    if (name == u_s && type == TypeDesc::TypeFloat) {
        ((float *)val)[0] = sg->u;
        if (derivatives) {
            ((float *)val)[1] = sg->dudx;
            ((float *)val)[2] = sg->dudy;
        }
        return true;
    }
    if (name == u_t && type == TypeDesc::TypeFloat) {
        ((float *)val)[0] = sg->v;
        if (derivatives) {
            ((float *)val)[1] = sg->dvdx;
            ((float *)val)[2] = sg->dvdy;
        }
        return true;
    }

    return false;
}


bool
OSLToyRenderer::get_osl_version (ShaderGlobals *sg, bool derivs, ustring object,
                                 TypeDesc type, ustring name, void *val)
{
    if (type == TypeDesc::TypeInt) {
        ((int *)val)[0] = OSL_VERSION;
        return true;
    }
    return false;
}


bool
OSLToyRenderer::get_camera_resolution (ShaderGlobals *sg, bool derivs, ustring object,
                                       TypeDesc type, ustring name, void *val)
{
    if (type == TypeIntArray2) {
        ((int *)val)[0] = m_xres;
        ((int *)val)[1] = m_yres;
        return true;
    }
    return false;
}


bool
OSLToyRenderer::get_camera_projection (ShaderGlobals *sg, bool derivs, ustring object,
                                       TypeDesc type, ustring name, void *val)
{
    if (type == TypeDesc::TypeString) {
        ((ustring *)val)[0] = m_projection;
        return true;
    }
    return false;
}


bool
OSLToyRenderer::get_camera_fov (ShaderGlobals *sg, bool derivs, ustring object,
                                TypeDesc type, ustring name, void *val)
{
    // N.B. in a real rederer, this may be time-dependent
    if (type == TypeDesc::TypeFloat) {
        ((float *)val)[0] = m_fov;
        if (derivs)
            memset ((char *)val+type.size(), 0, 2*type.size());
        return true;
    }
    return false;
}


bool
OSLToyRenderer::get_camera_pixelaspect (ShaderGlobals *sg, bool derivs, ustring object,
                                        TypeDesc type, ustring name, void *val)
{
    if (type == TypeDesc::TypeFloat) {
        ((float *)val)[0] = m_pixelaspect;
        if (derivs)
            memset ((char *)val+type.size(), 0, 2*type.size());
        return true;
    }
    return false;
}


bool
OSLToyRenderer::get_camera_clip (ShaderGlobals *sg, bool derivs, ustring object,
                                 TypeDesc type, ustring name, void *val)
{
    if (type == TypeFloatArray2) {
        ((float *)val)[0] = m_hither;
        ((float *)val)[1] = m_yon;
        if (derivs)
            memset ((char *)val+type.size(), 0, 2*type.size());
        return true;
    }
    return false;
}


bool
OSLToyRenderer::get_camera_clip_near (ShaderGlobals *sg, bool derivs, ustring object,
                                      TypeDesc type, ustring name, void *val)
{
    if (type == TypeDesc::TypeFloat) {
        ((float *)val)[0] = m_hither;
        if (derivs)
            memset ((char *)val+type.size(), 0, 2*type.size());
        return true;
    }
    return false;
}


bool
OSLToyRenderer::get_camera_clip_far (ShaderGlobals *sg, bool derivs, ustring object,
                                     TypeDesc type, ustring name, void *val)
{
    if (type == TypeDesc::TypeFloat) {
        ((float *)val)[0] = m_yon;
        if (derivs)
            memset ((char *)val+type.size(), 0, 2*type.size());
        return true;
    }
    return false;
}



bool
OSLToyRenderer::get_camera_shutter (ShaderGlobals *sg, bool derivs, ustring object,
                                    TypeDesc type, ustring name, void *val)
{
    if (type == TypeFloatArray2) {
        ((float *)val)[0] = m_shutter[0];
        ((float *)val)[1] = m_shutter[1];
        if (derivs)
            memset ((char *)val+type.size(), 0, 2*type.size());
        return true;
    }
    return false;
}


bool
OSLToyRenderer::get_camera_shutter_open (ShaderGlobals *sg, bool derivs, ustring object,
                                         TypeDesc type, ustring name, void *val)
{
    if (type == TypeDesc::TypeFloat) {
        ((float *)val)[0] = m_shutter[0];
        if (derivs)
            memset ((char *)val+type.size(), 0, 2*type.size());
        return true;
    }
    return false;
}


bool
OSLToyRenderer::get_camera_shutter_close (ShaderGlobals *sg, bool derivs, ustring object,
                                          TypeDesc type, ustring name, void *val)
{
    if (type == TypeDesc::TypeFloat) {
        ((float *)val)[0] = m_shutter[1];
        if (derivs)
            memset ((char *)val+type.size(), 0, 2*type.size());
        return true;
    }
    return false;
}


bool
OSLToyRenderer::get_camera_screen_window (ShaderGlobals *sg, bool derivs, ustring object,
                                          TypeDesc type, ustring name, void *val)
{
    // N.B. in a real rederer, this may be time-dependent
    if (type == TypeFloatArray4) {
        ((float *)val)[0] = m_screen_window[0];
        ((float *)val)[1] = m_screen_window[1];
        ((float *)val)[2] = m_screen_window[2];
        ((float *)val)[3] = m_screen_window[3];
        if (derivs)
            memset ((char *)val+type.size(), 0, 2*type.size());
        return true;
    }
    return false;
}


OSL_NAMESPACE_EXIT
