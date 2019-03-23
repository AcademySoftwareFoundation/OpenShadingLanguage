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

#include <OpenImageIO/parallel.h>

#include "simplerend.h"
#include "raytracer.h"
#include "shading.h"
using namespace OSL;

OSL_NAMESPACE_ENTER

static ustring u_camera("camera"), u_screen("screen");
static ustring u_NDC("NDC"), u_raster("raster");
static ustring u_perspective("perspective");
static ustring u_s("s"), u_t("t");
static TypeDesc TypeFloatArray2 (TypeDesc::FLOAT, 2);
static TypeDesc TypeFloatArray4 (TypeDesc::FLOAT, 4);
static TypeDesc TypeIntArray2 (TypeDesc::INT, 2);




SimpleRenderer::SimpleRenderer ()
{
    Matrix44 M;  M.makeIdentity();
    camera_params (M, u_perspective, 90.0f,
                   0.1f, 1000.0f, 256, 256);

    // Set up getters
    m_attr_getters[ustring("osl:version")] = &SimpleRenderer::get_osl_version;
    m_attr_getters[ustring("camera:resolution")] = &SimpleRenderer::get_camera_resolution;
    m_attr_getters[ustring("camera:projection")] = &SimpleRenderer::get_camera_projection;
    m_attr_getters[ustring("camera:pixelaspect")] = &SimpleRenderer::get_camera_pixelaspect;
    m_attr_getters[ustring("camera:screen_window")] = &SimpleRenderer::get_camera_screen_window;
    m_attr_getters[ustring("camera:fov")] = &SimpleRenderer::get_camera_fov;
    m_attr_getters[ustring("camera:clip")] = &SimpleRenderer::get_camera_clip;
    m_attr_getters[ustring("camera:clip_near")] = &SimpleRenderer::get_camera_clip_near;
    m_attr_getters[ustring("camera:clip_far")] = &SimpleRenderer::get_camera_clip_far;
    m_attr_getters[ustring("camera:shutter")] = &SimpleRenderer::get_camera_shutter;
    m_attr_getters[ustring("camera:shutter_open")] = &SimpleRenderer::get_camera_shutter_open;
    m_attr_getters[ustring("camera:shutter_close")] = &SimpleRenderer::get_camera_shutter_close;
}



OIIO::ParamValue*
SimpleRenderer::find_attribute(string_view name, TypeDesc searchtype,
                               bool casesensitive)
{
    auto iter = options.find(name, searchtype, casesensitive);
    if (iter != options.end())
        return &(*iter);
    return nullptr;
}



const OIIO::ParamValue*
SimpleRenderer::find_attribute(string_view name, TypeDesc searchtype,
                               bool casesensitive) const
{
    auto iter = options.find(name, searchtype, casesensitive);
    if (iter != options.end())
        return &(*iter);
    return nullptr;
}



void
SimpleRenderer::attribute (string_view name, TypeDesc type, const void *value)
{
    if (name.empty())  // Guard against bogus empty names
        return;
    // Don't allow duplicates
    auto f = find_attribute(name);
    if (!f) {
        options.resize(options.size() + 1);
        f = &options.back();
    }
    f->init(name, type, 1, value);
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
SimpleRenderer::get_matrix (ShaderGlobals *sg, Matrix44 &result,
                            TransformationPtr xform,
                            float time)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    result = *reinterpret_cast<const Matrix44*>(xform);
    return true;
}



bool
SimpleRenderer::get_matrix (ShaderGlobals *sg, Matrix44 &result,
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
SimpleRenderer::get_matrix (ShaderGlobals *sg, Matrix44 &result,
                            TransformationPtr xform)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    result = *reinterpret_cast<const Matrix44*>(xform);
    return true;
}



bool
SimpleRenderer::get_matrix (ShaderGlobals *sg, Matrix44 &result,
                            ustring from)
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
SimpleRenderer::get_inverse_matrix (ShaderGlobals *sg, Matrix44 &result,
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
SimpleRenderer::name_transform (const char *name, const OSL::Matrix44 &xform)
{
    std::shared_ptr<Transformation> M (new OSL::Matrix44 (xform));
    m_named_xforms[ustring(name)] = M;
}



bool
SimpleRenderer::get_array_attribute (ShaderGlobals *sg, bool derivatives, ustring object,
                                     TypeDesc type, ustring name,
                                     int index, void *val)
{
    AttrGetterMap::const_iterator g = m_attr_getters.find (name);
    if (g != m_attr_getters.end()) {
        AttrGetter getter = g->second;
        return (this->*(getter)) (sg, derivatives, object, type, name, val);
    }

    // If no named attribute was found, allow userdata to bind to the
    // attribute request.
    if (object.empty() && index == -1)
        return get_userdata (derivatives, name, type, sg, val);

    return false;
}



bool
SimpleRenderer::get_attribute (ShaderGlobals *sg, bool derivatives, ustring object,
                               TypeDesc type, ustring name, void *val)
{
    return get_array_attribute (sg, derivatives, object,
                                type, name, -1, val);
}



bool
SimpleRenderer::get_userdata (bool derivatives, ustring name, TypeDesc type,
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
SimpleRenderer::get_osl_version (ShaderGlobals *sg, bool derivs, ustring object,
                                    TypeDesc type, ustring name, void *val)
{
    if (type == TypeDesc::TypeInt) {
        ((int *)val)[0] = OSL_VERSION;
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_resolution (ShaderGlobals *sg, bool derivs, ustring object,
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
SimpleRenderer::get_camera_projection (ShaderGlobals *sg, bool derivs, ustring object,
                                    TypeDesc type, ustring name, void *val)
{
    if (type == TypeDesc::TypeString) {
        ((ustring *)val)[0] = m_projection;
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_fov (ShaderGlobals *sg, bool derivs, ustring object,
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
SimpleRenderer::get_camera_pixelaspect (ShaderGlobals *sg, bool derivs, ustring object,
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
SimpleRenderer::get_camera_clip (ShaderGlobals *sg, bool derivs, ustring object,
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
SimpleRenderer::get_camera_clip_near (ShaderGlobals *sg, bool derivs, ustring object,
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
SimpleRenderer::get_camera_clip_far (ShaderGlobals *sg, bool derivs, ustring object,
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
SimpleRenderer::get_camera_shutter (ShaderGlobals *sg, bool derivs, ustring object,
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
SimpleRenderer::get_camera_shutter_open (ShaderGlobals *sg, bool derivs, ustring object,
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
SimpleRenderer::get_camera_shutter_close (ShaderGlobals *sg, bool derivs, ustring object,
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
SimpleRenderer::get_camera_screen_window (ShaderGlobals *sg, bool derivs, ustring object,
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



void
SimpleRenderer::globals_from_hit(ShaderGlobals& sg, const Ray& r,
                                 const Dual2<float>& t, int id, bool flip)
{
    memset((char *)&sg, 0, sizeof(ShaderGlobals));
    Dual2<Vec3> P = r.point(t);
    sg.P = P.val(); sg.dPdx = P.dx(); sg.dPdy = P.dy();
    Dual2<Vec3> N = scene.normal(P, id);
    sg.Ng = sg.N = N.val();
    Dual2<Vec2> uv = scene.uv(P, N, sg.dPdu, sg.dPdv, id);
    sg.u = uv.val().x; sg.dudx = uv.dx().x; sg.dudy = uv.dy().x;
    sg.v = uv.val().y; sg.dvdx = uv.dx().y; sg.dvdy = uv.dy().y;
    sg.surfacearea = scene.surfacearea(id);
    sg.I = r.direction.val();
    sg.dIdx = r.direction.dx();
    sg.dIdy = r.direction.dy();
    sg.backfacing = sg.N.dot(sg.I) > 0;
    if (sg.backfacing) {
        sg.N = -sg.N;
        sg.Ng = -sg.Ng;
    }
    sg.flipHandedness = flip;

    // In our SimpleRenderer, the "renderstate" itself just a pointer to
    // the ShaderGlobals.
    sg.renderstate = &sg;
}

Vec3 SimpleRenderer::eval_background(const Dual2<Vec3>& dir, ShadingContext* ctx) {
    ShaderGlobals sg;
    memset((char *)&sg, 0, sizeof(ShaderGlobals));
    sg.I = dir.val();
    sg.dIdx = dir.dx();
    sg.dIdy = dir.dy();
    shadingsys->execute(*ctx, *shaders[backgroundShaderID], sg);
    return process_background_closure(sg.Ci);
}

Color3 SimpleRenderer::subpixel_radiance(float x, float y, Sampler& sampler, ShadingContext* ctx) {
    Ray r = camera.get(x, y);
    Color3 path_weight(1, 1, 1);
    Color3 path_radiance(0, 0, 0);
    int prev_id = -1;
    float bsdf_pdf = std::numeric_limits<float>::infinity(); // camera ray has only one possible direction
    bool flip = false;
    for (int b = 0; b <= max_bounces; b++) {
        // trace the ray against the scene
        Dual2<float> t; int id = prev_id;
        if (!scene.intersect(r, t, id)) {
            // we hit nothing? check background shader
            if (backgroundShaderID >= 0) {
                if (backgroundResolution > 0) {
                    float bg_pdf = 0;
                    Vec3 bg = background.eval(r.direction.val(), bg_pdf);
                    path_radiance += path_weight * bg * MIS::power_heuristic<MIS::WEIGHT_WEIGHT>(bsdf_pdf, bg_pdf);
                } else {
                    // we aren't importance sampling the background - so just run it directly
                    path_radiance += path_weight * eval_background(r.direction, ctx);
                }
            }
            break;
        }

        // construct a shader globals for the hit point
        ShaderGlobals sg;
        globals_from_hit(sg, r, t, id, flip);
        int shaderID = scene.shaderid(id);
        if (shaderID < 0 || !shaders[shaderID]) break; // no shader attached? done

        // execute shader and process the resulting list of closures
        shadingsys->execute (*ctx, *shaders[shaderID], sg);
        ShadingResult result;
        bool last_bounce = b == max_bounces;
        process_closure(result, sg.Ci, last_bounce);

        // add self-emission
        float k = 1;
        if (scene.islight(id)) {
            // figure out the probability of reaching this point
            float light_pdf = scene.shapepdf(id, r.origin.val(), sg.P);
            k = MIS::power_heuristic<MIS::WEIGHT_EVAL>(bsdf_pdf, light_pdf);
        }
        path_radiance += path_weight * k * result.Le;

        // last bounce? nothing left to do
        if (last_bounce) break;

        // build internal pdf for sampling between bsdf closures
        result.bsdf.prepare(sg, path_weight, b >= rr_depth);

        // get two random numbers
        Vec3 s = sampler.get();
        float xi = s.x;
        float yi = s.y;
        float zi = s.z;

        // trace one ray to the background
        if (backgroundResolution > 0) {
            Dual2<Vec3> bg_dir;
            float bg_pdf = 0, bsdf_pdf = 0;
            Vec3 bg = background.sample(xi, yi, bg_dir, bg_pdf);
            Color3 bsdf_weight = result.bsdf.eval(sg, bg_dir.val(), bsdf_pdf);
            Color3 contrib = path_weight * bsdf_weight * bg * MIS::power_heuristic<MIS::WEIGHT_WEIGHT>(bg_pdf, bsdf_pdf);
            if ((contrib.x + contrib.y + contrib.z) > 0) {
                int shadow_id = id;
                Ray shadow_ray = Ray(sg.P, bg_dir);
                Dual2<float> shadow_dist;
                if (!scene.intersect(shadow_ray, shadow_dist, shadow_id)) // ray reached the background?
                    path_radiance += contrib;
            }
        }

        // trace one ray to each light
        for (int lid = 0; lid < scene.num_prims(); lid++) {
            if (lid == id) continue; // skip self
            if (!scene.islight(lid)) continue; // doesn't want to be sampled as a light
            int shaderID = scene.shaderid(lid);
            if (shaderID < 0 || !shaders[shaderID]) continue; // no shader attached to this light
            // sample a random direction towards the object
            float light_pdf;
            Vec3 ldir = scene.sample(lid, sg.P, xi, yi, light_pdf);
            float bsdf_pdf = 0;
            Color3 bsdf_weight = result.bsdf.eval(sg, ldir, bsdf_pdf);
            Color3 contrib = path_weight * bsdf_weight * MIS::power_heuristic<MIS::EVAL_WEIGHT>(light_pdf, bsdf_pdf);
            if ((contrib.x + contrib.y + contrib.z) > 0) {
                Ray shadow_ray = Ray(sg.P, ldir);
                // trace a shadow ray and see if we actually hit the target
                // in this tiny renderer, tracing a ray is probably cheaper than evaluating the light shader
                int shadow_id = id; // ignore self hit
                Dual2<float> shadow_dist;
                if (scene.intersect(shadow_ray, shadow_dist, shadow_id) && shadow_id == lid) {
                    // setup a shader global for the point on the light
                    ShaderGlobals light_sg;
                    globals_from_hit(light_sg, shadow_ray, shadow_dist, lid, false);
                    // execute the light shader (for emissive closures only)
                    shadingsys->execute (*ctx, *shaders[shaderID], light_sg);
                    ShadingResult light_result;
                    process_closure(light_result, light_sg.Ci, true);
                    // accumulate contribution
                    path_radiance += contrib * light_result.Le;
                }
            }
        }

        // trace indirect ray and continue
        path_weight *= result.bsdf.sample(sg, xi, yi, zi, r.direction, bsdf_pdf);
        if (!(path_weight.x > 0) && !(path_weight.y > 0) && !(path_weight.z > 0))
            break; // filter out all 0's or NaNs
        prev_id = id;
        r.origin = Dual2<Vec3>(sg.P, sg.dPdx, sg.dPdy);
        flip ^= sg.Ng.dot(r.direction.val()) > 0;
    }
    return path_radiance;
}

Color3 SimpleRenderer::antialias_pixel(int x, int y, ShadingContext* ctx)
{
    Color3 result(0, 0, 0);
    for (int ay = 0, si = 0; ay < aa; ay++) {
        for (int ax = 0; ax < aa; ax++, si++) {
            Sampler sampler(x, y, si, aa);
            // jitter pixel coordinate [0,1)^2
            Vec3 j = sampler.get();
            // warp distribution to approximate a tent filter [-1,+1)^2
            j.x *= 2; j.x = j.x < 1 ? sqrtf(j.x) - 1 : 1 - sqrtf(2 - j.x);
            j.y *= 2; j.y = j.y < 1 ? sqrtf(j.y) - 1 : 1 - sqrtf(2 - j.y);
            // trace eye ray (apply jitter from center of the pixel)
            result += subpixel_radiance(x + 0.5f + j.x, y + 0.5f + j.y, sampler, ctx);
        }
    }
    return result / float(aa * aa);
}


void
SimpleRenderer::prepare_render ()
{
    // Retrieve and validate options
    aa = std::max (1, options.get_int("aa"));
    max_bounces = options.get_int("max_bounces");
    rr_depth = options.get_int("rr_depth");

    // prepare background importance table (if requested)
    if (backgroundResolution > 0 && backgroundShaderID >= 0) {
        // get a context so we can make several background shader calls
        OSL::PerThreadInfo *thread_info = shadingsys->create_thread_info();
        ShadingContext *ctx = shadingsys->get_context (thread_info);

        // build importance table to optimize background sampling
        auto evaler = [this](const Dual2<Vec3>& dir, ShadingContext* ctx){
            return this->eval_background(dir, ctx);
        };
        background.prepare(backgroundResolution, evaler, ctx);

        // release context
        shadingsys->release_context (ctx);
        shadingsys->destroy_thread_info(thread_info);
    } else {
        // we aren't directly evaluating the background
        backgroundResolution = 0;
    }
}



void
SimpleRenderer::render (int xres, int yres)
{
    ShadingSystem *shadingsys = this->shadingsys;
    OIIO::parallel_for_chunked (0, yres, 0,
      [&, this](int64_t ybegin, int64_t yend){
        // Request an OSL::PerThreadInfo for this thread.
        OSL::PerThreadInfo *thread_info = shadingsys->create_thread_info();

        // Request a shading context so that we can execute the shader.
        // We could get_context/release_context for each shading point,
        // but to save overhead, it's more efficient to reuse a context
        // within a thread.
        ShadingContext *ctx = shadingsys->get_context (thread_info);

        OIIO::ImageBuf::Iterator<float> p(pixelbuf, OIIO::ROI(0,xres,ybegin,yend));
        for ( ; !p.done(); ++p) {
            Color3 c = antialias_pixel(p.x(), p.y(), ctx);
            p[0] = c[0];
            p[1] = c[1];
            p[2] = c[2];
        }

        // We're done shading with this context.
        shadingsys->release_context (ctx);
        shadingsys->destroy_thread_info(thread_info);
    });
}



OSL_NAMESPACE_EXIT
