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


#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/argparse.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/timer.h>

#ifdef USE_EXTERNAL_PUGIXML
# include <pugixml.hpp>
#else
# include <OpenImageIO/pugixml.hpp>
#endif

#include <boost/thread.hpp>
#include <boost/ref.hpp>

#include "OSL/oslexec.h"
#include "simplerend.h"
#include "raytracer.h"
#include "background.h"
#include "shading.h"
#include "sampling.h"
#include "util.h"


using namespace OIIO;
using namespace OSL;

namespace { // anonymous namespace

ShadingSystem *shadingsys = NULL;
bool debug = false;
bool debug2 = false;
bool verbose = false;
bool stats = false;
bool O0 = false, O1 = false, O2 = false;
bool debugnan = false;
int xres = 640, yres = 480, aa = 1, max_bounces = 1000000, rr_depth = 5;
int num_threads = 0;
ErrorHandler errhandler;
SimpleRenderer rend;  // RendererServices
Camera camera;
Scene scene;
int backgroundShaderID = -1;
int backgroundResolution = 0;
Background background;
std::vector<ShaderGroupRef> shaders;
std::string scenefile, imagefile;

int get_filenames(int argc, const char *argv[])
{
    for (int i = 0; i < argc; i++) {
        if (scenefile.empty())
            scenefile = argv[i];
        else if (imagefile.empty())
            imagefile = argv[i];
    }
    return 0;
}

void getargs(int argc, const char *argv[])
{
    bool help = false;
    ArgParse ap;
    ap.options ("Usage:  testrender [options] scene.xml output.exr",
                "%*", get_filenames, "",
                "--help", &help, "Print help message",
                "-v", &verbose, "Verbose messages",
                "--debug", &debug, "Lots of debugging info",
                "--debug2", &debug2, "Even more debugging info",
                "--stats", &stats, "Print run statistics",
                "-r %d %d", &xres, &yres, "Render a WxH image",
                "-aa %d", &aa, "Trace NxN rays per pixel",
                "-t %d", &num_threads, "Render using N threads (default: auto-detect)",
                "-O0", &O0, "Do no runtime shader optimization",
                "-O1", &O1, "Do a little runtime shader optimization",
                "-O2", &O2, "Do lots of runtime shader optimization",
                "--debugnan", &debugnan, "Turn on 'debugnan' mode",
                NULL);
    if (ap.parse(argc, argv) < 0) {
        std::cerr << ap.geterror() << std::endl;
        ap.usage ();
        exit (EXIT_FAILURE);
    }
    if (help) {
        std::cout <<
            "testrender -- Test Renderer for Open Shading Language\n"
            "(c) Copyright 2009-2010 Sony Pictures Imageworks Inc. All Rights Reserved.\n";
        ap.usage ();
        exit (EXIT_SUCCESS);
    }
    if (scenefile.empty()) {
        std::cerr << "testrender: Must specify an xml scene file to open\n";
        ap.usage();
        exit (EXIT_FAILURE);
    }
    if (imagefile.empty()) {
        std::cerr << "testrender: Must specify a filename for output render\n";
        ap.usage();
        exit (EXIT_FAILURE);
    }
    if (debug || verbose)
        errhandler.verbosity (ErrorHandler::VERBOSE);
}

Vec3 strtovec(const char* str) {
    Vec3 v(0, 0, 0);
    sscanf(str, " %f , %f , %f", &v.x, &v.y, &v.z);
    return v;
}

int strtoint(const char* str) {
    int i = 0;
    sscanf(str, " %d", &i);
    return i;
}

float strtoflt(const char* str) {
    float f = 0;
    sscanf(str, " %f", &f);
    return f;
}

bool strtobool(const char* str) {
    return strcmp(str, "1") == 0 ||
           strcmp(str, "on") == 0 ||
           strcmp(str, "yes") == 0;
}

template <int N>
struct ParamStorage {
    ParamStorage() : fparamindex(0), iparamindex(0), sparamindex(0) {}

    void* Int(int i) {
        ASSERT(iparamindex < N);
        iparamdata[iparamindex] = i;
        iparamindex++;
        return &iparamdata[iparamindex - 1];
    }

    void* Float(float f) {
        ASSERT(fparamindex < N);
        fparamdata[fparamindex] = f;
        fparamindex++;
        return &fparamdata[fparamindex - 1];
    }

    void* Vec(float x, float y, float z) {
        Float(x);
        Float(y);
        Float(z);
        return &fparamdata[fparamindex - 3];
    }

    void* Str(const char* str) {
        ASSERT(sparamindex < N);
        sparamdata[sparamindex] = ustring(str);
        sparamindex++;
        return &sparamdata[sparamindex - 1];
    }
private:
    // storage for shader parameters
    float   fparamdata[N];
    int     iparamdata[N];
    ustring sparamdata[N];

    int fparamindex;
    int iparamindex;
    int sparamindex;
};


void parse_scene() {
    // setup default camera (now that resolution is finalized)
    camera = Camera(Vec3(0,0,0), Vec3(0,0,-1), Vec3(0,1,0), 90.0f, xres, yres);

    // load entire text file into a buffer
    std::ifstream file(scenefile.c_str(), std::ios::binary);
    std::vector<char> text((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    if (text.empty()) {
        std::cerr << "Error reading " << scenefile << "\n"
                  << "File is either missing or empty\n";
        exit (EXIT_FAILURE);
    }
    text.push_back(0); // make sure text ends with trailing 0

    // build DOM tree
    pugi::xml_document doc;
    pugi::xml_parse_result parse_result = doc.load_file(scenefile.c_str());
    if (!parse_result) {
        std::cerr << "XML parsed with errors: " << parse_result.description() << ", at offset " << parse_result.offset << "\n";
        exit (EXIT_FAILURE);
    }
    pugi::xml_node root = doc.child("World");
    if (!root) {
        std::cerr << "Error reading " << scenefile << "\n"
                  << "Root element <World> is missing\n";
        exit (EXIT_FAILURE);
    }
    // loop over all children of world
    for (pugi::xml_node node = root.first_child(); node; node = node.next_sibling()) {
        if (strcmp(node.name(), "Option") == 0) {
            for (pugi::xml_attribute attr = node.first_attribute(); attr; attr = attr.next_attribute()) {
                int i = 0;
                if (sscanf(attr.value(), " int %d ", &i) == 1) {
                    if (strcmp(attr.name(), "max_bounces") == 0)
                        max_bounces = i;
                    else if (strcmp(attr.name(), "rr_depth") == 0)
                        rr_depth = i;
                }
                // TODO: pass any extra options to shading system (or texture system?)
            }
        } else if (strcmp(node.name(), "Camera") == 0) {
            // defaults
            Vec3 eye(0,0,0);
            Vec3 dir(0,0,-1);
            Vec3 up(0,1,0);
            float fov = 90.f;

            // load camera (only first attribute counts if duplicates)
            pugi::xml_attribute eye_attr = node.attribute("eye");
            pugi::xml_attribute dir_attr = node.attribute("dir");
            pugi::xml_attribute  at_attr = node.attribute("look_at");
            pugi::xml_attribute  up_attr = node.attribute("up");
            pugi::xml_attribute fov_attr = node.attribute("fov");

            if (eye_attr) eye = strtovec(eye_attr.value());
            if (dir_attr) dir = strtovec(dir_attr.value()); else
            if ( at_attr) dir = strtovec( at_attr.value()) - eye;
            if ( up_attr)  up = strtovec( up_attr.value());
            if (fov_attr) fov = strtoflt(fov_attr.value());

            // create actual camera
            camera = Camera(eye, dir, up, fov, xres, yres);
        } else if (strcmp(node.name(), "Sphere") == 0) {
            // load sphere
            pugi::xml_attribute center_attr = node.attribute("center");
            pugi::xml_attribute radius_attr = node.attribute("radius");
            if (center_attr && radius_attr) {
                Vec3  center = strtovec(center_attr.value());
                float radius = strtoflt(radius_attr.value());
                if (radius > 0) {
                    pugi::xml_attribute light_attr = node.attribute("is_light");
                    bool is_light = light_attr ? strtobool(light_attr.value()) : false;
                    scene.add_sphere(Sphere(center, radius, int(shaders.size()) - 1, is_light));
                }
            }
        } else if (strcmp(node.name(), "Quad") == 0) {
            // load quad
            pugi::xml_attribute corner_attr = node.attribute("corner");
            pugi::xml_attribute edge_x_attr = node.attribute("edge_x");
            pugi::xml_attribute edge_y_attr = node.attribute("edge_y");
            if (corner_attr && edge_x_attr && edge_y_attr) {
                pugi::xml_attribute light_attr = node.attribute("is_light");
                bool is_light = light_attr ? strtobool(light_attr.value()) : false;
                Vec3 co = strtovec(corner_attr.value());
                Vec3 ex = strtovec(edge_x_attr.value());
                Vec3 ey = strtovec(edge_y_attr.value());
                scene.add_quad(Quad(co, ex, ey, int(shaders.size()) - 1, is_light));
            }
        } else if (strcmp(node.name(), "Background") == 0) {
            pugi::xml_attribute res_attr = node.attribute("resolution");
            if (res_attr)
                backgroundResolution = strtoint(res_attr.value());
            backgroundShaderID = int(shaders.size()) - 1;
        } else if (strcmp(node.name(), "ShaderGroup") == 0) {
            ShaderGroupRef group = shadingsys->ShaderGroupBegin();
            ParamStorage<1024> store; // scratch space to hold parameters until they are read by Shader()
            for (pugi::xml_node gnode = node.first_child(); gnode; gnode = gnode.next_sibling()) {
                if (strcmp(gnode.name(), "Parameter") == 0) {
                    // handle parameters
                    for (pugi::xml_attribute attr = gnode.first_attribute(); attr; attr = attr.next_attribute()) {
                        int i = 0; float x = 0, y = 0, z = 0;
                        if (sscanf(attr.value(), " int %d ", &i) == 1)
                            shadingsys->Parameter(attr.name(), TypeDesc::TypeInt, store.Int(i));
                        else if (sscanf(attr.value(), " float %f ", &x) == 1)
                            shadingsys->Parameter(attr.name(), TypeDesc::TypeFloat, store.Float(x));
                        else if (sscanf(attr.value(), " vector %f %f %f", &x, &y, &z) == 3)
                            shadingsys->Parameter(attr.name(), TypeDesc::TypeVector, store.Vec(x, y, z));
                        else if (sscanf(attr.value(), " point %f %f %f", &x, &y, &z) == 3)
                            shadingsys->Parameter(attr.name(), TypeDesc::TypePoint, store.Vec(x, y, z));
                        else if (sscanf(attr.value(), " color %f %f %f", &x, &y, &z) == 3)
                            shadingsys->Parameter(attr.name(), TypeDesc::TypeColor, store.Vec(x, y, z));
                        else
                            shadingsys->Parameter(attr.name(), TypeDesc::TypeString, store.Str(attr.value()));
                    }
                } else if (strcmp(gnode.name(), "Shader") == 0) {
                    pugi::xml_attribute  type_attr = gnode.attribute("type");
                    pugi::xml_attribute  name_attr = gnode.attribute("name");
                    pugi::xml_attribute layer_attr = gnode.attribute("layer");
                    const char* type = type_attr ? type_attr.value() : "surface";
                    if (name_attr && layer_attr)
                        shadingsys->Shader(type, name_attr.value(), layer_attr.value());
                } else if (strcmp(gnode.name(), "ConnectShaders") == 0) {
                    // FIXME: find a more elegant way to encode this
                    pugi::xml_attribute  sl = gnode.attribute("srclayer");
                    pugi::xml_attribute  sp = gnode.attribute("srcparam");
                    pugi::xml_attribute  dl = gnode.attribute("dstlayer");
                    pugi::xml_attribute  dp = gnode.attribute("dstparam");
                    if (sl && sp && dl && dp)
                        shadingsys->ConnectShaders(sl.value(), sp.value(),
                                                   dl.value(), dp.value());
                } else {
                    // unknow element?
                }
            }
            shadingsys->ShaderGroupEnd();
            shaders.push_back (group);
        } else {
            // unknown element?
        }
    }
    if (root.next_sibling()) {
        std::cerr << "Error reading " << scenefile << "\n"
                  << "Found multiple top-level elements\n";
        exit (EXIT_FAILURE);
    }
}

void globals_from_hit(ShaderGlobals& sg, const Ray& r, const Dual2<float>& t, int id, bool flip) {
    memset(&sg, 0, sizeof(ShaderGlobals));
    Dual2<Vec3> P = r.point(t);
    sg.P = P.val(); sg.dPdx = P.dx(); sg.dPdy = P.dy();
    Dual2<Vec3> N = scene.normal(P, id);
    sg.Ng = sg.N = N.val();
    Dual2<Vec2> uv = scene.uv(P, N, sg.dPdu, sg.dPdv, id);
    sg.u = uv.val().x; sg.dudx = uv.dx().x; sg.dudy = uv.dy().x;
    sg.v = uv.val().y; sg.dvdx = uv.dx().y; sg.dvdy = uv.dy().y;
    sg.surfacearea = scene.surfacearea(id);
    sg.I = r.d.val();
    sg.dIdx = r.d.dx();
    sg.dIdy = r.d.dy();
    sg.backfacing = sg.N.dot(sg.I) > 0;
    if (sg.backfacing) {
        sg.N = -sg.N;
        sg.Ng = -sg.Ng;
    }
    sg.flipHandedness = flip;
}

Vec3 eval_background(const Dual2<Vec3>& dir, ShadingContext* ctx) {
    ShaderGlobals sg;
    memset(&sg, 0, sizeof(ShaderGlobals));
    sg.I = dir.val();
    sg.dIdx = dir.dx();
    sg.dIdy = dir.dy();
    shadingsys->execute(*ctx, *shaders[backgroundShaderID], sg);
    return process_background_closure(sg.Ci);
}

float mis_weight(float invpdf, float otherpdf) {
    // power heuristic is: a^2 / (a^2+b^2)
    // simplifying: 1 / (1 + (b/a)^2)
    return 1 / (1 + otherpdf * otherpdf * invpdf * invpdf);
}

Color3 subpixel_radiance(float x, float y, Rng& rand, ShadingContext* ctx) {
    Ray r = camera.get(x, y);
    Color3 path_weight(1, 1, 1);
    Color3 path_radiance(0, 0, 0);
    int prev_id = -1;
    float bsdf_invpdf = 0;
    bool flip = false;
    for (int b = 0; b <= max_bounces; b++) {
        // trace the ray against the scene
        Dual2<float> t; int id = prev_id;
        if (!scene.intersect(r, t, id)) {
            // we hit nothing? check background shader
            if (backgroundShaderID >= 0) {
                if (backgroundResolution > 0) {
                    float bg_pdf = 0;
                    Vec3 bg = background.eval(r.d.val(), bg_pdf);
                    path_radiance += path_weight * bg * mis_weight(bsdf_invpdf, bg_pdf);
                } else {
                    // we aren't importance sampling the background - so just run it directly
                    path_radiance += path_weight * eval_background(r.d, ctx);
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
            float light_pdf = scene.shapepdf(id, r.o.val(), sg.P);
            k = mis_weight(bsdf_invpdf, light_pdf);
        }
        path_radiance += path_weight * k * result.Le;

        // last bounce? nothing left to do
        if (last_bounce) break;

        // build internal pdf for sampling between bsdf closures
        result.bsdf.prepare(sg, path_weight, b >= rr_depth);

        // get two random numbers
        float xi = rand;
        float yi = rand;

        // trace one ray to the background
        if (backgroundResolution > 0) {
            Dual2<Vec3> bg_dir;
            float bg_invpdf = 0, bsdf_pdf = 0;
            Vec3 bg = background.sample(xi, yi, bg_dir, bg_invpdf);
            Color3 bsdf_eval = result.bsdf.eval(sg, bg_dir.val(), bsdf_pdf);
            bsdf_eval *= path_weight * bg * bg_invpdf * mis_weight(bg_invpdf, bsdf_pdf);
            if ((bsdf_eval.x + bsdf_eval.y + bsdf_eval.z) > 0) {
                int shadow_id = id;
                Ray shadow_ray = Ray(sg.P, bg_dir);
                Dual2<float> shadow_dist;
                if (!scene.intersect(shadow_ray, shadow_dist, shadow_id)) // ray reached the background?
                    path_radiance += bsdf_eval;
            }
        }

        // trace one ray to each light
        for (int lid = 0; !result.bsdf.singular() && lid < scene.num_prims(); lid++) {
            if (lid == id) continue; // skip self
            if (!scene.islight(lid)) continue; // doesn't want to be sampled as a light
            int shaderID = scene.shaderid(lid);
            if (shaderID < 0 || !shaders[shaderID]) continue; // no shader attached to this light
            // sample a random direction towards the object
            float invpdf;
            Vec3 ldir = scene.sample(lid, sg.P, xi, yi, invpdf);
            float bsdf_pdf = 0;
            Color3 bsdf_eval = path_weight * result.bsdf.eval(sg, ldir, bsdf_pdf);
            if ((bsdf_eval.x + bsdf_eval.y + bsdf_eval.z) == 0) continue;
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
                // run the result through the bsdfs attached to the current point
                path_radiance += bsdf_eval * light_result.Le * invpdf * mis_weight(invpdf, bsdf_pdf);
            }
        }

        // trace indirect ray and continue
        path_weight *= result.bsdf.sample(sg, xi, yi, r.d, bsdf_invpdf);
        if (!(path_weight.x > 0) && !(path_weight.y > 0) && !(path_weight.z > 0))
            break; // filter out all 0's or NaNs
        prev_id = id;
        r.o = Dual2<Vec3>(sg.P, sg.dPdx, sg.dPdy);
        flip ^= sg.Ng.dot(r.d.val()) > 0;
    }
    return path_radiance;
}

Color3 antialias_pixel(int x, int y, Rng& rand, ShadingContext* ctx) {
    Color3 result(0, 0, 0);
    float invaa = 1.0f / aa;
    for (int ay = 0; ay < aa; ay++) {
        for (int ax = 0; ax < aa; ax++) {
            // jitter pixel coordinate [0,1)^2
            float jx = (ax + rand) * invaa;
            float jy = (ay + rand) * invaa;
            // warp distribution to approximate a tent filter [-1,+1)^2
            jx *= 2; jx = jx < 1 ? sqrtf(jx) - 1 : 1 - sqrtf(2 - jx);
            jy *= 2; jy = jy < 1 ? sqrtf(jy) - 1 : 1 - sqrtf(2 - jy);
            // trace eye ray (apply jitter from center of the pixel)
            result += subpixel_radiance(x + 0.5f + jx, y + 0.5f + jy, rand, ctx);
        }
    }
    return result * (invaa * invaa);
}

void scanline_worker(Counter& counter, std::vector<Color3>& pixels) {
    // Optional: high-performance apps may request this thread-specific
    // pointer in order to save a bit of time on each shade.  Just like
    // the name implies, a multithreaded renderer would need to do this
    // separately for each thread, and be careful to always use the same
    // thread_info each time for that thread.
    //
    // There's nothing wrong with a simpler app just passing NULL for
    // the thread_info; in such a case, the ShadingSystem will do the
    // necessary calls to find the thread-specific pointer itself, but
    // this will degrade performance just a bit.
    OSL::PerThreadInfo *thread_info = shadingsys->create_thread_info();

    // Request a shading context so that we can execute the shader.
    // We could get_context/release_constext for each shading point,
    // but to save overhead, it's more efficient to reuse a context
    // within a thread.
    ShadingContext *ctx = shadingsys->get_context (thread_info);

    int y;
    while (counter.getnext(y)) {
        for (int x = 0, i = xres * y;  x < xres;  ++x, ++i) {
            Rng rand(i);
            pixels[i] = antialias_pixel(x, y, rand, ctx);
        }
    }
    // We're done shading with this context.
    shadingsys->release_context (ctx);

    // Now that we're done rendering, release the thread=specific
    // pointer we saved.  A simple app could skip this; but if the app
    // asks for it (as we have in this example), then it should also
    // destroy it when done with it.
    shadingsys->destroy_thread_info(thread_info);
}


} // anonymous namespace

int main (int argc, const char *argv[]) {
    Timer timer;

    // Create a new shading system.  We pass it the RendererServices
    // object that services callbacks from the shading system, NULL for
    // the TextureSystem (which will create a default OIIO one), and
    // an error handler.
    shadingsys = new ShadingSystem (&rend, NULL, &errhandler);

    // Register the layout of all closures known to this renderer
    // Any closure used by the shader which is not registered, or
    // registered with a different number of arguments will lead
    // to a runtime error.
    register_closures(shadingsys);

    // Remember that each shader parameter may optionally have a
    // metadata hint [[int lockgeom=...]], where 0 indicates that the
    // parameter may be overridden by the geometry itself, for example
    // with data interpolated from the mesh vertices, and a value of 1
    // means that it is "locked" with respect to the geometry (i.e. it
    // will not be overridden with interpolated or
    // per-geometric-primitive data).
    // 
    // In order to most fully optimize shader, we typically want any
    // shader parameter not explicitly specified to default to being
    // locked (i.e. no per-geometry override):
    shadingsys->attribute("lockgeom", 1);

    // Read command line arguments
    getargs (argc, argv);

    // Setup common attributes
    shadingsys->attribute ("debug", debug2 ? 2 : (debug ? 1 : 0));
    shadingsys->attribute ("compile_report", debug|debug2);
    const char *opt_env = getenv ("TESTSHADE_OPT");  // overrides opt
    if (opt_env)
        shadingsys->attribute ("optimize", atoi(opt_env));
    else if (O0 || O1 || O2)
        shadingsys->attribute ("optimize", O2 ? 2 : (O1 ? 1 : 0));
    shadingsys->attribute ("debugnan", debugnan);

    // Loads a scene, creating camera, geometry and assigning shaders
    parse_scene();

    // validate options
    if (aa < 1) aa = 1;
    if (num_threads < 1)
        num_threads = boost::thread::hardware_concurrency();

    // prepare background importance table (if requested)
    if (backgroundResolution > 0 && backgroundShaderID >= 0) {
        // get a context so we can make several background shader calls
        OSL::PerThreadInfo *thread_info = shadingsys->create_thread_info();
        ShadingContext *ctx = shadingsys->get_context (thread_info);

        // build importance table to optimize background sampling
        background.prepare(backgroundResolution, eval_background, ctx);

        // release context
        shadingsys->release_context (ctx);
        shadingsys->destroy_thread_info(thread_info);
    } else {
        // we aren't directly evaluating the background
        backgroundResolution = 0;
    }

    double setuptime = timer.lap ();

    std::vector<Color3> pixels(xres * yres, Color3(0,0,0));

    // Create shared counter to iterate over one scanline at a time
    Counter scanline_counter(errhandler, yres, "Rendering");
    // launch a scanline worker for each thread
    boost::thread_group workers;
    for (int i = 0; i < num_threads; i++)
        workers.add_thread(new boost::thread(scanline_worker, boost::ref(scanline_counter), boost::ref(pixels)));
    workers.join_all();

    // Write image to disk
    ImageOutput* out = ImageOutput::create(imagefile);
    ImageSpec spec(xres, yres, 3, TypeDesc::HALF);
    if (out && out->open(imagefile, spec)) {
        out->write_image(TypeDesc::TypeFloat, &pixels[0]);
    } else {
        errhandler.error("Unable to write output image");
    }
    delete out;

    // Print some debugging info
    if (debug || stats) {
        double runtime = timer.lap();
        std::cout << "\n";
        std::cout << "Setup: " << OIIO::Strutil::timeintervalformat (setuptime,2) << "\n";
        std::cout << "Run  : " << OIIO::Strutil::timeintervalformat (runtime,2) << "\n";
        std::cout << "\n";
        std::cout << shadingsys->getstats (5) << "\n";
        OIIO::TextureSystem *texturesys = shadingsys->texturesys();
        if (texturesys)
            std::cout << texturesys->getstats (5) << "\n";
        std::cout << ustring::getstats() << "\n";
    }

    // We're done with the shading system now, destroy it
    shaders.clear ();  // Must release the group refs first
    delete shadingsys;

    return EXIT_SUCCESS;
}
