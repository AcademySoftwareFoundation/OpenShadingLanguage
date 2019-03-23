/*
Copyright (c) 2009-2018 Sony Pictures Imageworks Inc., et al.
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


#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/argparse.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/timer.h>
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/thread.h>
#include <OpenImageIO/parallel.h>

#include <pugixml.hpp>

#ifdef USING_OIIO_PUGI
namespace pugi = OIIO::pugi;
#endif

#include <OSL/oslexec.h>
#include <OSL/optix_compat.h>
#include "optixrend.h"
#include "raytracer.h"
#include "sampling.h"
#include "shading.h"
#include "simplerend.h"
#include "util.h"


// The pre-compiled renderer support library LLVM bitcode is embedded into the
// testoptix executable and made available through these variables.
extern int rend_llvm_compiled_ops_size;
extern char rend_llvm_compiled_ops_block[];

using namespace OSL;

namespace { // anonymous namespace

static ShadingSystem *shadingsys = NULL;
static bool debug1 = false;
static bool debug2 = false;
static bool verbose = false;
static bool runstats = false;
static bool saveptx = false;
static bool warmup = false;
static bool profile = false;
static bool O0 = false, O1 = false, O2 = false;
static bool debugnan = false;
static bool debug_uninit = false;
static bool userdata_isconnected = false;
static std::string extraoptions;
static std::string texoptions;
static int xres = 640, yres = 480;
static int aa = 1, max_bounces = 1000000, rr_depth = 5;
static int num_threads = 0;
static int iters = 1;
static ErrorHandler errhandler;
static SimpleRenderer *rend = nullptr;
//static OptixRenderer *optixrend = nullptr;
static std::string scenefile, imagefile;
static std::string shaderpath;
static bool shadingsys_options_set = false;
static bool use_optix = false;




// Set shading system global attributes based on command line options.
static void
set_shadingsys_options ()
{
    shadingsys->attribute ("debug", debug2 ? 2 : (debug1 ? 1 : 0));
    shadingsys->attribute ("compile_report", debug1|debug2);
    int opt = 2;  // default
    if (O0) opt = 0;
    if (O1) opt = 1;
    if (O2) opt = 2;
    if (const char *opt_env = getenv ("TESTSHADE_OPT"))  // overrides opt
        opt = atoi(opt_env);
    shadingsys->attribute ("optimize", opt);
    shadingsys->attribute ("profile", int(profile));
    shadingsys->attribute ("lockgeom", 1);
    shadingsys->attribute ("debug_nan", debugnan);
    shadingsys->attribute ("debug_uninit", debug_uninit);
    shadingsys->attribute ("userdata_isconnected", userdata_isconnected);
    if (! shaderpath.empty())
        shadingsys->attribute ("searchpath:shader", shaderpath);
    if (extraoptions.size())
        shadingsys->attribute ("options", extraoptions);
    if (texoptions.size())
        shadingsys->texturesys()->attribute ("options", texoptions);
    shadingsys_options_set = true;
}



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
    OIIO::ArgParse ap;
    ap.options ("Usage:  testrender [options] scene.xml outputfilename",
                "%*", get_filenames, "",
                "--help", &help, "Print help message",
                "-v", &verbose, "Verbose messages",
                "--optix", &use_optix, "Use OptiX if available",
                "--debug", &debug1, "Lots of debugging info",
                "--debug2", &debug2, "Even more debugging info",
                "--runstats", &runstats, "Print run statistics",
                "--stats", &runstats, "", // DEPRECATED 1.7
                "--profile", &profile, "Print profile information",
                "--saveptx", &saveptx, "Save the generated PTX",
                "--warmup", &warmup, "Perform a warmup launch",
                "--res %d %d", &xres, &yres, "Make an W x H image",
                "-r %d %d", &xres, &yres, "", // synonym for -res
                "-aa %d", &aa, "Trace NxN rays per pixel",
                "-t %d", &num_threads, "Render using N threads (default: auto-detect)",
                "--iters %d", &iters, "Number of iterations",
                "-O0", &O0, "Do no runtime shader optimization",
                "-O1", &O1, "Do a little runtime shader optimization",
                "-O2", &O2, "Do lots of runtime shader optimization",
                "--debugnan", &debugnan, "Turn on 'debugnan' mode",
                "--path %s", &shaderpath, "Specify oso search path",
                "--options %s", &extraoptions, "Set extra OSL options",
                "--texoptions %s", &texoptions, "Set extra TextureSystem options",
                NULL);
    if (ap.parse(argc, argv) < 0) {
        std::cerr << ap.geterror() << std::endl;
        ap.usage ();
        exit (EXIT_FAILURE);
    }
    if (help) {
        std::cout <<
            "testrender -- Test Renderer for Open Shading Language\n"
             OSL_COPYRIGHT_STRING "\n";
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
    if (debug1 || verbose)
        errhandler.verbosity (ErrorHandler::VERBOSE);
}

Vec3 strtovec(string_view str) {
    Vec3 v(0, 0, 0);
    OIIO::Strutil::parse_float (str, v[0]);
    OIIO::Strutil::parse_char (str, ',');
    OIIO::Strutil::parse_float (str, v[1]);
    OIIO::Strutil::parse_char (str, ',');
    OIIO::Strutil::parse_float (str, v[2]);
    return v;
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



void
parse_scene(SimpleRenderer* rend, Camera &camera, Scene& scene,
            std::vector<ShaderGroupRef>& shaders)
{
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
            if (fov_attr) fov = OIIO::Strutil::from_string<float>(fov_attr.value());

            // create actual camera
            camera = Camera(eye, dir, up, fov, xres, yres);
        } else if (strcmp(node.name(), "Sphere") == 0) {
            // load sphere
            pugi::xml_attribute center_attr = node.attribute("center");
            pugi::xml_attribute radius_attr = node.attribute("radius");
            if (center_attr && radius_attr) {
                Vec3  center = strtovec(center_attr.value());
                float radius = OIIO::Strutil::from_string<float>(radius_attr.value());
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
                rend->backgroundResolution = OIIO::Strutil::from_string<int>(res_attr.value());
            rend->backgroundShaderID = int(shaders.size()) - 1;
        } else if (strcmp(node.name(), "ShaderGroup") == 0) {
            ShaderGroupRef group;
            pugi::xml_attribute name_attr = node.attribute("name");
            std::string name = name_attr? name_attr.value() : "group";
            pugi::xml_attribute type_attr = node.attribute("type");
            std::string shadertype = type_attr ? type_attr.value() : "surface";
            pugi::xml_attribute commands_attr = node.attribute("commands");
            std::string commands = commands_attr ? commands_attr.value() : node.text().get();
            if (commands.size())
                group = shadingsys->ShaderGroupBegin (name, shadertype, commands);
            else
                group = shadingsys->ShaderGroupBegin (name);
            ParamStorage<1024> store; // scratch space to hold parameters until they are read by Shader()
            for (pugi::xml_node gnode = node.first_child(); gnode; gnode = gnode.next_sibling()) {
                if (strcmp(gnode.name(), "Parameter") == 0) {
                    // handle parameters
                    for (pugi::xml_attribute attr = gnode.first_attribute(); attr; attr = attr.next_attribute()) {
                        int i = 0; float x = 0, y = 0, z = 0;
                        if (sscanf(attr.value(), " int %d ", &i) == 1)
                            shadingsys->Parameter(*group, attr.name(),
                                                  TypeDesc::TypeInt, store.Int(i));
                        else if (sscanf(attr.value(), " float %f ", &x) == 1)
                            shadingsys->Parameter(*group, attr.name(),
                                                  TypeDesc::TypeFloat, store.Float(x));
                        else if (sscanf(attr.value(), " vector %f %f %f", &x, &y, &z) == 3)
                            shadingsys->Parameter(*group, attr.name(),
                                                  TypeDesc::TypeVector, store.Vec(x, y, z));
                        else if (sscanf(attr.value(), " point %f %f %f", &x, &y, &z) == 3)
                            shadingsys->Parameter(*group, attr.name(),
                                                  TypeDesc::TypePoint, store.Vec(x, y, z));
                        else if (sscanf(attr.value(), " color %f %f %f", &x, &y, &z) == 3)
                            shadingsys->Parameter(*group, attr.name(),
                                                  TypeDesc::TypeColor, store.Vec(x, y, z));
                        else
                            shadingsys->Parameter(*group, attr.name(),
                                                  TypeDesc::TypeString, store.Str(attr.value()));
                    }
                } else if (strcmp(gnode.name(), "Shader") == 0) {
                    pugi::xml_attribute  type_attr = gnode.attribute("type");
                    pugi::xml_attribute  name_attr = gnode.attribute("name");
                    pugi::xml_attribute layer_attr = gnode.attribute("layer");
                    const char* type = type_attr ? type_attr.value() : "surface";
                    if (name_attr && layer_attr)
                        shadingsys->Shader(*group, type, name_attr.value(),
                                           layer_attr.value());
                } else if (strcmp(gnode.name(), "ConnectShaders") == 0) {
                    // FIXME: find a more elegant way to encode this
                    pugi::xml_attribute  sl = gnode.attribute("srclayer");
                    pugi::xml_attribute  sp = gnode.attribute("srcparam");
                    pugi::xml_attribute  dl = gnode.attribute("dstlayer");
                    pugi::xml_attribute  dp = gnode.attribute("dstparam");
                    if (sl && sp && dl && dp)
                        shadingsys->ConnectShaders(*group,
                                                   sl.value(), sp.value(),
                                                   dl.value(), dp.value());
                } else {
                    // unknow element?
                }
            }
            shadingsys->ShaderGroupEnd(*group);
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




} // anonymous namespace



int
main (int argc, const char *argv[])
{
    try {
        using namespace OIIO;
        Timer timer;

        // Read command line arguments
        getargs (argc, argv);

        if (use_optix)
            rend = new OptixRenderer;
        else
            rend = new SimpleRenderer;

        // Create a new shading system.  We pass it the RendererServices
        // object that services callbacks from the shading system, NULL for
        // the TextureSystem (which will create a default OIIO one), and
        // an error handler.
        shadingsys = new ShadingSystem (rend, NULL, &errhandler);
        rend->shadingsys = shadingsys;

        // Other renderer and global options
        rend->attribute("saveptx", (int)saveptx);
        rend->attribute("max_bounces", max_bounces);
        rend->attribute("rr_depth", rr_depth);
        rend->attribute("aa", aa);
        OIIO::attribute("threads", num_threads);

        // Register the layout of all closures known to this renderer
        // Any closure used by the shader which is not registered, or
        // registered with a different number of arguments will lead
        // to a runtime error.
        register_closures(shadingsys);

        // Setup common attributes
        set_shadingsys_options();

        // Loads a scene, creating camera, geometry and assigning shaders
        parse_scene(rend, rend->camera, rend->scene, rend->shaders);

        rend->prepare_render ();

        rend->pixelbuf.reset (ImageSpec(xres, yres, 3, TypeDesc::FLOAT));

        double setuptime = timer.lap ();

        if (warmup)
            rend->warmup();
        double warmuptime = timer.lap ();

        // Launch the kernel to render the scene
        for (int i = 0; i < iters; ++i)
            rend->render (xres, yres);
        double runtime = timer.lap ();

        rend->finalize_pixel_buffer ();

        // Write image to disk
        if (Strutil::iends_with (imagefile, ".jpg") ||
            Strutil::iends_with (imagefile, ".jpeg") ||
            Strutil::iends_with (imagefile, ".gif") ||
            Strutil::iends_with (imagefile, ".png")) {
            // JPEG, GIF, and PNG images should be automatically saved as sRGB
            // because they are almost certainly supposed to be displayed on web
            // pages.
            ImageBufAlgo::colorconvert (rend->pixelbuf, rend->pixelbuf,
                                        "linear", "sRGB", false, "", "");
        }
        rend->pixelbuf.set_write_format (TypeDesc::HALF);
        if (! rend->pixelbuf.write (imagefile))
            errhandler.error ("Unable to write output image: %s",
                              rend->pixelbuf.geterror());

        // Print some debugging info
        if (debug1 || runstats || profile) {
            double writetime = timer.lap();
            std::cout << "\n";
            std::cout << "Setup : " << OIIO::Strutil::timeintervalformat (setuptime,4) << "\n";
            if (warmup) {
                std::cout << "Warmup: " << OIIO::Strutil::timeintervalformat (warmuptime,4) << "\n";
            }
            std::cout << "Run   : " << OIIO::Strutil::timeintervalformat (runtime,4) << "\n";
            std::cout << "Write : " << OIIO::Strutil::timeintervalformat (writetime,4) << "\n";
            std::cout << "\n";
            std::cout << shadingsys->getstats (5) << "\n";
            OIIO::TextureSystem *texturesys = shadingsys->texturesys();
            if (texturesys)
                std::cout << texturesys->getstats (5) << "\n";
            std::cout << ustring::getstats() << "\n";
        }

        // We're done with the shading system now, destroy it
        rend->shaders.clear ();  // Must release the group refs first
        delete shadingsys;
        delete rend;
#ifdef OSL_USE_OPTIX
    } catch (const optix::Exception& e) {
        printf("Optix Error: %s\n", e.what());
#endif
    } catch (const std::exception& e) {
        printf("Unknown Error: %s\n", e.what());
    }

    return EXIT_SUCCESS;
}
