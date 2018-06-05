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
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/argparse.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/timer.h>
#include <OpenImageIO/filesystem.h>

#include <pugixml.hpp>

#ifdef USING_OIIO_PUGI
namespace pugi = OIIO::pugi;
#endif

#include <OSL/oslexec.h>

#include "optixrend.h"
#include "raytracer.h"
#include "shading.h"

#include <optix_world.h>


// The pre-compiled renderer support library LLVM bitcode is embedded into the
// testoptix executable and made available through these variables.
extern int rend_llvm_compiled_ops_size;
extern char rend_llvm_compiled_ops_block[];

using namespace OSL;

namespace { // anonymous namespace

ShadingSystem *shadingsys = NULL;
bool debug1 = false;
bool debug2 = false;
bool verbose = false;
bool runstats = false;
bool saveptx = false;
bool warmup = false;
int profile = 0;
bool O0 = false, O1 = false, O2 = false;
bool debugnan = false;
static std::string extraoptions;
int xres = 640, yres = 480, aa = 1, max_bounces = 1000000, rr_depth = 5;
int num_threads = 0;
ErrorHandler errhandler;
OptixRenderer rend;  // RendererServices
Camera camera;
Scene scene;
int backgroundShaderID = -1;
int backgroundResolution = 0;
std::vector<ShaderGroupRef> shaders;
std::string scenefile, imagefile;
static std::string shaderpath;

// NB: Unused parameters are left in place so that the parse_scene() from
//     testrender can be used as-is (and they will eventually be used, when
//     path tracing is added to testoptix)

optix::Context optix_ctx = NULL;

static std::string renderer_ptx;  // ray generation, etc.
static std::string wrapper_ptx;   // hit programs



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
    ap.options ("Usage:  testoptix [options] scenefile imagefile",
                "%*", get_filenames, "",
                "--help", &help, "Print help message",
                "-v", &verbose, "Verbose messages",
                "--debug", &debug1, "Lots of debugging info",
                "--debug2", &debug2, "Even more debugging info",
                "--runstats", &runstats, "Print run statistics",
                "--saveptx", &saveptx, "Save the generated PTX",
                "--warmup", &warmup, "Perform a warmup launch",
                "-r %d %d", &xres, &yres, "Render a WxH image",
                "-aa %d", &aa, "Trace NxN rays per pixel",
                "-t %d", &num_threads, "Render using N threads (default: auto-detect)",
                "-O0", &O0, "Do no runtime shader optimization",
                "-O1", &O1, "Do a little runtime shader optimization",
                "-O2", &O2, "Do lots of runtime shader optimization",
                "--debugnan", &debugnan, "Turn on 'debugnan' mode",
                "--path %s", &shaderpath, "Specify oso search path",
                "--options %s", &extraoptions, "Set extra OSL options",
                NULL);
    if (ap.parse(argc, argv) < 0) {
        std::cerr << ap.geterror() << std::endl;
        ap.usage ();
        exit (EXIT_FAILURE);
    }
    if (help) {
        std::cout <<
            "testoptix -- Simple test for OptiX functionality\n"
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
                    scene.spheres.emplace_back (center, radius, int(shaders.size()) - 1, is_light);
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
                scene.quads.emplace_back (co, ex, ey, int(shaders.size()) - 1, is_light);
            }
        } else if (strcmp(node.name(), "Background") == 0) {
            pugi::xml_attribute res_attr = node.attribute("resolution");
            if (res_attr)
                backgroundResolution = OIIO::Strutil::from_string<int>(res_attr.value());
            backgroundShaderID = int(shaders.size()) - 1;
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


// Copies the specified device buffer into an output vector, assuming that
// the buffer is in FLOAT3 format (and that Vec3 and float3 have the same
// underlying representation).
std::vector<OSL::Color3>
get_pixel_buffer (const std::string& buffer_name, int width, int height)
{
    const OSL::Color3* buffer_ptr =
        static_cast<OSL::Color3*>(optix_ctx[buffer_name]->getBuffer()->map());

    if (! buffer_ptr) {
        std::cerr << "Unable to map buffer " << buffer_name << std::endl;
        exit (EXIT_FAILURE);
    }

    std::vector<OSL::Color3> pixels;
    std::copy (&buffer_ptr[0], &buffer_ptr[width * height], back_inserter(pixels));

    optix_ctx[buffer_name]->getBuffer()->unmap();

    return pixels;
}


void load_ptx_from_file (std::string& ptx_string, const char* filename)
{
    if (! OIIO::Filesystem::read_text_file (filename, ptx_string)) {
        std::cerr << "Unable to load " << filename << std::endl;
        exit (EXIT_FAILURE);
    }
}


void init_optix_context ()
{
    // Set up the OptiX context
    optix_ctx = optix::Context::create();

    optix_ctx->setRayTypeCount (2);
    optix_ctx->setEntryPointCount (1);
    optix_ctx->setStackSize (2048);
    optix_ctx->setPrintEnabled (true);

    optix_ctx["radiance_ray_type"]->setUint  (0u);
    optix_ctx["shadow_ray_type"  ]->setUint  (1u);
    optix_ctx["bg_color"         ]->setFloat (0.0f, 0.0f, 0.0f);
    optix_ctx["bad_color"        ]->setFloat (1.0f, 0.0f, 1.0f);

    // Create the output buffer
    optix::Buffer buffer = optix_ctx->createBuffer (RT_BUFFER_OUTPUT,
                                                    RT_FORMAT_FLOAT3,
                                                    xres, yres);
    optix_ctx["output_buffer"]->set (buffer);

    // Load the renderer CUDA source and generate PTX for it
    std::string filename = std::string(PTX_PATH) + "/renderer.ptx";
    load_ptx_from_file (renderer_ptx, filename.c_str());

    // Create the OptiX programs and set them on the optix::Context
    optix_ctx->setRayGenerationProgram (0, optix_ctx->createProgramFromPTXString (renderer_ptx, "raygen"));
    optix_ctx->setMissProgram          (0, optix_ctx->createProgramFromPTXString (renderer_ptx, "miss"));
    optix_ctx->setExceptionProgram     (0, optix_ctx->createProgramFromPTXString (renderer_ptx, "exception"));

    // Load the PTX for the wrapper program. It will be used to create OptiX
    // Materials from the OSL ShaderGroups
    filename = std::string(PTX_PATH) + "/wrapper.ptx";
    load_ptx_from_file (wrapper_ptx, filename.c_str());

    // Load the PTX for the primitives
    std::string sphere_ptx;
    filename = std::string(PTX_PATH) + "/sphere.ptx";
    load_ptx_from_file (sphere_ptx, filename.c_str());

    std::string quad_ptx;
    filename = std::string(PTX_PATH) + "/quad.ptx";
    load_ptx_from_file (quad_ptx, filename.c_str());

    // Create the sphere and quad intersection programs, and save them on the
    // Scene so that they don't need to be regenerated for each primitive in the
    // scene
    scene.create_geom_programs (optix_ctx, sphere_ptx, quad_ptx);
}


void make_optix_materials ()
{
    optix::Program closest_hit = optix_ctx->createProgramFromPTXString(
        wrapper_ptx, "closest_hit_osl");

    optix::Program any_hit = optix_ctx->createProgramFromPTXString(
        wrapper_ptx, "any_hit_shadow");

    int mtl_id = 0;

    // Optimize each ShaderGroup in the scene, and use the resulting PTX to create
    // OptiX Programs which can be called by the closest hit program in the wrapper
    // to execute the compiled OSL shader.
    for (const auto& groupref : shaders) {
        shadingsys->optimize_group (groupref.get());

        std::string group_name, init_name, entry_name;

        shadingsys->getattribute (groupref.get(), "group_name",       OSL::TypeDesc::PTR, &group_name);
        shadingsys->getattribute (groupref.get(), "group_init_name",  OSL::TypeDesc::PTR, &init_name);
        shadingsys->getattribute (groupref.get(), "group_entry_name", OSL::TypeDesc::PTR, &entry_name);

        // Retrieve the compiled ShaderGroup PTX
        std::string osl_ptx;
        shadingsys->getattribute (groupref.get(), "ptx_compiled_version",
                                  OSL::TypeDesc::PTR, &osl_ptx);

        if (osl_ptx.empty()) {
            std::cerr << "Failed to generate PTX for ShaderGroup "
                      << group_name << std::endl;
            exit (EXIT_FAILURE);
        }

        if (saveptx) {
            std::ofstream out (group_name + "_" + std::to_string( mtl_id++ ) + ".ptx");
            out << osl_ptx;
            out.close();
        }

        // Create a new Material using the wrapper PTX
        optix::Material mtl = optix_ctx->createMaterial();
        mtl->setClosestHitProgram (0, closest_hit);
        mtl->setAnyHitProgram (1, any_hit);

        // Create Programs from the init and group_entry functions
        optix::Program osl_init = optix_ctx->createProgramFromPTXString (
            osl_ptx, init_name);

        optix::Program osl_group = optix_ctx->createProgramFromPTXString (
            osl_ptx, entry_name);

        // Set the OSL functions as Callable Programs so that they can be
        // executed by the closest hit program in the wrapper
        mtl["osl_init_func" ]->setProgramId (osl_init );
        mtl["osl_group_func"]->setProgramId (osl_group);

        scene.optix_mtls.push_back(mtl);
    }

    rend.update_string_table();
}


void finalize_scene ()
{
    // Create a GeometryGroup to contain the scene geometry
    optix::GeometryGroup geom_group = optix_ctx->createGeometryGroup();

    optix_ctx["top_object"  ]->set (geom_group);
    optix_ctx["top_shadower"]->set (geom_group);

    // NB: Since the scenes in the test suite consist of only a few primitives,
    //     using 'NoAccel' instead of 'Trbvh' might yield a slight performance
    //     improvement. For more complex scenes (e.g., scenes with meshes),
    //     using 'Trbvh' is recommended to achieve maximum performance.
    geom_group->setAcceleration (optix_ctx->createAcceleration ("Trbvh"));

    // Translate the primitives parsed from the scene description into OptiX scene
    // objects
    for (const auto& sphere : scene.spheres) {
        optix::Geometry sphere_geom = optix_ctx->createGeometry();
        sphere.setOptixVariables (sphere_geom, scene.sphere_bounds, scene.sphere_intersect);

        optix::GeometryInstance sphere_gi = optix_ctx->createGeometryInstance (
            sphere_geom, &scene.optix_mtls[sphere.shaderid()], &scene.optix_mtls[sphere.shaderid()]+1);

        geom_group->addChild (sphere_gi);
    }

    for (const auto& quad : scene.quads) {
        optix::Geometry quad_geom = optix_ctx->createGeometry();
        quad.setOptixVariables (quad_geom, scene.quad_bounds, scene.quad_intersect);

        optix::GeometryInstance quad_gi = optix_ctx->createGeometryInstance (
            quad_geom, &scene.optix_mtls[quad.shaderid()], &scene.optix_mtls[quad.shaderid()]+1);

        geom_group->addChild (quad_gi);
    }

    // Set the camera variables on the OptiX Context, to be used by the ray gen program
    optix_ctx["eye" ]->setFloat (vec3_to_float3 (camera.eye));
    optix_ctx["dir" ]->setFloat (vec3_to_float3 (camera.dir));
    optix_ctx["cx"  ]->setFloat (vec3_to_float3 (camera.cx));
    optix_ctx["cy"  ]->setFloat (vec3_to_float3 (camera.cy));
    optix_ctx["invw"]->setFloat (camera.invw);
    optix_ctx["invh"]->setFloat (camera.invh);

    optix_ctx->validate();
}

} // anonymous namespace


int main (int argc, const char *argv[])
{
    using namespace OIIO;
    Timer timer;

    // Read command line arguments
    getargs (argc, argv);

    shadingsys = new ShadingSystem (&rend, NULL, &errhandler);
    register_closures(shadingsys);

    shadingsys->attribute ("lockgeom",           1);
    shadingsys->attribute ("debug",              0);
    shadingsys->attribute ("optimize",           2);
    shadingsys->attribute ("opt_simplify_param", 1);
    shadingsys->attribute ("range_checking",     0);

    // Setup common attributes
    shadingsys->attribute ("debug", debug2 ? 2 : (debug1 ? 1 : 0));
    shadingsys->attribute ("compile_report", debug1|debug2);

    std::vector<char> lib_bitcode;
    std::copy (&rend_llvm_compiled_ops_block[0],
               &rend_llvm_compiled_ops_block[rend_llvm_compiled_ops_size],
               back_inserter(lib_bitcode));

    shadingsys->attribute ("lib_bitcode", OSL::TypeDesc::UINT8, &lib_bitcode);

    if (! shaderpath.empty())
        shadingsys->attribute ("searchpath:shader", shaderpath);
    else
        shadingsys->attribute ("searchpath:shader", OIIO::Filesystem::parent_path (scenefile));

    // Loads a scene, creating camera, geometry and assigning shaders
    parse_scene();

    // Set up the OptiX Context
    init_optix_context();

    // Set up the string table. This allocates a block of Unified Memory to hold
    // all of the static strings used by the OSL shaders. The strings can be
    // accessed via OptiX variables that hold pointers to the strings in the
    // table.
    rend.init_string_table(optix_ctx);

    // Convert the OSL ShaderGroups accumulated during scene parsing into
    // OptiX Materials
    make_optix_materials();

    // Set up the OptiX scene graph
    finalize_scene();

    double setuptime = timer.lap ();

    // Perform a tiny launch to warm up the OptiX context
    if (warmup)
        optix_ctx->launch (0, 1, 1);

    double warmuptime = timer.lap ();

    // Launch the GPU kernel to render the scene
    optix_ctx->launch (0, xres, yres);
    double runtime = timer.lap ();

    // Copy the output image from the device buffer
    std::vector<OSL::Color3> pixels = get_pixel_buffer ("output_buffer", xres, yres);

    // Make an ImageBuf that wraps it ('pixels' still owns the memory)
    ImageBuf pixelbuf (ImageSpec(xres, yres, 3, TypeDesc::FLOAT), pixels.data());
    pixelbuf.set_write_format (TypeDesc::HALF);

    // Write image to disk
    if (Strutil::iends_with (imagefile, ".jpg") ||
        Strutil::iends_with (imagefile, ".jpeg") ||
        Strutil::iends_with (imagefile, ".gif") ||
        Strutil::iends_with (imagefile, ".png")) {
        // JPEG, GIF, and PNG images should be automatically saved as sRGB
        // because they are almost certainly supposed to be displayed on web
        // pages.
        ImageBufAlgo::colorconvert (pixelbuf, pixelbuf,
                                    "linear", "sRGB", false, "", "");
    }
    pixelbuf.set_write_format (TypeDesc::HALF);
    if (! pixelbuf.write (imagefile))
        errhandler.error ("Unable to write output image: %s",
                          pixelbuf.geterror().c_str());

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
    }

    shaders.clear ();
    delete shadingsys;

    return EXIT_SUCCESS;
}
