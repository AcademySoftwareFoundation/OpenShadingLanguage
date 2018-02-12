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
#include <OpenImageIO/filesystem.h>

#include <OSL/oslexec.h>

#include <nvrtc.h>
#include <optix_world.h>

#include "raytracer.h"


using namespace OSL;

namespace { // anonymous namespace

int xres = 640;
int yres = 480;

// PTX strings
static std::string renderer_ptx;
static std::string sphere_ptx;
static std::string quad_ptx;

// Options strings
static std::string imagefile;
static std::string extraoptions;

bool optix_exceptions = false;

optix::Context optix_ctx = NULL;

// Containers for the scene geometry to be used when constructing the scene
std::vector<Sphere> spheres;
std::vector<Quad> quads;

Camera camera;


int get_filenames(int argc, const char *argv[])
{
    for (int i = 0; i < argc; i++) {
        if (imagefile.empty())
            imagefile = argv[i];
    }
    return 0;
}


void getargs(int argc, const char *argv[])
{
    bool help = false;
    OIIO::ArgParse ap;
    ap.options ("Usage:  testoptix [options] imagefile",
                "%*", get_filenames, "",
                "--help", &help, "Print help message",
                "--exceptions", &optix_exceptions, "Enable OptiX device exceptions and printing",
                "-r %d %d", &xres, &yres, "Render a WxH image",
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
    if (imagefile.empty()) {
        std::cerr << "testrender: Must specify a filename for output render\n";
        ap.usage();
        exit (EXIT_FAILURE);
    }
}


// Helper function to convert from Imath::Vec3 to optix::float3
optix::float3 vec3_to_float3 (const Vec3& vec)
{
    return optix::make_float3 (vec.x, vec.y, vec.z);
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


// Compiles a CUDA source file to PTX using NVRTC and returns it as a string.
void get_ptx_from_cu_file (std::string& ptx_string, const char* filename)
{
    // Read the CUDA source file into a std::string
    std::string cu_source;
    if (! OIIO::Filesystem::read_text_file (filename, cu_source)) {
        std::cerr << "Unable to load " << filename << std::endl;
        exit (EXIT_FAILURE);
    }

    // Set up the default NVRTC options
    std::vector<const char *> options;
    const std::string src_dir    = std::string("-I") + OIIO::Filesystem::parent_path (filename);
    const std::string cuda_inc   = std::string("-I") + CUDA_INCLUDE_DIR;
    const std::string optix_inc  = std::string("-I") + OPTIX_INCLUDE_DIR;
    const std::string optixu_inc = std::string("-I") + OPTIX_INCLUDE_DIR + "/optixu";

    options.push_back (src_dir.c_str());
    options.push_back (cuda_inc.c_str());
    options.push_back (optix_inc.c_str());
    options.push_back (optixu_inc.c_str());

    // NB: The 'gpu-architecture' option specifies the minimum compute
    //     capability required to execute the generated PTX and may need
    //     to be changed depending on the requirements of the input CUDA.
    options.push_back ("--gpu-architecture=compute_35");
    options.push_back ("--use_fast_math");
    options.push_back ("--device-as-default-execution-space");
    options.push_back ("--relocatable-device-code=true");
    options.push_back ("-D__x86_64");

    nvrtcProgram prog = 0;
    nvrtcResult result = nvrtcCreateProgram (&prog, cu_source.c_str(),
                                             filename, 0, nullptr, nullptr);
    if (result != NVRTC_SUCCESS) {
        std::cerr << "NVRTC program creation failed: "
                  << nvrtcGetErrorString (result) << std::endl;
        exit (EXIT_FAILURE);
    }

    // JIT compile CU to PTX
    result = nvrtcCompileProgram (prog, static_cast<int>(options.size()),
                                  options.data());

    if (result != NVRTC_SUCCESS) {
        std::cerr << "NVRTC Compilation failed: "
                  << nvrtcGetErrorString (result) << std::endl;

        size_t log_size = 0;
        nvrtcGetProgramLogSize(prog, &log_size);

        std::string log;
        log.resize (log_size);
        nvrtcGetProgramLog (prog, &log[0]);

        std::cerr << log << std::endl;
        exit (EXIT_FAILURE);
    }

    // Retrieve the PTX string
    size_t ptx_size = 0;
    result = nvrtcGetPTXSize (prog, &ptx_size);
    if (result != NVRTC_SUCCESS || ptx_size < 1) {
        std::cerr << "NVRTC PTX retrieval failed: "
                  << nvrtcGetErrorString (result) << std::endl;
        exit (EXIT_FAILURE);
    }

    ptx_string.resize (ptx_size);
    result = nvrtcGetPTX (prog, &ptx_string[0]);
    if (result != NVRTC_SUCCESS) {
        std::cerr << "NVRTC PTX retrieval failed: "
                  << nvrtcGetErrorString (result) << std::endl;
        exit (EXIT_FAILURE);
    }

    nvrtcDestroyProgram (&prog);
}


void init_optix_context ()
{
    // Set up the OptiX context
    optix_ctx = optix::Context::create();
    optix_ctx->setRayTypeCount (1);
    optix_ctx->setEntryPointCount (1);
    optix_ctx->setStackSize (2048);

    // OptiX device exceptions and printing are disabled by default, but they
    // can be enabled for debugging using the command-line option --exceptions
    if (optix_exceptions) {
        optix_ctx->setExceptionEnabled (RT_EXCEPTION_ALL, true);
        optix_ctx->setPrintEnabled (true);
    }

    optix_ctx["radiance_ray_type"]->setUint  (0u);
    optix_ctx["bg_color"         ]->setFloat (0.0f, 0.0f, 0.0f);
    optix_ctx["bad_color"        ]->setFloat (1.0f, 0.0f, 1.0f);

    // Create the output buffer
    optix::Buffer buffer = optix_ctx->createBuffer (RT_BUFFER_OUTPUT,
                                                    RT_FORMAT_FLOAT3,
                                                    xres, yres);
    optix_ctx["output_buffer"]->set (buffer);

    // Load the renderer CUDA source and generate PTX for it
    std::string filename = std::string(CUDA_SRC_PATH) + "/renderer.cu";
    get_ptx_from_cu_file (renderer_ptx, filename.c_str());

    // Create the OptiX programs and set them on the Context
    optix_ctx->setRayGenerationProgram (0, optix_ctx->createProgramFromPTXString (renderer_ptx, "raygen"));
    optix_ctx->setMissProgram          (0, optix_ctx->createProgramFromPTXString (renderer_ptx, "miss"));
    optix_ctx->setExceptionProgram     (0, optix_ctx->createProgramFromPTXString (renderer_ptx, "exception"));

    // Create the sphere intersection program
    filename = std::string(CUDA_SRC_PATH) + "/sphere.cu";
    get_ptx_from_cu_file (sphere_ptx, filename.c_str());

    // Create the quad intersection program
    filename = std::string(CUDA_SRC_PATH) + "/quad.cu";
    get_ptx_from_cu_file (quad_ptx, filename.c_str());
}


void setup_scene()
{
    // Set up the camera
    Vec3 eye(0.0f, 100.0f, 300.0f);
    Vec3 lookat(0.0f, 0.0f, 0.0f);
    Vec3 up(0.0f, 1.0f, 0.0f);
    camera = Camera(eye, lookat - eye, up, 70.0f, xres, yres);

    // Set the camera variables on the OptiX Context, to be used by the ray gen program
    optix_ctx["eye" ]->setFloat (vec3_to_float3 (camera.eye));
    optix_ctx["dir" ]->setFloat (vec3_to_float3 (camera.dir));
    optix_ctx["cx"  ]->setFloat (vec3_to_float3 (camera.cx));
    optix_ctx["cy"  ]->setFloat (vec3_to_float3 (camera.cy));
    optix_ctx["invw"]->setFloat (camera.invw);
    optix_ctx["invh"]->setFloat (camera.invh);

    // Specify the scene geometry. The objects would typically be parsed from a
    // scene file, but for this example they are hard-coded.
    quads.emplace_back (Vec3(-200.0f, 0.0f, 0.0f),
                        Vec3(400.0f, 0.0f, 0.0f),
                        Vec3(0.0f, 400.0f, 0.0f),
                        0, false);

    quads.emplace_back (Vec3(-200.0f, 0.0f, 0.0f),
                        Vec3(0.0f, 0.0f, 400.0f),
                        Vec3(400.0f, 0.0f, 0.0f),
                        0, false);

    spheres.emplace_back (Vec3(-60.0f, 30.0f, 120.0f), 30.0f, 0, false);
    spheres.emplace_back (Vec3(0.0f, 30.0f, 120.0f), 30.0f, 0, false);
    spheres.emplace_back (Vec3(60.0f, 30.0f, 120.0f), 30.0f, 0, false);

    // Make an OptiX material to be shared by all objects in the scene. In this
    // example the material is a simple normal shader.
    optix::Material optix_mtl = optix_ctx->createMaterial();
    optix_mtl->setClosestHitProgram (0, optix_ctx->createProgramFromPTXString (renderer_ptx, "closest_hit"));

    // Create the bounding and intersection programs needed for acceleration
    // structure building and ray traversal
    optix::Program sphere_bounds = optix_ctx->createProgramFromPTXString (sphere_ptx, "bounds");
    optix::Program sphere_intersect = optix_ctx->createProgramFromPTXString (sphere_ptx, "intersect");
    optix::Program quad_bounds = optix_ctx->createProgramFromPTXString (quad_ptx, "bounds");
    optix::Program quad_intersect = optix_ctx->createProgramFromPTXString (quad_ptx, "intersect");

    // Create a GeometryGroup to contain the individual scene primitives
    optix::GeometryGroup geom_group = optix_ctx->createGeometryGroup();
    geom_group->setAcceleration (optix_ctx->createAcceleration ("Trbvh"));
    optix_ctx["top_object"  ]->set (geom_group);
    optix_ctx["top_shadower"]->set (geom_group);

    // Convert each sphere to an OptiX scene object and add it to the top-level
    // GeometryGroup
    for (const auto& s : spheres) {
        optix::Geometry sphere_geom = optix_ctx->createGeometry();
        sphere_geom["sphere"]->setFloat (optix::make_float4(s.c.x, s.c.y, s.c.z, sqrtf(s.r2)));
        sphere_geom["r2"]->setFloat (s.r2);

        sphere_geom->setPrimitiveCount (1u);
        sphere_geom->setBoundingBoxProgram (sphere_bounds);
        sphere_geom->setIntersectionProgram (sphere_intersect);

        optix::GeometryInstance sphere_gi =
            optix_ctx->createGeometryInstance (sphere_geom, &optix_mtl, &optix_mtl+1);

        geom_group->addChild (sphere_gi);
    }

    // Convert each quad to an OptiX scene object and add it to the top-level
    // GeometryGroup
    for (const auto& q : quads) {
        optix::Geometry quad_geom = optix_ctx->createGeometry();
        quad_geom["p" ]->setFloat (vec3_to_float3 (q.p));
        quad_geom["ex"]->setFloat (vec3_to_float3 (q.ex));
        quad_geom["ey"]->setFloat (vec3_to_float3 (q.ey));
        quad_geom["n" ]->setFloat (vec3_to_float3 (q.n));
        quad_geom["eu"]->setFloat (q.eu);
        quad_geom["ev"]->setFloat (q.ev);

        quad_geom->setPrimitiveCount (1u);
        quad_geom->setBoundingBoxProgram (quad_bounds);
        quad_geom->setIntersectionProgram (quad_intersect);

        optix::GeometryInstance quad_gi =
            optix_ctx->createGeometryInstance (quad_geom, &optix_mtl, &optix_mtl+1);

        geom_group->addChild (quad_gi);
    }

    optix_ctx->validate();
}


} // anonymous namespace

int main (int argc, const char *argv[])
{
    using namespace OIIO;

    // Read command line arguments
    getargs (argc, argv);

    // Set up the OptiX Context
    init_optix_context();

    // Construct a scene similar to those in the testsuite,
    // as a proxy for loading a scene from file.
    setup_scene();

    // Launch the GPU kernel to render the scene
    optix_ctx->launch (0, xres, yres);

    // Copy the output image from the device buffer
    std::vector<OSL::Color3> pixels = get_pixel_buffer ("output_buffer", xres, yres);

    // Write the image to disk
    ImageBuf pixelbuf(ImageSpec(xres, yres, 3, TypeDesc::FLOAT), pixels.data());
    pixelbuf.set_write_format (TypeDesc::HALF);
    if (! pixelbuf.write (imagefile)) {
        std::cerr << "Unable to write image" << std::endl;
        exit (EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}
