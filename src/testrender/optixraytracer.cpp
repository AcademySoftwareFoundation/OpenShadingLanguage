/*
Copyright (c) 2019 Sony Pictures Imageworks Inc., et al.
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
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOTSS
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


#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/sysutil.h>

#include <OSL/oslconfig.h>

#include "optixraytracer.h"
#include "../liboslexec/opcolor.h"

// The pre-compiled renderer support library LLVM bitcode is embedded
// into the executable and made available through these variables.
extern int rend_llvm_compiled_ops_size;
extern unsigned char rend_llvm_compiled_ops_block[];


OSL_NAMESPACE_ENTER



OptixRaytracer::~OptixRaytracer ()
{
#ifdef OSL_USE_OPTIX
    m_str_table.freetable();
    if (m_optix_ctx)
        m_optix_ctx->destroy();
#endif
}



std::string
OptixRaytracer::load_ptx_file (string_view filename)
{
#ifdef OSL_USE_OPTIX
    std::vector<std::string> paths = {
        OIIO::Filesystem::parent_path(OIIO::Sysutil::this_program_path()),
        PTX_PATH
    };
    std::string filepath = OIIO::Filesystem::searchpath_find (filename, paths,
                                                              false);
    if (OIIO::Filesystem::exists(filepath)) {
        std::string ptx_string;
        if (OIIO::Filesystem::read_text_file (filepath, ptx_string))
            return ptx_string;
    }
#endif
    errhandler().severe ("Unable to load %s", filename);
    return {};
}



bool
OptixRaytracer::init_optix_context (int xres, int yres)
{
#ifdef OSL_USE_OPTIX
    shadingsys->attribute ("lib_bitcode", {OSL::TypeDesc::UINT8, rend_llvm_compiled_ops_size},
                           rend_llvm_compiled_ops_block);

    // Set up the OptiX context
    m_optix_ctx = optix::Context::create();

    // Set up the string table. This allocates a block of CUDA device memory to
    // hold all of the static strings used by the OSL shaders. The strings can
    // be accessed via OptiX variables that hold pointers to the table entries.
    m_str_table.init(m_optix_ctx);

    if (m_optix_ctx->getEnabledDeviceCount() != 1)
        errhandler().warning ("Only one CUDA device is currently supported");

    m_optix_ctx->setRayTypeCount (2);
    m_optix_ctx->setEntryPointCount (1);
    m_optix_ctx->setStackSize (2048);
    m_optix_ctx->setPrintEnabled (true);

    // Load the renderer CUDA source and generate PTX for it
    std::string progName = "optix_raytracer.ptx";
    std::string renderer_ptx = load_ptx_file(progName);
    if (renderer_ptx.empty()) {
        errhandler().severe ("Could not find PTX for the raygen program");
        return false;
    }

    // Create the OptiX programs and set them on the optix::Context
    m_program = m_optix_ctx->createProgramFromPTXString(renderer_ptx, "raygen");
    m_optix_ctx->setRayGenerationProgram(0, m_program);

    if (scene.num_prims()) {
        m_optix_ctx["radiance_ray_type"]->setUint  (0u);
        m_optix_ctx["shadow_ray_type"  ]->setUint  (1u);
        m_optix_ctx["bg_color"         ]->setFloat (0.0f, 0.0f, 0.0f);
        m_optix_ctx["bad_color"        ]->setFloat (1.0f, 0.0f, 1.0f);

        // Create the OptiX programs and set them on the optix::Context
        if (renderer_ptx.size()) {
            m_optix_ctx->setMissProgram (0, m_optix_ctx->createProgramFromPTXString (renderer_ptx, "miss"));
            m_optix_ctx->setExceptionProgram (0, m_optix_ctx->createProgramFromPTXString (renderer_ptx, "exception"));
        }

        // Load the PTX for the wrapper program. It will be used to
        // create OptiX Materials from the OSL ShaderGroups
        m_materials_ptx = load_ptx_file ("wrapper.ptx");
        if (m_materials_ptx.empty())
            return false;

        // Load the PTX for the primitives
        std::string sphere_ptx = load_ptx_file ("sphere.ptx");
        std::string quad_ptx = load_ptx_file ("quad.ptx");
        if (sphere_ptx.empty() || quad_ptx.empty())
            return false;

        // Create the sphere and quad intersection programs.
        sphere_bounds    = m_optix_ctx->createProgramFromPTXString (sphere_ptx, "bounds");
        quad_bounds      = m_optix_ctx->createProgramFromPTXString (quad_ptx,   "bounds");
        sphere_intersect = m_optix_ctx->createProgramFromPTXString (sphere_ptx, "intersect");
        quad_intersect   = m_optix_ctx->createProgramFromPTXString (quad_ptx,   "intersect");
    }
#endif
    return true;
}



bool
OptixRaytracer::synch_attributes ()
{
#ifdef OSL_USE_OPTIX
    // FIXME -- this is for testing only
    // Make some device strings to test userdata parameters
    uint64_t addr1 = register_string ("ud_str_1", "");
    uint64_t addr2 = register_string ("userdata string", "");
    m_optix_ctx["test_str_1"]->setUserData (sizeof(char*), &addr1);
    m_optix_ctx["test_str_2"]->setUserData (sizeof(char*), &addr2);

    {
        const char* name = OSL_NAMESPACE_STRING "::pvt::s_color_system";

        pvt::ColorSystem* colorSys = nullptr;
        shadingsys->getattribute("colorsystem", TypeDesc::PTR, (void*)&colorSys);
        if (colorSys == nullptr) {
            errhandler().error ("No colorsystem available.");
            return false;
        }

        // Get the size data-size, minus the ustring size
        const size_t dataSize = sizeof(pvt::ColorSystem) - sizeof(StringParam);

        optix::Buffer buffer = m_optix_ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
        if (!buffer) {
            errhandler().error ("Could not create buffer for '%s'.", name);
            return false;
        }

        // set the elemet size to char
        buffer->setElementSize(sizeof(char));

        // and number of elements to the actual size needed.
        buffer->setSize(dataSize + sizeof(DeviceString));

        // copy the base data
        char* dstData = (char*) buffer->map();
        if (!dstData) {
            errhandler().error ("Could not map buffer for '%s' (size: %lu).",
                                 name, dataSize);
            return false;
        }
        ::memcpy(dstData, colorSys, dataSize);

        // convert the ustring to a device string
        uint64_t devStr = register_string (colorSys->colorspace().string(), "");

        // FIXME -- Should probably handle alignment better.
        // then copy the device string to the end
        ::memcpy(dstData+dataSize, &devStr, sizeof(devStr));

        buffer->unmap();
        m_optix_ctx[name]->setBuffer(buffer);
    }
#endif
    return true;
}



bool
OptixRaytracer::make_optix_materials ()
{
#ifdef OSL_USE_OPTIX
    optix::Program closest_hit, any_hit;
    if (scene.num_prims()) {
        closest_hit = m_optix_ctx->createProgramFromPTXString(
            m_materials_ptx, "closest_hit_osl");
        any_hit = m_optix_ctx->createProgramFromPTXString(
            m_materials_ptx, "any_hit_shadow");
    }

    // Stand-in: names of shader outputs to preserve
    // FIXME
    std::vector<const char*> outputs { "Cout" };

    // Optimize each ShaderGroup in the scene, and use the resulting
    // PTX to create OptiX Programs which can be called by the closest
    // hit program in the wrapper to execute the compiled OSL shader.
    int mtl_id = 0;
    for (const auto& groupref : shaders()) {
        shadingsys->attribute (groupref.get(), "renderer_outputs",
                               TypeDesc(TypeDesc::STRING, outputs.size()),
                               outputs.data());

        shadingsys->optimize_group (groupref.get(), nullptr);

        if (!scene.num_prims()) {
            if (!shadingsys->find_symbol (*groupref.get(), ustring(outputs[0]))) {
                errhandler().warning ("Requested output '%s', which wasn't found",
                                      outputs[0]);
            }
        }

        std::string group_name, init_name, entry_name;
        shadingsys->getattribute (groupref.get(), "groupname",        group_name);
        shadingsys->getattribute (groupref.get(), "group_init_name",  init_name);
        shadingsys->getattribute (groupref.get(), "group_entry_name", entry_name);

        // Retrieve the compiled ShaderGroup PTX
        std::string osl_ptx;
        shadingsys->getattribute (groupref.get(), "ptx_compiled_version",
                                  OSL::TypeDesc::PTR, &osl_ptx);

        if (osl_ptx.empty()) {
            errhandler().error ("Failed to generate PTX for ShaderGroup %s",
                                group_name);
            return false;
        }

        if (options.get_int("saveptx")) {
            std::ofstream out (group_name + "_" + std::to_string(mtl_id++) + ".ptx");
            out << osl_ptx;
            out.close();
        }

        // Create Programs from the init and group_entry functions,
        // and set the OSL functions as Callable Programs so that they
        // can be executed by the closest hit program in the wrapper
        optix::Program osl_init = m_optix_ctx->createProgramFromPTXString (
            osl_ptx, init_name);
        optix::Program osl_group = m_optix_ctx->createProgramFromPTXString (
            osl_ptx, entry_name);
        if (scene.num_prims()) {
            // Create a new Material using the wrapper PTX
            optix::Material mtl = m_optix_ctx->createMaterial();
            mtl->setClosestHitProgram (0, closest_hit);
            mtl->setAnyHitProgram (1, any_hit);

            // Set the OSL functions as Callable Programs so that they can be
            // executed by the closest hit program in the wrapper
            mtl["osl_init_func" ]->setProgramId (osl_init );
            mtl["osl_group_func"]->setProgramId (osl_group);
            scene.optix_mtls.push_back(mtl);
        } else {
            // Grid shading
            m_program["osl_init_func" ]->setProgramId (osl_init );
            m_program["osl_group_func"]->setProgramId (osl_group);
        }
    }
    if (!synch_attributes())
        return false;
#endif
    return true;
}



bool
OptixRaytracer::finalize_scene()
{
#ifdef OSL_USE_OPTIX
    make_optix_materials();

    // Create a GeometryGroup to contain the scene geometry
    optix::GeometryGroup geom_group = m_optix_ctx->createGeometryGroup();

    m_optix_ctx["top_object"  ]->set (geom_group);
    m_optix_ctx["top_shadower"]->set (geom_group);

    // NB: Since the scenes in the test suite consist of only a few primitives,
    //     using 'NoAccel' instead of 'Trbvh' might yield a slight performance
    //     improvement. For more complex scenes (e.g., scenes with meshes),
    //     using 'Trbvh' is recommended to achieve maximum performance.
    geom_group->setAcceleration (m_optix_ctx->createAcceleration ("Trbvh"));

    // Translate the primitives parsed from the scene description into OptiX scene
    // objects
    for (const auto& sphere : scene.spheres) {
        optix::Geometry sphere_geom = m_optix_ctx->createGeometry();
        sphere.setOptixVariables (sphere_geom, sphere_bounds, sphere_intersect);

        optix::GeometryInstance sphere_gi = m_optix_ctx->createGeometryInstance (
            sphere_geom, &scene.optix_mtls[sphere.shaderid()], &scene.optix_mtls[sphere.shaderid()]+1);

        geom_group->addChild (sphere_gi);
    }

    for (const auto& quad : scene.quads) {
        optix::Geometry quad_geom = m_optix_ctx->createGeometry();
        quad.setOptixVariables (quad_geom, quad_bounds, quad_intersect);

        optix::GeometryInstance quad_gi = m_optix_ctx->createGeometryInstance (
            quad_geom, &scene.optix_mtls[quad.shaderid()], &scene.optix_mtls[quad.shaderid()]+1);

        geom_group->addChild (quad_gi);
    }

    // Set the camera variables on the OptiX Context, to be used by the ray gen program
    m_optix_ctx["eye" ]->setFloat (vec3_to_float3 (camera.eye));
    m_optix_ctx["dir" ]->setFloat (vec3_to_float3 (camera.dir));
    m_optix_ctx["cx"  ]->setFloat (vec3_to_float3 (camera.cx));
    m_optix_ctx["cy"  ]->setFloat (vec3_to_float3 (camera.cy));
    m_optix_ctx["invw"]->setFloat (camera.invw);
    m_optix_ctx["invh"]->setFloat (camera.invh);

    // Create the output buffer
    optix::Buffer buffer = m_optix_ctx->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, camera.xres, camera.yres);
    m_optix_ctx["output_buffer"]->set(buffer);

    m_optix_ctx->validate();
#endif
    return true;
}



/// Return true if the texture handle (previously returned by
/// get_texture_handle()) is a valid texture that can be subsequently
/// read or sampled.
bool
OptixRaytracer::good(TextureHandle *handle)
{
#ifdef OSL_USE_OPTIX
    return intptr_t(handle) != RT_TEXTURE_ID_NULL;
#else
    return false;
#endif
}



/// Given the name of a texture, return an opaque handle that can be
/// used with texture calls to avoid the name lookups.
RendererServices::TextureHandle*
OptixRaytracer::get_texture_handle (ustring filename, ShadingContext* shading_context)
{
#ifdef OSL_USE_OPTIX
    auto itr = m_samplers.find(filename);
    if (itr == m_samplers.end()) {
        optix::TextureSampler sampler = context()->createTextureSampler();
        sampler->setWrapMode(0, RT_WRAP_REPEAT);
        sampler->setWrapMode(1, RT_WRAP_REPEAT);
        sampler->setWrapMode(2, RT_WRAP_REPEAT);

        sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
        sampler->setIndexingMode(false ? RT_TEXTURE_INDEX_ARRAY_INDEX : RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        sampler->setMaxAnisotropy(1.0f);


        OIIO::ImageBuf image;
        if (!image.init_spec(filename, 0, 0)) {
            errhandler().error ("Could not load: %s", filename);
            return (TextureHandle*)(intptr_t(RT_TEXTURE_ID_NULL));
        }
        int nchan = image.spec().nchannels;

        OIIO::ROI roi = OIIO::get_roi_full(image.spec());
        int width = roi.width(), height = roi.height();
        std::vector<float> pixels(width * height * nchan);
        image.get_pixels(roi, OIIO::TypeDesc::FLOAT, pixels.data());

        optix::Buffer buffer = context()->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, width, height);

        float* device_ptr = static_cast<float*>(buffer->map());
        unsigned int pixel_idx = 0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                memcpy(device_ptr, &pixels[pixel_idx], sizeof(float) * nchan);
                device_ptr += 4;
                pixel_idx += nchan;
            }
        }
        buffer->unmap();
        sampler->setBuffer(buffer);
        itr = m_samplers.emplace(std::move(filename), std::move(sampler)).first;

    }
    return (RendererServices::TextureHandle*) intptr_t(itr->second->getId());
#else
    return nullptr;
#endif
}



void
OptixRaytracer::prepare_render()
{
#ifdef OSL_USE_OPTIX
    // Set up the OptiX Context
    init_optix_context(camera.xres, camera.yres);

    // Set up the OptiX scene graph
    finalize_scene ();
#endif
}



void
OptixRaytracer::warmup()
{
#ifdef OSL_USE_OPTIX
    // Perform a tiny launch to warm up the OptiX context
    m_optix_ctx->launch (0, 1, 1);
#endif
}



void
OptixRaytracer::render(int xres, int yres)
{
#ifdef OSL_USE_OPTIX
    m_optix_ctx->launch (0, xres, yres);
#endif
}



void
OptixRaytracer::finalize_pixel_buffer ()
{
#ifdef OSL_USE_OPTIX
    std::string buffer_name = "output_buffer";
    const void* buffer_ptr = m_optix_ctx[buffer_name]->getBuffer()->map();
    if (! buffer_ptr)
        errhandler().severe ("Unable to map buffer %s", buffer_name);
    pixelbuf.set_pixels (OIIO::ROI::All(), OIIO::TypeFloat, buffer_ptr);
#endif
}



void
OptixRaytracer::clear()
{
    shaders().clear();
#ifdef OSL_USE_OPTIX
    if (m_optix_ctx) {
        m_optix_ctx->destroy();
        m_optix_ctx = nullptr;
    }
#endif
}

OSL_NAMESPACE_EXIT
