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
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

#include "optixrend.h"
#include "raytracer.h"

#include "../liboslexec/splineimpl.h"

OSL_NAMESPACE_ENTER

bool loadPtxFromFile (const std::string& progName, std::string& ptx_string)
{
    std::string filepath = std::string(PTX_PATH) + "/" + progName;
    if (OIIO::Filesystem::read_text_file (filepath, ptx_string))
        return true;

    std::cerr << "Unable to load '" << filepath << "'\n";
    return false;
}

bool Scene::init (optix::Context optix_ctx, const std::string& renderer, std::string& materials) {
    optix_ctx["radiance_ray_type"]->setUint  (0u);
    optix_ctx["shadow_ray_type"  ]->setUint  (1u);
    optix_ctx["bg_color"         ]->setFloat (0.0f, 0.0f, 0.0f);
    optix_ctx["bad_color"        ]->setFloat (1.0f, 0.0f, 1.0f);

    // Create the OptiX programs and set them on the optix::Context
    optix_ctx->setMissProgram          (0, optix_ctx->createProgramFromPTXString (renderer, "miss"));
    optix_ctx->setExceptionProgram     (0, optix_ctx->createProgramFromPTXString (renderer, "exception"));

    // Load the PTX for the wrapper program. It will be used to create OptiX
    // Materials from the OSL ShaderGroups
    if (! loadPtxFromFile("wrapper.ptx", materials))
        return false;

    std::string sphere_ptx, quad_ptx;
    if (! loadPtxFromFile("sphere.ptx", sphere_ptx))
        return false;
    if (! loadPtxFromFile("quad.ptx", quad_ptx))
        return false;

    // Create the sphere and quad intersection programs, and save them on the
    // Scene so that they don't need to be regenerated for each primitive in the
    // scene
    create_geom_programs (optix_ctx, sphere_ptx, quad_ptx);
    return true;
}

void Scene::finalize(optix::Context optix_ctx) {
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
    for (const auto& sphere : spheres) {
        optix::Geometry sphere_geom = optix_ctx->createGeometry();
        sphere.setOptixVariables (sphere_geom, sphere_bounds, sphere_intersect);

        optix::GeometryInstance sphere_gi = optix_ctx->createGeometryInstance (
            sphere_geom, &optix_mtls[sphere.shaderid()], &optix_mtls[sphere.shaderid()]+1);

        geom_group->addChild (sphere_gi);
    }

    for (const auto& quad : quads) {
        optix::Geometry quad_geom = optix_ctx->createGeometry();
        quad.setOptixVariables (quad_geom, quad_bounds, quad_intersect);

        optix::GeometryInstance quad_gi = optix_ctx->createGeometryInstance (
            quad_geom, &optix_mtls[quad.shaderid()], &optix_mtls[quad.shaderid()]+1);

        geom_group->addChild (quad_gi);
    }

    // Set the camera variables on the OptiX Context, to be used by the ray gen program
    optix_ctx["eye" ]->setFloat (vec3_to_float3 (camera.eye));
    optix_ctx["dir" ]->setFloat (vec3_to_float3 (camera.dir));
    optix_ctx["cx"  ]->setFloat (vec3_to_float3 (camera.cx));
    optix_ctx["cy"  ]->setFloat (vec3_to_float3 (camera.cy));
}

/// Return true if the texture handle (previously returned by
/// get_texture_handle()) is a valid texture that can be subsequently
/// read or sampled.
bool OptixRenderer::good(TextureHandle *handle) {
    return intptr_t(handle) != RT_TEXTURE_ID_NULL;
}

/// Given the name of a texture, return an opaque handle that can be
/// used with texture calls to avoid the name lookups.
RendererServices::TextureHandle* OptixRenderer::get_texture_handle (ustring filename) {
    auto itr = m_samplers.find(filename);
    if (itr == m_samplers.end()) {
        optix::TextureSampler sampler = m_context->createTextureSampler();
        sampler->setWrapMode(0, RT_WRAP_REPEAT);
        sampler->setWrapMode(1, RT_WRAP_REPEAT);
        sampler->setWrapMode(2, RT_WRAP_REPEAT);

        sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
        sampler->setIndexingMode(false ? RT_TEXTURE_INDEX_ARRAY_INDEX : RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        sampler->setMaxAnisotropy(1.0f);


        OIIO::ImageBuf image;
        if (!image.init_spec(filename, 0, 0)) {
            std::cerr << "Could not load:" << filename << "\n";
            return (TextureHandle*)(intptr_t(RT_TEXTURE_ID_NULL));
        }
        int nchan = image.spec().nchannels;

        OIIO::ROI roi = OIIO::get_roi_full(image.spec());
        int width = roi.width(), height = roi.height();
        std::vector<float> pixels(width * height * nchan);
        image.get_pixels(roi, OIIO::TypeDesc::FLOAT, pixels.data());

        optix::Buffer buffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, width, height);

        float* device_ptr = static_cast<float*>(buffer->map());
        unsigned int pixel_idx = 0;
        for (unsigned y = 0; y < height; ++y) {
            for (unsigned x = 0; x < width; ++x) {
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
}

bool OptixRenderer::init(const std::string& progName, int xres, int yres, Scene* scene)
{
    // Set up the OptiX context
    m_context = optix::Context::create();
    m_width  = xres;
    m_height = yres;

    ASSERT ((m_context->getEnabledDeviceCount() == 1) &&
            "Only one CUDA device is currently supported");

    m_context->setRayTypeCount (2);
    m_context->setEntryPointCount (1);
    m_context->setStackSize (2048);
    m_context->setPrintEnabled (true);

    // Load the renderer CUDA source and generate PTX for it
    std::string rendererPTX;
    if (! loadPtxFromFile(progName, rendererPTX))
        return false;

    // Create the OptiX programs and set them on the optix::Context
    m_program = m_context->createProgramFromPTXString(rendererPTX, "raygen");
    m_context->setRayGenerationProgram (0, m_program);

    // Set up the string table. This allocates a block of CUDA device memory to
    // hold all of the static strings used by the OSL shaders. The strings can
    // be accessed via OptiX variables that hold pointers to the table entries.
    m_str_table.init(m_context);

    if (scene && ! scene->init(m_context, rendererPTX, m_materials_ptx))
        return false;

    return static_cast<bool>(m_program);
}


bool OptixRenderer::finalize(ShadingSystem* shadingsys, bool saveptx, Scene* scene)
{
    int curMtl = 0;
    optix::Program closest_hit, any_hit;
    if (scene) {
        closest_hit = m_context->createProgramFromPTXString(m_materials_ptx, "closest_hit_osl");
        any_hit = m_context->createProgramFromPTXString(m_materials_ptx, "any_hit_shadow");
    }

    const char* outputs = "Cout";

    // Optimize each ShaderGroup in the scene, and use the resulting PTX to create
    // OptiX Programs which can be called by the closest hit program in the wrapper
    // to execute the compiled OSL shader.
    for (auto&& groupref : m_shaders) {
        if (!scene && outputs) {
            shadingsys->attribute (groupref.get(), "renderer_outputs", TypeDesc(TypeDesc::STRING, 1), &outputs);
        }

        shadingsys->optimize_group (groupref.get(), nullptr);

        if (!scene && outputs) {
            if (!shadingsys->find_symbol (*groupref.get(), ustring(outputs))) {
                std::cout << "Requested output '" << outputs << "', which wasn't found\n";
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
            std::cerr << "Failed to generate PTX for ShaderGroup "
                      << group_name << std::endl;
            return false;
        }

        if (saveptx) {
            std::ofstream out (group_name + "_" + std::to_string(curMtl++) + ".ptx");
            out << osl_ptx;
            out.close();
        }

        // Create Programs from the init and group_entry functions, and 
        // set the OSL functions as Callable Programs so that they can be
        // executed by the closest hit program in the wrapper
        //
        optix::Program osl_init = m_context->createProgramFromPTXString(osl_ptx, init_name);
        optix::Program osl_group = m_context->createProgramFromPTXString(osl_ptx, entry_name);

        if (scene) {
            // Create a new Material using the wrapper PTX
            scene->optix_mtls.emplace_back(m_context->createMaterial());

            optix::Material& mtl = scene->optix_mtls.back();
            mtl->setClosestHitProgram (0, closest_hit);
            mtl->setAnyHitProgram (1, any_hit);

            mtl["osl_init_func" ]->setProgramId (osl_init );
            mtl["osl_group_func"]->setProgramId (osl_group);
        } else {
            m_program["osl_init_func" ]->setProgramId (osl_init );
            m_program["osl_group_func"]->setProgramId (osl_group);
        }
    }

    if (scene)
        scene->finalize(m_context);

    // Create the output buffer
    optix::Buffer buffer = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, m_width, m_height);
    m_context["output_buffer"]->set(buffer);

    m_context["invw"]->setFloat (1.f / float(m_width));
    m_context["invh"]->setFloat (1.f / float(m_height));
    m_context->validate();

    return true;
}

std::vector<OSL::Color3>
OptixRenderer::getPixelBuffer(const std::string& buffer_name, int width, int height)
{
    const OSL::Color3* buffer_ptr =
        static_cast<OSL::Color3*>(m_context[buffer_name]->getBuffer()->map());

    if (! buffer_ptr) {
        std::cerr << "Unable to map buffer " << buffer_name << std::endl;
        exit (EXIT_FAILURE);
    }

    std::vector<OSL::Color3> pixels;
    std::copy (&buffer_ptr[0], &buffer_ptr[width * height], back_inserter(pixels));

    m_context[buffer_name]->getBuffer()->unmap();

    return pixels;
}

bool
OptixRenderer::saveImage(const std::string& buffer_name, int width, int height,
                         const std::string& imagefile, OIIO::ErrorHandler* errHandler)
{
    std::vector<OSL::Color3> pixels = getPixelBuffer("output_buffer", width, height);

    // Make an ImageBuf that wraps it ('pixels' still owns the memory)
    OIIO::ImageBuf pixelbuf(OIIO::ImageSpec(width, height, 3, TypeDesc::FLOAT), pixels.data());
    pixelbuf.set_write_format(TypeDesc::HALF);

    // Write image to disk
    if (OIIO::Strutil::iends_with(imagefile, ".jpg") ||
        OIIO::Strutil::iends_with(imagefile, ".jpeg") ||
        OIIO::Strutil::iends_with(imagefile, ".gif") ||
        OIIO::Strutil::iends_with(imagefile, ".png")) {
        // JPEG, GIF, and PNG images should be automatically saved as sRGB
        // because they are almost certainly supposed to be displayed on web
        // pages.
        OIIO::ImageBufAlgo::colorconvert(pixelbuf, pixelbuf, "linear", "sRGB", false, "", "");
    }

    pixelbuf.set_write_format (TypeDesc::HALF);
    if (pixelbuf.write(imagefile))
        return true;

    if (errHandler)
        errHandler->error("Unable to write output image: %s", pixelbuf.geterror().c_str());
    return false;
}

void
OptixRenderer::clear() {
    m_shaders.clear();
    m_context->destroy();
}

OSL_NAMESPACE_EXIT