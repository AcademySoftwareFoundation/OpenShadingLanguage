// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// A minimal standalone renderer that shades with OSL, built against an
// installed OSL. The ray tracer is from "Ray Tracing in One Weekend" by
// Peter Shirley, Trevor David Black, and Steve Hollasch
// (https://raytracing.github.io), with its materials replaced by OSL shaders.

#include "osl/material.h"
#include "osl/shading.h"
#include "camera.h"
#include "rtweekend.h"
#include "scene.h"

#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>

#include <OpenImageIO/imagebuf.h>

#include <cstdio>
#include <cstdlib>
#include <string>

// The scene is world space only, so every transform is identity.
class TinyRendererServices : public OSL::RendererServices {
public:
    bool get_matrix(OSL::ShaderGlobals*, OSL::Matrix44& result,
                    OSL::TransformationPtr, float) override
    {
        result.makeIdentity();
        return true;
    }

    bool get_matrix(OSL::ShaderGlobals*, OSL::Matrix44& result,
                    OSL::TransformationPtr) override
    {
        result.makeIdentity();
        return true;
    }
};



bool
write_image(const camera& cam, const std::string& filename)
{
    const int width  = cam.image_width;
    const int height = cam.height();

    OIIO::ImageBuf buf(OIIO::ImageSpec(width, height, 3, OIIO::TypeFloat));
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            const color& c
                = cam.pixels()[static_cast<std::size_t>(j) * width + i];
            const float rgb[3] = { static_cast<float>(c.x()),
                                   static_cast<float>(c.y()),
                                   static_cast<float>(c.z()) };
            buf.setpixel(i, j, rgb);
        }
    }

    if (!buf.write(filename, OIIO::TypeHalf)) {
        OSL::print(stderr, "could not write {}: {}\n", filename,
                   buf.geterror());
        return false;
    }
    return true;
}



int
main()
{
    // As in RTIOW, the materials do the shading, so each one holds a
    // reference to the shading system. The shading system is declared
    // before the scene so that it outlives them.
    TinyRendererServices services;
    OSL::ShadingSystem shadingsys(&services);
    shadingsys.attribute("searchpath:shader", TINYRENDER_SHADER_DIR);
    register_closures(shadingsys);

    hittable_list world;
    if (!setup_scene(shadingsys, world)) {
        OSL::print(stderr, "could not build the scene's materials from {}\n",
                   TINYRENDER_SHADER_DIR);
        return EXIT_FAILURE;
    }

    camera cam = setup_camera();
    cam.render(world, shadingsys);

    if (!write_image(cam, "out.exr"))
        return EXIT_FAILURE;

    OSL::print("Rendered {}x{} image with {} samples\n", cam.image_width,
               cam.height(), cam.samples_per_pixel);
    return EXIT_SUCCESS;
}
