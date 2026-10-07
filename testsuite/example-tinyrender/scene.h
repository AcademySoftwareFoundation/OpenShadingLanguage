// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include "osl/material.h"
#include "camera.h"
#include "rtweekend.h"

#include <memory>

inline bool
setup_scene(OSL::ShadingSystem& shadingsys, hittable_list& world)
{
    using OSL::Color3;
    using std::make_shared;

    bool ok = true;

    auto ground  = make_matte(shadingsys, Color3(0.5f, 0.5f, 0.5f), ok);
    auto blue    = make_matte(shadingsys, Color3(0.1f, 0.2f, 0.5f), ok);
    auto green   = make_matte(shadingsys, Color3(0.2f, 0.6f, 0.2f), ok);
    auto glass   = make_glass(shadingsys, 1.5f, ok);
    auto bubble  = make_glass(shadingsys, 1.0f / 1.5f, ok);
    auto gold    = make_metal(shadingsys, Color3(0.8f, 0.6f, 0.2f), 0.3f, ok);
    auto mirror  = make_metal(shadingsys, Color3(0.8f, 0.8f, 0.8f), 0.0f, ok);
    auto plastic = make_plastic(shadingsys, Color3(0.7f, 0.1f, 0.1f), ok);

    world.add(make_shared<sphere>(point3(0.0, -100.5, -1.0), 100.0, ground));

    world.add(make_shared<sphere>(point3(-1.1, 0.0, -1.5), 0.5, blue));
    world.add(make_shared<sphere>(point3(0.0, 0.0, -1.5), 0.5, glass));
    world.add(make_shared<sphere>(point3(0.0, 0.0, -1.5), 0.4, bubble));
    world.add(make_shared<sphere>(point3(1.1, 0.0, -1.5), 0.5, gold));

    world.add(make_shared<sphere>(point3(-0.9, -0.25, -0.4), 0.25, plastic));
    world.add(make_shared<sphere>(point3(-0.3, -0.25, -0.4), 0.25, mirror));
    world.add(make_shared<sphere>(point3(0.3, -0.25, -0.4), 0.25, glass));
    world.add(make_shared<sphere>(point3(0.9, -0.25, -0.4), 0.25, green));

    return ok;
}



// Resolution and sample count are small to keep the test quick.
inline camera
setup_camera()
{
    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 160;
    cam.samples_per_pixel = 64;
    cam.max_depth         = 10;

    cam.vfov     = 30;
    cam.lookfrom = point3(0, 1.6, 3.2);
    cam.lookat   = point3(0, -0.1, -1);
    cam.vup      = vec3(0, 1, 0);

    return cam;
}
