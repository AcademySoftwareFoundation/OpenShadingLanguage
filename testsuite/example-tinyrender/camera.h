// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include "osl/material.h"
#include "rtweekend.h"

#include <OpenImageIO/parallel.h>

#include <cstddef>
#include <cstdint>
#include <vector>

class camera {
public:
    double aspect_ratio   = 1.0;
    int image_width       = 100;
    int samples_per_pixel = 10;
    int max_depth         = 10;

    double vfov     = 90;  // Vertical field of view
    point3 lookfrom = point3(0, 0, 0);
    point3 lookat   = point3(0, 0, -1);
    vec3 vup        = vec3(0, 1, 0);  // Camera-relative "up" direction

    int height() const { return image_height; }

    // Linear color. The book's gamma correction is left out, since the
    // output is an EXR and those hold linear values.
    const std::vector<color>& pixels() const { return frameBuffer; }

    void render(const hittable& world, OSL::ShadingSystem& shadingsys)
    {
        initialize();

        OIIO::parallel_for_chunked(
            0, image_height, 0,
            [this, &world, &shadingsys](int64_t jbegin, int64_t jend) {
                // A ShadingContext can only be used by one thread at a time.
                OSL::PerThreadInfo* thread_info
                    = shadingsys.create_thread_info();
                OSL::ShadingContext* ctx = shadingsys.get_context(thread_info);

                for (int j = static_cast<int>(jbegin); j < jend; j++) {
                    // Reseed per row, so the image is the same for any number
                    // of threads.
                    seed_random_generator(static_cast<unsigned int>(j));

                    for (int i = 0; i < image_width; i++) {
                        color pixel_color(0, 0, 0);
                        for (int sample = 0; sample < samples_per_pixel;
                             sample++) {
                            ray r = get_ray(i, j);
                            pixel_color += ray_color(r, max_depth, world, *ctx);
                        }
                        frameBuffer[static_cast<std::size_t>(j) * image_width
                                    + i] = pixel_samples_scale * pixel_color;
                    }
                }

                shadingsys.release_context(ctx);
                shadingsys.destroy_thread_info(thread_info);
            });
    }

private:
    int image_height           = 0;
    double pixel_samples_scale = 0;
    point3 center;
    point3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    vec3 u, v, w;  // Camera frame basis vectors
    std::vector<color> frameBuffer;

    void initialize()
    {
        image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        frameBuffer.resize(static_cast<std::size_t>(image_width)
                           * image_height);
        pixel_samples_scale = 1.0 / samples_per_pixel;

        center = lookfrom;

        // Determine viewport dimensions.
        double focal_length    = (lookfrom - lookat).length();
        double theta           = degrees_to_radians(vfov);
        double h               = std::tan(theta / 2);
        double viewport_height = 2 * h * focal_length;
        double viewport_width  = viewport_height
                                 * (static_cast<double>(image_width)
                                    / image_height);

        // Calculate the u, v, w unit basis vectors for the camera frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        vec3 viewport_u = viewport_width * u;
        vec3 viewport_v = viewport_height * -v;

        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        point3 viewport_upper_left = center - (focal_length * w)
                                     - viewport_u / 2 - viewport_v / 2;
        pixel00_loc                = viewport_upper_left
                                     + 0.5 * (pixel_delta_u + pixel_delta_v);
    }

    ray get_ray(int i, int j) const
    {
        // A ray from the camera center to a random point around pixel i, j.
        vec3 offset         = sample_square();
        point3 pixel_sample = pixel00_loc + ((i + offset.x()) * pixel_delta_u)
                              + ((j + offset.y()) * pixel_delta_v);

        return ray(center, pixel_sample - center);
    }

    static vec3 sample_square()
    {
        // A random point in the [-.5,-.5]-[+.5,+.5] unit square.
        double x = random_double() - 0.5;
        double y = random_double() - 0.5;
        return vec3(x, y, 0);
    }

    color ray_color(const ray& r, int depth, const hittable& world,
                    OSL::ShadingContext& ctx) const
    {
        // Past the bounce limit, no more light is gathered.
        if (depth <= 0)
            return color(0, 0, 0);

        hit_record rec;

        if (world.hit(r, interval(0.001, infinity), rec)) {
            ray scattered;
            color attenuation;
            if (rec.mat->scatter(r, rec, ctx, attenuation, scattered))
                return attenuation
                       * ray_color(scattered, depth - 1, world, ctx);
            return color(0, 0, 0);
        }

        vec3 unit_direction = unit_vector(r.direction());
        double a            = 0.5 * (unit_direction.y() + 1.0);
        return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
    }
};
