// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "shading.h"

#include "rtweekend.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>

using OSL::TypeDesc;
using OSL::TypeFloat;
using OSL::TypeVector;

void
register_closures(OSL::ShadingSystem& shadingsys)
{
    // Describe the memory layout of each closure's parameters to OSL.
    constexpr int MaxParams = 4;

    struct BuiltinClosures {
        const char* name;
        int id;
        OSL::ClosureParam params[MaxParams];
    };

    const BuiltinClosures builtins[] = {
        { "diffuse",
          DIFFUSE_ID,
          { CLOSURE_VECTOR_PARAM(DiffuseParams, N),
            CLOSURE_FINISH_PARAM(DiffuseParams) } },
        { "rtiow_metal",
          RTIOW_METAL_ID,
          { CLOSURE_VECTOR_PARAM(RtiowMetalParams, N),
            CLOSURE_FLOAT_PARAM(RtiowMetalParams, fuzz),
            CLOSURE_FINISH_PARAM(RtiowMetalParams) } },
        { "rtiow_dielectric",
          RTIOW_DIELECTRIC_ID,
          { CLOSURE_VECTOR_PARAM(RtiowDielectricParams, N),
            CLOSURE_FLOAT_PARAM(RtiowDielectricParams, eta),
            CLOSURE_FINISH_PARAM(RtiowDielectricParams) } },
    };

    for (const BuiltinClosures& b : builtins)
        shadingsys.register_closure(b.name, b.id, b.params, nullptr, nullptr);
}



constexpr std::size_t max_lobes = 8;

struct Lobe {
    int id             = 0;
    const void* params = nullptr;
    color weight;       // closure weight accumulated down the tree
    double albedo = 0;  // rough estimate of the light this lobe carries
};

using LobeList = std::array<Lobe, max_lobes>;

vec3
from_osl(const OSL::Vec3& v)
{
    return vec3(v.x, v.y, v.z);
}



double
average(const color& c)
{
    return (c.x() + c.y() + c.z()) / 3;
}



// Schlick's approximation, as in the book's dielectric::reflectance().
double
schlick_reflectance(double cosine, double eta)
{
    double r0 = (1 - eta) / (1 + eta);
    r0        = r0 * r0;
    return r0 + (1 - r0) * std::pow(1 - cosine, 5);
}



// The book's lambertian::scatter().
bool
sample_diffuse(const vec3& n, vec3& wi, color& value)
{
    // A unit vector added to the normal is a cosine-weighted hemisphere
    // sample, and cosine weighting makes f * cos / pdf collapse to 1.
    vec3 direction = n + random_unit_vector();

    if (direction.near_zero())
        direction = n;

    wi    = direction;
    value = color(1, 1, 1);
    return true;
}



// The book's metal::scatter().
bool
sample_rtiow_metal(const RtiowMetalParams* p, const vec3& I, vec3& wi,
                   color& value)
{
    vec3 n      = from_osl(p->N);
    double fuzz = std::fmin(static_cast<double>(p->fuzz), 1.0);

    wi = unit_vector(reflect(I, n)) + fuzz * random_unit_vector();

    // Fuzz can push the ray below the surface, where the book absorbs it.
    if (dot(wi, n) <= 0)
        return false;

    value = color(1, 1, 1);
    return true;
}



// The book's dielectric::scatter().
bool
sample_rtiow_dielectric(const RtiowDielectricParams* p, const vec3& I, vec3& wi,
                        color& value)
{
    vec3 n = from_osl(p->N);
    // incident over transmitted, already flipped by the shader
    double eta = p->eta;

    double cos_theta = std::fmin(dot(-I, n), 1.0);
    double sin_theta = std::sqrt(1 - cos_theta * cos_theta);

    // Past the critical angle the ray can only reflect.
    bool cannot_refract = eta * sin_theta > 1.0;

    if (cannot_refract || schlick_reflectance(cos_theta, eta) > random_double())
        wi = reflect(I, n);
    else
        wi = refract(I, n, eta);

    value = color(1, 1, 1);
    return true;
}



void
flatten(const OSL::ClosureColor* c, const color& weight, LobeList& lobes,
        std::size_t& count)
{
    if (c == nullptr)
        return;

    if (count >= lobes.size()) {
        static std::atomic<bool> warned { false };
        if (!warned.exchange(true))
            OSL::print(stderr,
                       "warning: a closure tree has more than {} lobes. "
                       "Ignoring the rest...\n",
                       max_lobes);
        return;
    }

    switch (c->id) {
    case OSL::ClosureColor::MUL: {
        const OSL::ClosureMul* m = c->as_mul();
        flatten(m->closure,
                weight * color(m->weight.x, m->weight.y, m->weight.z), lobes,
                count);
        break;
    }

    case OSL::ClosureColor::ADD: {
        const OSL::ClosureAdd* a = c->as_add();
        flatten(a->closureA, weight, lobes, count);
        flatten(a->closureB, weight, lobes, count);
        break;
    }

    default: {
        const OSL::ClosureComponent* comp = c->as_comp();
        color w = weight * color(comp->w.x, comp->w.y, comp->w.z);

        lobes[count] = { comp->id, comp->data(), w };
        count++;
        break;
    }
    }
}



bool
sample_lobe(const Lobe& l, const vec3& I, vec3& wi, color& value)
{
    switch (l.id) {
    case DIFFUSE_ID:
        return sample_diffuse(
            from_osl(static_cast<const DiffuseParams*>(l.params)->N), wi,
            value);

    case RTIOW_METAL_ID:
        return sample_rtiow_metal(static_cast<const RtiowMetalParams*>(l.params),
                                  I, wi, value);

    case RTIOW_DIELECTRIC_ID:
        return sample_rtiow_dielectric(
            static_cast<const RtiowDielectricParams*>(l.params), I, wi, value);

    default: return false;
    }
}



bool
sample_closure(const OSL::ClosureColor* Ci, const vec3& I, vec3& wi,
               color& weight)
{
    LobeList lobes;
    std::size_t count = 0;

    flatten(Ci, color(1, 1, 1), lobes, count);
    if (count == 0)
        return false;

    double total = 0;
    for (std::size_t i = 0; i < count; i++) {
        lobes[i].albedo = std::fmax(0.0, average(lobes[i].weight));
        total += lobes[i].albedo;
    }

    if (total <= 0)
        return false;

    double selection_point   = random_double() * total;
    std::size_t chosen       = count - 1;
    double cumulative_albedo = 0;

    for (std::size_t i = 0; i < count; i++) {
        cumulative_albedo += lobes[i].albedo;
        if (selection_point < cumulative_albedo) {
            chosen = i;
            break;
        }
    }

    double probability = lobes[chosen].albedo / total;
    if (probability <= 0)
        return false;

    color value;
    if (!sample_lobe(lobes[chosen], I, wi, value))
        return false;

    weight = lobes[chosen].weight * value / probability;
    return true;
}
