// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include "rtweekend.h"
#include "shading.h"

#include <OSL/oslexec.h>

#include <cstring>
#include <memory>
#include <utility>

// The book has one material class each for matte, metal and glass. Here a
// single class covers all three: the OSL shader it holds decides which.
class material {
public:
    material(OSL::ShadingSystem& shadingsys, OSL::ShaderGroupRef group)
        : shadingsys(shadingsys), group(std::move(group))
    {
    }

    [[nodiscard]] bool scatter(const ray& r_in, const hit_record& rec,
                               OSL::ShadingContext& ctx, color& attenuation,
                               ray& scattered) const
    {
        vec3 I = unit_vector(r_in.direction());

        OSL::ShaderGlobals sg;
        globals_from_hit(sg, I, rec);

        // 0 is fine for the thread index: it is only read by renderers that
        // collect shader messages per thread, and this one does not.
        if (!shadingsys.execute(ctx, *group, /*thread_index=*/0,
                                /*shadeindex=*/0, sg, nullptr, nullptr))
            return false;

        if (sg.Ci == nullptr)
            return false;

        vec3 wi;
        if (!sample_closure(sg.Ci, I, wi, attenuation))
            return false;

        scattered = ray(rec.p, wi);
        return true;
    }

private:
    OSL::ShadingSystem& shadingsys;
    OSL::ShaderGroupRef group;

    // The book's vec3 holds doubles, and OSL's Vec3 holds floats.
    static OSL::Vec3 to_osl(const vec3& v)
    {
        return OSL::Vec3(static_cast<float>(v.x()), static_cast<float>(v.y()),
                         static_cast<float>(v.z()));
    }

    static void globals_from_hit(OSL::ShaderGlobals& sg, const vec3& I,
                                 const hit_record& rec)
    {
        // Imath vectors start uninitialized, so `{}` would not zero this.
        std::memset(static_cast<void*>(&sg), 0, sizeof(sg));

        sg.P = to_osl(rec.p);
        sg.I = to_osl(I);

        // rec.normal already faces the ray, which is what OSL wants.
        sg.N = sg.Ng  = to_osl(rec.normal);
        sg.backfacing = rec.front_face ? 0 : 1;

        sg.u    = static_cast<float>(rec.u);
        sg.v    = static_cast<float>(rec.v);
        sg.dPdu = to_osl(rec.dpdu);
        sg.dPdv = to_osl(rec.dpdv);

        // Only light shaders would read this, and there are none.
        sg.surfacearea = 1;
    }
};



// Each material is a shader group holding one surface shader.

inline std::shared_ptr<material>
make_matte(OSL::ShadingSystem& shadingsys, const OSL::Color3& albedo, bool& ok)
{
    OSL::ShaderGroupRef group = shadingsys.ShaderGroupBegin("matte");
    ok &= shadingsys.Parameter(*group, "Cs", OSL::TypeColor, &albedo);
    ok &= shadingsys.Shader(*group, "surface", "matte", "layer");
    ok &= shadingsys.ShaderGroupEnd(*group);
    return std::make_shared<material>(shadingsys, group);
}



inline std::shared_ptr<material>
make_metal(OSL::ShadingSystem& shadingsys, const OSL::Color3& albedo,
           float fuzz, bool& ok)
{
    OSL::ShaderGroupRef group = shadingsys.ShaderGroupBegin("metal");
    ok &= shadingsys.Parameter(*group, "Cs", OSL::TypeColor, &albedo);
    ok &= shadingsys.Parameter(*group, "fuzz", OSL::TypeFloat, &fuzz);
    ok &= shadingsys.Shader(*group, "surface", "metal", "layer");
    ok &= shadingsys.ShaderGroupEnd(*group);
    return std::make_shared<material>(shadingsys, group);
}



inline std::shared_ptr<material>
make_glass(OSL::ShadingSystem& shadingsys, float ior, bool& ok)
{
    OSL::ShaderGroupRef group = shadingsys.ShaderGroupBegin("glass");
    ok &= shadingsys.Parameter(*group, "ior", OSL::TypeFloat, &ior);
    ok &= shadingsys.Shader(*group, "surface", "glass", "layer");
    ok &= shadingsys.ShaderGroupEnd(*group);
    return std::make_shared<material>(shadingsys, group);
}



// An addition to the book's three materials, to test a sum of closures.
inline std::shared_ptr<material>
make_plastic(OSL::ShadingSystem& shadingsys, const OSL::Color3& albedo,
             bool& ok)
{
    OSL::ShaderGroupRef group = shadingsys.ShaderGroupBegin("plastic");
    ok &= shadingsys.Parameter(*group, "Cs", OSL::TypeColor, &albedo);
    ok &= shadingsys.Shader(*group, "surface", "plastic", "layer");
    ok &= shadingsys.ShaderGroupEnd(*group);
    return std::make_shared<material>(shadingsys, group);
}
