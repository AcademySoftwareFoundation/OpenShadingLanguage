/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
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

#include "genclosure.h"
#include "oslclosure.h"
#include "oslexec.h"
#include "oslcomp.h"
#include "OpenImageIO/dassert.h"



#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {

namespace pvt {

extern ClosureParam bsdf_diffuse_params[];
extern ClosureParam bsdf_translucent_params[];
extern ClosureParam bsdf_reflection_params[];
extern ClosureParam bsdf_refraction_params[];
extern ClosureParam bsdf_dielectric_params[];
extern ClosureParam bsdf_transparent_params[];
extern ClosureParam bsdf_microfacet_ggx_params[];
extern ClosureParam bsdf_microfacet_ggx_refraction_params[];
extern ClosureParam bsdf_microfacet_beckmann_params[];
extern ClosureParam bsdf_microfacet_beckmann_refraction_params[];
extern ClosureParam bsdf_ward_params[];
extern ClosureParam bsdf_phong_params[];
extern ClosureParam bsdf_phong_ramp_params[];
extern ClosureParam bsdf_hair_diffuse_params[];
extern ClosureParam bsdf_hair_specular_params[];
extern ClosureParam bsdf_ashikhmin_velvet_params[];
extern ClosureParam bsdf_cloth_params[];
extern ClosureParam bsdf_cloth_specular_params[];
extern ClosureParam bsdf_fakefur_diffuse_params[];
extern ClosureParam bsdf_fakefur_specular_params[];
extern ClosureParam bsdf_fakefur_skin_params[];
extern ClosureParam bsdf_westin_backscatter_params[];
extern ClosureParam bsdf_westin_sheen_params[];
extern ClosureParam closure_bssrdf_cubic_params[];
extern ClosureParam closure_emission_params[];
extern ClosureParam closure_background_params[];
extern ClosureParam closure_subsurface_params[];

void bsdf_diffuse_prepare(RendererServices *, int id, void *data);
void bsdf_translucent_prepare(RendererServices *, int id, void *data);
void bsdf_reflection_prepare(RendererServices *, int id, void *data);
void bsdf_refraction_prepare(RendererServices *, int id, void *data);
void bsdf_dielectric_prepare(RendererServices *, int id, void *data);
void bsdf_transparent_prepare(RendererServices *, int id, void *data);
void bsdf_microfacet_ggx_prepare(RendererServices *, int id, void *data);
void bsdf_microfacet_ggx_refraction_prepare(RendererServices *, int id, void *data);
void bsdf_microfacet_beckmann_prepare(RendererServices *, int id, void *data);
void bsdf_microfacet_beckmann_refraction_prepare(RendererServices *, int id, void *data);
void bsdf_ward_prepare(RendererServices *, int id, void *data);
void bsdf_phong_prepare(RendererServices *, int id, void *data);
void bsdf_phong_ramp_prepare(RendererServices *, int id, void *data);
void bsdf_hair_diffuse_prepare(RendererServices *, int id, void *data);
void bsdf_hair_specular_prepare(RendererServices *, int id, void *data);
void bsdf_ashikhmin_velvet_prepare(RendererServices *, int id, void *data);
void bsdf_cloth_prepare(RendererServices *, int id, void *data);
void bsdf_cloth_specular_prepare(RendererServices *, int id, void *data);
void bsdf_fakefur_diffuse_prepare(RendererServices *, int id, void *data);
void bsdf_fakefur_specular_prepare(RendererServices *, int id, void *data);
void bsdf_fakefur_skin_prepare(RendererServices *, int id, void *data);
void bsdf_westin_backscatter_prepare(RendererServices *, int id, void *data);
void bsdf_westin_sheen_prepare(RendererServices *, int id, void *data);
void closure_bssrdf_cubic_prepare(RendererServices *, int id, void *data);
void closure_emission_prepare(RendererServices *, int id, void *data);
void closure_background_prepare(RendererServices *, int id, void *data);
void closure_subsurface_prepare(RendererServices *, int id, void *data);

} // namespace pvt

struct BuiltinClosure {
    int                id;
    const char         *name;
    ClosureParam       *params;
    PrepareClosureFunc prepare;
};

BuiltinClosure builtin_closures[NBUILTIN_CLOSURES] = {
    { CLOSURE_BSDF_DIFFUSE_ID,                        "diffuse",     pvt::bsdf_diffuse_params,
      pvt::bsdf_diffuse_prepare },
    { CLOSURE_BSDF_TRANSLUCENT_ID,                    "translucent", pvt::bsdf_translucent_params,
      pvt::bsdf_translucent_prepare },
    { CLOSURE_BSDF_REFLECTION_ID,                     "reflection",  pvt::bsdf_reflection_params,
      pvt::bsdf_reflection_prepare },
    { CLOSURE_BSDF_REFRACTION_ID,                     "refraction",  pvt::bsdf_refraction_params,
      pvt::bsdf_refraction_prepare },
    { CLOSURE_BSDF_DIELECTRIC_ID,                     "dielectric",  pvt::bsdf_dielectric_params,
      pvt::bsdf_dielectric_prepare },
    { CLOSURE_BSDF_TRANSPARENT_ID,                    "transparent", pvt::bsdf_transparent_params,
      pvt::bsdf_transparent_prepare },
    { CLOSURE_BSDF_MICROFACET_GGX_ID,                 "microfacet_ggx", pvt::bsdf_microfacet_ggx_params,
      pvt::bsdf_microfacet_ggx_prepare },
    { CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID,      "microfacet_ggx_refraction", pvt::bsdf_microfacet_ggx_refraction_params,
      pvt::bsdf_microfacet_ggx_refraction_prepare },
    { CLOSURE_BSDF_MICROFACET_BECKMANN_ID,            "microfacet_beckmann", pvt::bsdf_microfacet_beckmann_params,
      pvt::bsdf_microfacet_beckmann_prepare },
    { CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID, "microfacet_beckmann_refraction", pvt::bsdf_microfacet_beckmann_refraction_params,
      pvt::bsdf_microfacet_beckmann_refraction_prepare },
    { CLOSURE_BSDF_WARD_ID,                           "ward",          pvt::bsdf_ward_params,
      pvt::bsdf_ward_prepare },
    { CLOSURE_BSDF_PHONG_ID,                          "phong",         pvt::bsdf_phong_params,
      pvt::bsdf_phong_prepare },
    { CLOSURE_BSDF_PHONG_RAMP_ID,                     "phong_ramp",    pvt::bsdf_phong_ramp_params,
      pvt::bsdf_phong_ramp_prepare },
    { CLOSURE_BSDF_HAIR_DIFFUSE_ID,                   "hair_diffuse",  pvt::bsdf_hair_diffuse_params,
      pvt::bsdf_hair_diffuse_prepare },
    { CLOSURE_BSDF_HAIR_SPECULAR_ID,                  "hair_specular", pvt::bsdf_hair_specular_params,
      pvt::bsdf_hair_specular_prepare },
    { CLOSURE_BSDF_ASHIKHMIN_VELVET_ID,               "ashikhmin_velvet", pvt::bsdf_ashikhmin_velvet_params,
      pvt::bsdf_ashikhmin_velvet_prepare },
    { CLOSURE_BSDF_CLOTH_ID,                          "cloth",           pvt::bsdf_cloth_params,
      pvt::bsdf_cloth_prepare },
    { CLOSURE_BSDF_CLOTH_SPECULAR_ID,                 "cloth_specular",  pvt::bsdf_cloth_specular_params,
      pvt::bsdf_cloth_specular_prepare },
    { CLOSURE_BSDF_FAKEFUR_DIFFUSE_ID,                "fakefur_diffuse", pvt::bsdf_fakefur_diffuse_params,
      pvt::bsdf_fakefur_diffuse_prepare },
    { CLOSURE_BSDF_FAKEFUR_SPECULAR_ID,               "fakefur_specular",pvt::bsdf_fakefur_specular_params,
      pvt::bsdf_fakefur_specular_prepare },
    { CLOSURE_BSDF_FAKEFUR_SKIN_ID,                   "fakefur_skin",    pvt::bsdf_fakefur_skin_params,
      pvt::bsdf_fakefur_skin_prepare },
    { CLOSURE_BSDF_WESTIN_BACKSCATTER_ID,             "westin_backscatter", pvt::bsdf_westin_backscatter_params,
      pvt::bsdf_westin_backscatter_prepare },
    { CLOSURE_BSDF_WESTIN_SHEEN_ID,                   "westin_sheen", pvt::bsdf_westin_sheen_params,
      pvt::bsdf_westin_sheen_prepare },
    { CLOSURE_BSSRDF_CUBIC_ID,                        "bssrdf_cubic", pvt::closure_bssrdf_cubic_params,
      pvt::closure_bssrdf_cubic_prepare },
    { CLOSURE_EMISSION_ID,                            "emission",     pvt::closure_emission_params,
      pvt::closure_emission_prepare },
    { CLOSURE_BACKGROUND_ID,                          "background",   pvt::closure_background_params,
      pvt::closure_background_prepare },
    { CLOSURE_SUBSURFACE_ID,                          "subsurface",   pvt::closure_subsurface_params,
      pvt::closure_subsurface_prepare } };



static void generic_closure_setup(RendererServices *, int id, void *data)
{
   ClosurePrimitive *prim = (ClosurePrimitive *)data;
   prim->setup();
}



static bool generic_closure_compare(int id, const void *dataA, const void *dataB)
{
   ClosurePrimitive *primA = (ClosurePrimitive *)dataA;
   ClosurePrimitive *primB = (ClosurePrimitive *)dataB;
   return primA->mergeable (primB);
}



void ShadingSystem::register_builtin_closures(ShadingSystem *ss)
{
    for (int cid = 0; cid < NBUILTIN_CLOSURES; ++cid)
    {
        BuiltinClosure *clinfo = &builtin_closures[cid];
        int j;
        for (j = 0; clinfo->params[j].type != TypeDesc(); ++j);
        int size = clinfo->params[j].offset;
        ASSERT(clinfo->id == cid);
        ss->register_closure (clinfo->name, cid, clinfo->params, size, clinfo->prepare, generic_closure_setup, generic_closure_compare,
                              reckless_offsetof(ClosurePrimitive, m_custom_labels),
                              ClosurePrimitive::MAXCUSTOM);
    }
}

void ShadingSystem::register_builtin_closures(OSLCompiler *cc)
{
    for (int cid = 0; cid < NBUILTIN_CLOSURES; ++cid)
    {
        BuiltinClosure *clinfo = &builtin_closures[cid];
        int j;
        for (j = 0; clinfo->params[j].type != TypeDesc(); ++j);
        ASSERT(clinfo->id == cid);
        cc->register_closure (clinfo->name, clinfo->params, true);
    }
}



}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
