#pragma once

#include <optix_math.h>


namespace {  // anonymous namespace

#ifdef __cplusplus
    typedef optix::float3 float3;
#endif

struct ShaderGlobals {
    float3 P, dPdx, dPdy;
    float3 dPdz;
    float3 I, dIdx, dIdy;
    float3 N;
    float3 Ng;
    float  u, dudx, dudy;
    float  v, dvdx, dvdy;
    float3 dPdu, dPdv;
    float  time;
    float  dtime;
    float3 dPdtime;
    float3 Ps, dPsdx, dPsdy;
    void*  renderstate;
    void*  tracedata;
    void*  objdata;
    void*  context;
    void*  renderer;
    void*  object2common;
    void*  shader2common;
    void*  Ci;
    float  surfacearea;
    int    raytype;
    int    flipHandedness;
    int    backfacing;
};


enum RayType {
    CAMERA       = 1,
    SHADOW       = 2,
    REFLECTION   = 4,
    REFRACTION   = 8,
    DIFFUSE      = 16,
    GLOSSY       = 32,
    SUBSURFACE   = 64,
    DISPLACEMENT = 128
};


// Closures supported by the OSL sample renderer.  This list is mosly aspirational.
enum ClosureIDs {
    EMISSION_ID = 1,
    BACKGROUND_ID,
    DIFFUSE_ID,
    OREN_NAYAR_ID,
    TRANSLUCENT_ID,
    PHONG_ID,
    WARD_ID,
    MICROFACET_ID,
    REFLECTION_ID,
    FRESNEL_REFLECTION_ID,
    REFRACTION_ID,
    TRANSPARENT_ID,
    DEBUG_ID,
    HOLDOUT_ID,
};


struct ClosureColor {
    enum ClosureID { COMPONENT_BASE_ID = 0, MUL = -1, ADD = -2 };
    int id;
};


struct ClosureComponent : public ClosureColor {
    float3 w;
    char   mem[8];
};


struct ClosureMul : public ClosureColor {
    float3        weight;
    ClosureColor* closure;
};


struct ClosureAdd : public ClosureColor {
    ClosureColor* closureA;
    ClosureColor* closureB;
};

}  // anonymous namespace
