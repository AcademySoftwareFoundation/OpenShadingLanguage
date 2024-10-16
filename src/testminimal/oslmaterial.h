// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once
#include <OSL/dual_vec.h>
#include <OSL/genclosure.h>
#include <OSL/oslclosure.h>
#include <OSL/oslcomp.h>
#include <OSL/oslconfig.h>
#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>

#if OSL_USE_BATCHED
#    include <OSL/batched_rendererservices.h>
#    include <OSL/batched_shaderglobals.h>
#endif

class OSLMaterial;

#if OSL_USE_BATCHED
template<int batch_width> class BatchedOSLMaterial;

using OSL::Vec3;

/// Custom BatchedRendererServices
template<int batch_width>
class CustomBatchedRendererServices
    : public OSL::BatchedRendererServices<batch_width> {
public:
    explicit CustomBatchedRendererServices(BatchedOSLMaterial<batch_width>& m);

    //OIIO::ErrorHandler& errhandler() const { return *m_errhandler; }
    /// Turn information at hitpoint into ShaderGlobals for OSL
    void globals_from_hit(OSL::BatchedShaderGlobals<batch_width>& bsg)
    {
        // Uniform
        auto& usg = bsg.uniform;
        // Zero it all
        std::memset(&usg, 0, sizeof(OSL::UniformShaderGlobals));
        usg.raytype = 1;  // 1 stands for camera ray?
        // Varying
        auto& vsg = bsg.varying;

        //assign_all(vsg.shader2common, TransformationPtr(&Mshad));
        //assign_all(vsg.object2common, TransformationPtr(&Mobj));

        for (int i = 0; i < batch_width; i++)
            vsg.P[i] = { 0.0f, 0.0f, 0.0f };

        for (int i = 0; i < batch_width; i++)
            vsg.I[i] = { 0.0f, 0.0f, -1.0f };  // incident ray
        for (int i = 0; i < batch_width; i++)
            vsg.N[i] = { 0.0f, 0.0f, 1.0f };  // shading normal
        for (int i = 0; i < batch_width; i++)
            vsg.Ng[i] = { 0.0f, 0.0f, 1.0f };  // true geometric normal

        assign_all(vsg.u,
                   0.5f);  // 2D surface parameter u, and its differentials.
        assign_all(vsg.v,
                   0.5f);  // 2D surface parameter u, and its differentials.


        //if (false == vary_udxdy) {
        assign_all(vsg.dudx, 0.0f);  //uscale / xres);
        assign_all(vsg.dudy, 0.0f);
        //}
        //if (false == vary_vdxdy) {
        assign_all(vsg.dvdx, 0.0f);
        assign_all(vsg.dvdy, 0.0f);  //vscale / yres);
        //}


        //if (false == vary_Pdxdy) {
        //    assign_all(vsg.dPdx, Vec3(vsg.dudx[0], vsg.dudy[0], 0.0f));
        //    assign_all(vsg.dPdy, Vec3(vsg.dvdx[0], vsg.dvdy[0], 0.0f));
        //}

        assign_all(vsg.dPdz,
                   Vec3(0.0f, 0.0f, 0.0f));  // just use 0 for volume tangent

        // Tangents of P with respect to surface u,v
        assign_all(vsg.dPdu, Vec3(1.0f, 0.0f, 0.0f));
        assign_all(vsg.dPdv, Vec3(0.0f, 1.0f, 0.0f));

        assign_all(vsg.I, Vec3(0, 0, 0));
        assign_all(vsg.dIdx, Vec3(0, 0, 0));
        assign_all(vsg.dIdy, Vec3(0, 0, 0));

        // That also implies that our normal points to (0,0,1)
        assign_all(vsg.N, Vec3(0, 0, 1));
        assign_all(vsg.Ng, Vec3(0, 0, 1));

        assign_all(vsg.time, 0.0f);
        assign_all(vsg.dtime, 0.0f);
        assign_all(vsg.dPdtime, Vec3(0, 0, 0));

        assign_all(vsg.Ps, Vec3(0, 0, 0));
        assign_all(vsg.dPsdx, Vec3(0, 0, 0));
        assign_all(vsg.dPsdy, Vec3(0, 0, 0));

        assign_all(vsg.surfacearea, 1.0f);
        assign_all(vsg.flipHandedness, 0);
        assign_all(vsg.backfacing, 0);

        assign_all(vsg.Ci, (::OSL::ClosureColor*)NULL);
    }

    bool is_overridden_get_inverse_matrix_WmWxWf() const override
    {
        return false;
    };
    bool is_overridden_get_matrix_WmWsWf() const override { return false; };
    bool is_overridden_get_inverse_matrix_WmsWf() const override
    {
        return false;
    };
    bool is_overridden_get_inverse_matrix_WmWsWf() const override
    {
        return false;
    };
    bool is_overridden_texture() const override { return false; };
    bool is_overridden_texture3d() const override { return false; };
    bool is_overridden_environment() const override { return false; };
    bool is_overridden_pointcloud_search() const override { return false; };
    bool is_overridden_pointcloud_get() const override { return false; };
    bool is_overridden_pointcloud_write() const override { return false; };

    BatchedOSLMaterial<batch_width>& m_sr;

private:
};
#endif

/// Custom RendererServices for non-batched case
class OSLMaterial : public OSL::RendererServices {
public:
    OSLMaterial();

    void run_test(OSL::ShadingSystem* ss, OSL::PerThreadInfo* thread_info,
                  OSL::ShadingContext* context, char* shader_name);

    OIIO::ErrorHandler& errhandler() const { return *m_errhandler; }

    /// Turn information at hitpoint into ShaderGlobals for OSL
    void globals_from_hit(OSL::ShaderGlobals& sg)
    {
        sg.P    = { 0.0f, 0.0f, 0.0f };  // surface pos
        sg.dPdx = { 0.0f, 0.0f, 0.0f };
        sg.dPdy = { 0.0f, 0.0f, 0.0f };
        sg.dPdz = { 0.0f, 0.0f, 0.0f };  // for volume shading only

        sg.I    = { 0.0f, 0.0f, -1.0f };  // incident ray
        sg.dIdx = { 0.0f, 0.0f, 0.0f };
        sg.dIdy = { 0.0f, 0.0f, 0.0f };

        sg.N  = { 0.0f, 0.0f, 1.0f };  // shading normal
        sg.Ng = { 0.0f, 0.0f, 1.0f };  // true geometric normal

        sg.u    = 0.5f;  // 2D surface parameter u, and its differentials.
        sg.dudx = 0.0f;
        sg.dudy = 0.0f;
        sg.v    = 0.5f;  // 2D surface parameter v, and its differentials.
        sg.dvdx = 0.0f;
        sg.dvdy = 0.0f;

        // Surface tangents: derivative of P with respect to surface u and v.
        sg.dPdu = { 1.0f, 0.0f, 0.0f };
        sg.dPdv = { 0.0f, 1.0f, 0.0f };

        sg.time  = 0.0f;
        sg.dtime = 0.001f;

        // Velocity vector: derivative of position P with respect to time.
        sg.dPdtime = { 0.0f, 0.0f, 0.0f };

        // For lights or light attenuation shaders: the point being illuminated (???)
        sg.Ps    = { 0.0f, 0.0f, 0.0f };
        sg.dPsdx = { 0.0f, 0.0f, 0.0f };
        sg.dPsdy = { 0.0f, 0.0f, 0.0f };

        // Renderer user pointers
        sg.renderstate = NULL;
        sg.tracedata   = NULL;
        sg.objdata     = NULL;

        sg.renderer = this;

        sg.raytype        = 1;  // 1 stands for camera ray?
        sg.flipHandedness = 0;
        sg.backfacing     = 0;

        // output closure, needs to be null initialized
        sg.Ci = NULL;
    }

    // ShaderGroupRef storage
    std::vector<OSL::ShaderGroupRef>& shaders() { return m_shaders; }
    std::vector<OSL::ShaderGroupRef> m_shaders;

private:
    std::unique_ptr<OIIO::ErrorHandler> m_errhandler;
};

#if OSL_USE_BATCHED

/// Custom RendererServices for batched case
template<int batch_width>
class BatchedOSLMaterial : public OSL::RendererServices {
public:
    BatchedOSLMaterial();

    void run_test(OSL::ShadingSystem* ss, OSL::PerThreadInfo* thread_info,
                  OSL::ShadingContext* context, char* shader_name);

    OIIO::ErrorHandler& errhandler() const { return *m_errhandler; }

    // ShaderGroupRef storage
    std::vector<OSL::ShaderGroupRef>& shaders() { return m_shaders; }
    std::vector<OSL::ShaderGroupRef> m_shaders;

    OSL::BatchedRendererServices<batch_width>*
    batched(OSL::WidthOf<batch_width>) override
    {
        return &m_batch;
    }

    CustomBatchedRendererServices<batch_width> m_batch;

private:
    std::unique_ptr<OIIO::ErrorHandler> m_errhandler;
};

#endif
