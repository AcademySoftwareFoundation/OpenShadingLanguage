// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include "oslexec_pvt.h"

OSL_NAMESPACE_ENTER

namespace pvt {



class BatchedAnalysis {
public:
    using ShaderGroup = OSL::ShaderGroup;

    BatchedAnalysis(ShadingSystemImpl& shadingsys, ShaderGroup& group);

    ShadingSystemImpl& shadingsys() const { return m_shadingsys; }
    RendererServices* renderer() const { return shadingsys().renderer(); }

    void analyze_layer(ShaderInstance* inst);

    void dump_layer(ShaderInstance* inst);
    void dump_symbol_uniformity(ShaderInstance* inst);

    ShaderGroup& group() const { return m_group; }

protected:
    ShadingSystemImpl& m_shadingsys;
    ShaderGroup& m_group;
};


};  // namespace pvt
OSL_NAMESPACE_EXIT
