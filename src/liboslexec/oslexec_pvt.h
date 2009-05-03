/*****************************************************************************
 *
 *             Copyright (c) 2009 Sony Pictures Imageworks, Inc.
 *                            All rights reserved.
 *
 *  This material contains the confidential and proprietary information
 *  of Sony Pictures Imageworks, Inc. and may not be disclosed, copied or
 *  duplicated in any form, electronic or hardcopy, in whole or in part,
 *  without the express prior written consent of Sony Pictures Imageworks,
 *  Inc. This copyright notice does not imply publication.
 *
 *****************************************************************************/

#ifndef OSLEXEC_PVT_H
#define OSLEXEC_PVT_H

#include "OpenImageIO/ustring.h"
#include "OpenImageIO/typedesc.h"
#include "OpenImageIO/thread.h"
#include "OpenImageIO/refcnt.h"

#include "osl_pvt.h"


namespace OSL {
namespace pvt {


/// ShaderMaster is the full internal representation of a complete
/// shader that would be a .oso file on disk: symbols, instructions,
/// arguments, you name it.  A master copy is shared by all the 
/// individual instances of the shader.
class ShaderMaster : public RefCnt {
public:
    typedef intrusive_ptr<ShaderMaster> ref;
    ShaderMaster () { }
    ~ShaderMaster () { }

    void print ();  // Debugging

private:
    ShaderType m_shadertype;            ///< Type of shader
    ustring m_shadername;               ///< Shader name
    OpcodeVec m_ops;                    ///< Actual code instructions
    std::vector<int> m_args;            ///< Arguments for all the ops
    // Need the code offsets for each code block
    SymbolVec m_symbols;                ///< Symbols used by the shader
    std::vector<int> m_idefaults;       ///< int default values
    std::vector<float> m_fdefaults;     ///< float default values
    std::vector<ustring> m_sdefaults;   ///< string default values
    friend class OSOReaderToMaster;
};



/// ShaderInstance is a particular instance of a shader, with its own
/// set of parameter values, coordinate transform, and connections to
/// other instances within the same shader group.
class ShaderInstance : public RefCnt {
public:
    ShaderInstance () { }
    ~ShaderInstance () { }
private:
    ShaderMaster::ref m_master;         ///< Reference to the master
    ustring m_layername;                ///< Name of this layer
    // Need instance values for each parameter (int, float, string)

};



class ShadingSystemImpl
{
public:
    ShadingSystemImpl () { }
    ~ShadingSystemImpl () { }

    ShaderMaster::ref loadshader (const char *name);

private:

};


}; // namespace pvt
}; // namespace OSL


#endif /* OSLEXEC_PVT_H */
