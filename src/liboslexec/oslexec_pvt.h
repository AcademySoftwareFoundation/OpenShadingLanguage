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
    typedef intrusive_ptr<ShaderMaster> Ref;
    ShaderMaster () { }
    ~ShaderMaster () { }

private:
    ShaderType m_shadertype;            ///< Type of shader
    ustring m_shadername;               ///< Shader name
    // Need the code
    // Need the code offsets for each code block
    // Need the argument list (ints)
    // Need the symbols
    // Need constant values (int, float, string)
    // Need default values for each parameter (int, float, string)
};


/// ShaderInstance is a particular instance of a shader, with its own
/// set of parameter values, coordinate transform, and connections to
/// other instances within the same shader group.
class ShaderInstance : public RefCnt {
public:
    ShaderInstance () { }
    ~ShaderInstance () { }
private:
    ShaderMaster::Ref m_master;         ///< Reference to the master
    ustring m_layername;                ///< Name of this layer
    // Need instance values for each parameter (int, float, string)

};



class ShadingSystemImpl
{
public:
    ShadingSystemImpl () { }
    ~ShadingSystemImpl () { }

    ShaderMaster::Ref read_shader (const char *name);

private:
    friend class OSOReaderToMaster;
};


}; // namespace pvt
}; // namespace OSL


#endif /* OSLEXEC_PVT_H */
