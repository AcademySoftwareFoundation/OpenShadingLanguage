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


#include <vector>
#include <string>
#include <cstdio>

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/dassert.h"
#include "OpenImageIO/thread.h"

#include "oslexec_pvt.h"
#include "osoreader.h"




namespace OSL {

namespace pvt {   // OSL::pvt


/// Custom subclass of OSOReader that provide callbacks that set all the
/// right fields in the ShaderMaster.
class OSOReaderToMaster : public OSOReader
{
public:
    OSOReaderToMaster (ShaderMaster &master) : m_master(master) { }
    virtual ~OSOReaderToMaster () { }
    virtual void version (const char *specid, float version) { }
    virtual void shader (const char *shadertype, const char *name) { }
    virtual void symbol (SymType symtype, TypeSpec typespec, const char *name) { }
    virtual void hint (const char *string) { }
    virtual void codemarker (const char *name) { }
    virtual void instruction (int label, const char *opcode) { }
    virtual void instruction_arg (const char *name) { }
    virtual void instruction_jump (int target) { }

private:
    ShaderMaster &m_master;
};



ShaderMaster::Ref
read_shader (const char *name)
{
    OSOReader oso;
    std::string filename = name;   // FIXME -- do search, etc.
    bool ok = oso.parse (name);
    return ok ? NULL /* FIXME */ : NULL;
}


}; // namespace pvt
}; // namespace OSL
