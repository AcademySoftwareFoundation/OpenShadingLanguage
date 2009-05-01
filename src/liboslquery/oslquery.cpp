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

#include "OpenImageIO/thread.h"
#include "OpenImageIO/dassert.h"

#include "oslquery.h"
#include "../liboslexec/osoreader.h"
using namespace OSL;
using namespace OSL::pvt;


namespace OSL {

namespace pvt {


// Custom subclass of OSOReader that just reads the .oso file and fills
// out the right fields in the OSLQuery.
class OSOReaderQuery : public OSOReader
{
public:
    OSOReaderQuery (OSLQuery &query) : m_query(query), m_reading_param(false)
    { }
    virtual ~OSOReaderQuery () { }
    virtual void version (const char *specid, float version) { }
    virtual void shader (const char *shadertype, const char *name) {
        m_query.m_shadername = name;
        m_query.m_shadertype = shadertype;
    }
    virtual void symbol (SymType symtype, TypeSpec typespec, const char *name)
    {
        if (symtype == SymTypeParam || symtype == SymTypeOutputParam) {
            m_reading_param = true;
            OSLQuery::Parameter p;
            p.name = name;
            p.type = typespec.simpletype();   // FIXME -- struct & closure
            p.isoutput = (symtype == SymTypeOutputParam);
            p.varlenarray = (typespec.arraylength() < 0);
            p.isstruct = typespec.is_structure();
            p.isclosure = typespec.is_closure();
            m_query.m_params.push_back (p);
        } else {
            m_reading_param = false;
        }
    }
    virtual void symdefault (int def) {
        if (m_reading_param && m_query.nparams() > 0) {
            OSLQuery::Parameter &p (m_query.m_params[m_query.nparams()-1]);
            if (p.type.basetype == TypeDesc::FLOAT)
                p.fdefault.push_back ((float)def);
            else
                p.idefault.push_back (def);
            p.validdefault = true;
        }
    }
    virtual void symdefault (float def) {
        if (m_reading_param && m_query.nparams() > 0) {
            OSLQuery::Parameter &p (m_query.m_params[m_query.nparams()-1]);
            p.fdefault.push_back (def);
            p.validdefault = true;
        }
    }
    virtual void symdefault (const char *def) {
        if (m_reading_param && m_query.nparams() > 0) {
            OSLQuery::Parameter &p (m_query.m_params[m_query.nparams()-1]);
            p.sdefault.push_back (std::string(def));
            p.validdefault = true;
        }
    }
    virtual void hint (const char *string) {
        // FIXME
    }
    virtual void codemarker (const char *name) { }
    virtual void instruction (int label, const char *opcode) { }
    virtual void instruction_arg (const char *name) { }
    virtual void instruction_jump (int target) { }

private:
    OSLQuery &m_query;
    bool m_reading_param;     // Are we reading a param now?
};


};  // end namespace OSL::pvt




OSLQuery::OSLQuery ()
{
}



OSLQuery::~OSLQuery ()
{
}



bool
OSLQuery::open (const std::string &shadername,
                const std::string &searchpath)
{
    OSOReaderQuery oso (*this);
    std::string filename = shadername;   // FIXME -- do search, etc.
    bool ok = oso.parse (filename);
    return ok;
}


};   // end namespace OSL

