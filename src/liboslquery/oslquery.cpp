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


// ptr is a pointer to a char pointer.  Read chars from *ptr until you
// hit the end of the string, or the next char is one of stop1 or stop2.
// At that point, return what we hit, and also update *ptr to then point
// to the next character following the stop character.
static std::string
readuntil (const char **ptr, char stop1, char stop2=-1)
{
    std::string s;
    while (**ptr == ' ')
        ++(*ptr);
    while (**ptr && **ptr != stop1 && **ptr != stop2) {
        s += (**ptr);
        ++(*ptr);
    }
    if (**ptr == stop1 || **ptr == stop2)
        ++(*ptr);
    while (**ptr == ' ')
        ++(*ptr);
    return s;
}



static TypeDesc
string_to_type (const char *s)
{
    TypeDesc t;
    std::string tname = readuntil (&s, ' ');
    if (tname == "int")
        t = TypeDesc::TypeInt;
    if (tname == "float")
        t = TypeDesc::TypeFloat;
    if (tname == "color")
        t = TypeDesc::TypeColor;
    if (tname == "point")
        t = TypeDesc::TypePoint;
    if (tname == "vector")
        t = TypeDesc::TypeVector;
    if (tname == "normal")
        t = TypeDesc::TypeNormal;
    if (tname == "matrix")
        t = TypeDesc::TypeMatrix;
    if (tname == "string")
        t = TypeDesc::TypeString;
    if (*s == '[') {
        ++s;
        if (*s == ']')
            t.arraylen = -1;
        else
            t.arraylen = atoi (s);
    }
    return t;
}



// Custom subclass of OSOReader that just reads the .oso file and fills
// out the right fields in the OSLQuery.
class OSOReaderQuery : public OSOReader
{
public:
    OSOReaderQuery (OSLQuery &query) : m_query(query), m_reading_param(false)
    { }
    virtual ~OSOReaderQuery () { }
    virtual void version (const char *specid, float version) { }
    virtual void shader (const char *shadertype, const char *name);
    virtual void symbol (SymType symtype, TypeSpec typespec, const char *name);
    virtual void symdefault (int def);
    virtual void symdefault (float def);
    virtual void symdefault (const char *def);
    virtual void hint (const char *hintstring);
    virtual void codemarker (const char *name);
    virtual void instruction (int label, const char *opcode) { }
    virtual void instruction_arg (const char *name) { }
    virtual void instruction_jump (int target) { }

private:
    OSLQuery &m_query;
    bool m_reading_param;     // Are we reading a param now?
};



void
OSOReaderQuery::shader (const char *shadertype, const char *name)
{
    m_query.m_shadername = name;
    m_query.m_shadertype = shadertype;
}



void
OSOReaderQuery::symbol (SymType symtype, TypeSpec typespec, const char *name)
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



void
OSOReaderQuery::symdefault (int def)
{
    if (m_reading_param && m_query.nparams() > 0) {
        OSLQuery::Parameter &p (m_query.m_params[m_query.nparams()-1]);
        if (p.type.basetype == TypeDesc::FLOAT)
            p.fdefault.push_back ((float)def);
        else
            p.idefault.push_back (def);
        p.validdefault = true;
    }
}



void
OSOReaderQuery::symdefault (float def)
{
    if (m_reading_param && m_query.nparams() > 0) {
        OSLQuery::Parameter &p (m_query.m_params[m_query.nparams()-1]);
        p.fdefault.push_back (def);
        p.validdefault = true;
    }
}



void
OSOReaderQuery::symdefault (const char *def)
{
    if (m_reading_param && m_query.nparams() > 0) {
        OSLQuery::Parameter &p (m_query.m_params[m_query.nparams()-1]);
        p.sdefault.push_back (std::string(def));
        p.validdefault = true;
    }
}



void
OSOReaderQuery::hint (const char *hintstring)
{
    if (m_reading_param && ! strncmp (hintstring, "%meta{", 6)) {
        hintstring += 6;
        // std::cerr << "  Metadata '" << hintstring << "'\n";
        std::string type = readuntil (&hintstring, ',', '}');
        std::string name = readuntil (&hintstring, ',', '}');
        // std::cerr << "    " << name << " : " << type << "\n";
        OSLQuery::Parameter p;
        p.name = name;
        p.type = string_to_type (type.c_str());
        if (p.type.basetype == TypeDesc::STRING) {
            while (*hintstring == ' ')
                ++hintstring;
            while (hintstring[0] == '\"') {
                ++hintstring;
                p.sdefault.push_back (readuntil (&hintstring, '\"'));
            }
        } else if (p.type.basetype == TypeDesc::INT) {
            while (*hintstring == ' ')
                ++hintstring;
            while (*hintstring && *hintstring != '}') {
                p.idefault.push_back (atoi (hintstring));
                readuntil (&hintstring, ',', '}');
            }
        } else if (p.type.basetype == TypeDesc::FLOAT) {
            while (*hintstring == ' ')
                ++hintstring;
            while (*hintstring && *hintstring != '}') {
                p.fdefault.push_back (atof (hintstring));
                readuntil (&hintstring, ',', '}');
            }
        }
        m_query.m_params[m_query.nparams()-1].metadata.push_back (p);
    }
    // std::cerr << "Hint '" << hintstring << "'\n";
}



void
OSOReaderQuery::codemarker (const char *name)
{
    m_reading_param = false;
}


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

