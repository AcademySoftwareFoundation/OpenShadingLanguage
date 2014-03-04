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


#include <vector>
#include <string>
#include <cstdio>

#include "OSL/oslquery.h"
#include "../liboslexec/osoreader.h"
using namespace OSL;
using namespace OSL::pvt;

#include <OpenImageIO/filesystem.h>
namespace Filesystem = OIIO::Filesystem;


OSL_NAMESPACE_ENTER
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
    virtual void version (const char *specid, int major, int minor) { }
    virtual void shader (const char *shadertype, const char *name);
    virtual void symbol (SymType symtype, TypeSpec typespec, const char *name);
    virtual void symdefault (int def);
    virtual void symdefault (float def);
    virtual void symdefault (const char *def);
    virtual void hint (const char *hintstring);
    virtual void codemarker (const char *name);
    virtual bool parse_code_section () { return false; }
    virtual bool stop_parsing_at_temp_symbols () { return true; }

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
        return;
    }
    if (m_reading_param && ! strncmp (hintstring, "%structfields{", 14)) {
        hintstring += 14;
        OSLQuery::Parameter &param (m_query.m_params[m_query.nparams()-1]);
        while (*hintstring) {
            std::string afield = readuntil (&hintstring, ',', '}');
            param.fields.push_back (afield);
        }
        return;
    }
    if (m_reading_param && ! strncmp (hintstring, "%struct{", 8)) {
        hintstring += 8;
        if (*hintstring == '\"')  // skip quote
            ++hintstring;
        OSLQuery::Parameter &param (m_query.m_params[m_query.nparams()-1]);
        param.structname = readuntil (&hintstring, '\"', '}');
        return;
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
    std::string filename = shadername;

    // Add file extension if not already there
    if (Filesystem::file_extension (filename) != std::string("oso"))
        filename += ".oso";

    // Apply search paths
    if (! searchpath.empty ()) {
        std::vector<std::string> dirs;
        Filesystem::searchpath_split (searchpath, dirs);
        filename = Filesystem::searchpath_find (filename, dirs);
    }
    if (filename.empty()) {
        m_error = std::string("File \"") + shadername + "\" could not be found";
        return false;
    }

    bool ok = oso.parse_file (filename);
    return ok;
}


OSL_NAMESPACE_EXIT
