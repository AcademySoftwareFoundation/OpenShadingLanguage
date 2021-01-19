// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <cstdio>
#include <string>
#include <vector>

#include "../liboslexec/osoreader.h"
#include <OSL/oslquery.h>
using namespace OSL;
using namespace OSL::pvt;

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/strutil.h>
namespace Filesystem = OIIO::Filesystem;
namespace Strutil    = OIIO::Strutil;
using OIIO::string_view;


OSL_NAMESPACE_ENTER
namespace pvt {


// Custom subclass of OSOReader that just reads the .oso file and fills
// out the right fields in the OSLQuery.
class OSOReaderQuery final : public OSOReader {
public:
    OSOReaderQuery(OSLQuery& query)
        : m_query(query), m_reading_param(false), m_default_values(0)
    {
    }
    virtual ~OSOReaderQuery() {}
    virtual void version(const char* /*specid*/, int /*major*/, int /*minor*/)
    {
    }
    virtual void shader(const char* shadertype, const char* name);
    virtual void symbol(SymType symtype, TypeSpec typespec, const char* name);
    virtual void symdefault(int def);
    virtual void symdefault(float def);
    virtual void symdefault(const char* def);
    virtual void parameter_done();
    virtual void hint(string_view hintstring);
    virtual void codemarker(const char* name);
    virtual bool parse_code_section() { return false; }
    virtual bool stop_parsing_at_temp_symbols() { return true; }

private:
    OSLQuery& m_query;
    bool m_reading_param;  // Are we reading a param now?
    int m_default_values;  // How many default values have we read?
};



void
OSOReaderQuery::shader(const char* shadertype, const char* name)
{
    m_query.m_shadername     = name;
    m_query.m_shadertypename = shadertype;
}



void
OSOReaderQuery::symbol(SymType symtype, TypeSpec typespec, const char* name)
{
    if (symtype == SymTypeParam || symtype == SymTypeOutputParam) {
        m_reading_param  = true;
        m_default_values = 0;
        OSLQuery::Parameter p;
        p.name        = name;
        p.type        = typespec.simpletype();  // FIXME -- struct & closure
        p.isoutput    = (symtype == SymTypeOutputParam);
        p.varlenarray = typespec.is_unsized_array();
        p.isstruct    = typespec.is_structure();
        p.isclosure   = typespec.is_closure();
        m_query.m_params.push_back(p);
    } else {
        m_reading_param = false;
    }
}



void
OSOReaderQuery::symdefault(int def)
{
    if (m_reading_param && m_query.nparams() > 0) {
        OSLQuery::Parameter& p(m_query.m_params[m_query.nparams() - 1]);
        if (p.type.basetype == TypeDesc::FLOAT)
            p.fdefault.push_back((float)def);
        else
            p.idefault.push_back(def);
        p.validdefault = true;
        m_default_values++;
    }
}



void
OSOReaderQuery::symdefault(float def)
{
    if (m_reading_param && m_query.nparams() > 0) {
        OSLQuery::Parameter& p(m_query.m_params[m_query.nparams() - 1]);
        p.fdefault.push_back(def);
        p.validdefault = true;
        m_default_values++;
    }
}



void
OSOReaderQuery::symdefault(const char* def)
{
    if (m_reading_param && m_query.nparams() > 0) {
        OSLQuery::Parameter& p(m_query.m_params[m_query.nparams() - 1]);
        p.sdefault.emplace_back(def);
        p.validdefault = true;
        m_default_values++;
    }
}



void
OSOReaderQuery::parameter_done()
{
    if (m_reading_param && m_query.nparams() > 0) {
        // Make sure all value defaults have the right number of elements in
        // case they were only partially initialized.
        OSLQuery::Parameter& p(m_query.m_params.back());
        int nvalues;
        if (p.varlenarray)
            nvalues = m_default_values;
        else
            nvalues = p.type.numelements() * p.type.aggregate;
        if (p.type.basetype == TypeDesc::INT) {
            p.idefault.resize(nvalues, 0);
            p.data = &p.idefault[0];
        } else if (p.type.basetype == TypeDesc::FLOAT) {
            p.fdefault.resize(nvalues, 0);
            p.data = &p.fdefault[0];
        } else if (p.type.basetype == TypeDesc::STRING) {
            p.sdefault.resize(nvalues, ustring());
            p.data = &p.sdefault[0];
        }
        if (p.spacename.size())
            p.spacename.resize(p.type.numelements(), ustring());
    }

    m_reading_param = false;
}



#if OPENIMAGEIO_VERSION < 20109
// A bug present in Strutil::parse_string until OIIO 2.1.8 caused incorrect
// parsing of hints with embedded escaped quote characters. Supply a copy
// of the fixed version here until our minimum OIIO is new enough to be
// sure that the bug is fixed.
static bool
parse_string(string_view& str, string_view& val, bool eat = true,
             OIIO::Strutil::QuoteBehavior keep_quotes
             = OIIO::Strutil::DeleteQuotes) noexcept
{
    using namespace OIIO::Strutil;
    string_view p = str;
    skip_whitespace(p);
    if (str.empty())
        return false;
    char lead_char    = p.front();
    bool quoted       = parse_char(p, '\"') || parse_char(p, '\'');
    const char *begin = p.begin(), *end = p.begin();
    bool escaped = false;  // was the prior character a backslash
    while (end != p.end()) {
        if (isspace(*end) && !quoted)
            break;  // not quoted and we hit whitespace: we're done
        if (quoted && *end == lead_char && !escaped)
            break;  // closing quote -- we're done (beware embedded quote)
        escaped = (end[0] == '\\') && (!escaped);
        ++end;
    }
    if (quoted && keep_quotes == KeepQuotes) {
        if (*end == lead_char)
            val = string_view(begin - 1, size_t(end - begin) + 2);
        else
            val = string_view(begin - 1, size_t(end - begin) + 1);
    } else {
        val = string_view(begin, size_t(end - begin));
    }
    p.remove_prefix(size_t(end - begin));
    if (quoted && p.size() && p[0] == lead_char)
        p.remove_prefix(1);  // eat closing quote
    if (eat)
        str = p;
    return quoted || val.size();
}
#endif



void
OSOReaderQuery::hint(string_view hintstring)
{
    if (!Strutil::parse_char(hintstring, '%'))
        return;
    if (Strutil::parse_prefix(hintstring, "meta{")) {
        // std::cerr << "  Metadata '" << hintstring << "'\n";
        Strutil::skip_whitespace(hintstring);
        std::string type = Strutil::parse_until(hintstring, ",}");
        Strutil::parse_char(hintstring, ',');
        std::string name = Strutil::parse_until(hintstring, ",}");
        Strutil::parse_char(hintstring, ',');
        // std::cerr << "    " << name << " : " << type << "\n";
        OSLQuery::Parameter p;
        p.name = name;
        p.type = TypeDesc(type);
        if (p.type.basetype == TypeDesc::STRING) {
            string_view val;
#if OPENIMAGEIO_VERSION >= 20109
            using Strutil::parse_string;
#endif
            while (parse_string(hintstring, val)) {
                p.sdefault.emplace_back(OIIO::Strutil::unescape_chars(val));
                if (Strutil::parse_char(hintstring, '}'))
                    break;
                Strutil::parse_char(hintstring, ',');
            }
        } else if (p.type.basetype == TypeDesc::INT) {
            int val;
            while (Strutil::parse_int(hintstring, val)) {
                p.idefault.push_back(val);
                Strutil::parse_char(hintstring, ',');
            }
        } else if (p.type.basetype == TypeDesc::FLOAT) {
            float val;
            while (Strutil::parse_float(hintstring, val)) {
                p.fdefault.push_back(val);
                Strutil::parse_char(hintstring, ',');
            }
        }
        Strutil::parse_char(hintstring, '}');
        if (m_reading_param)  // Parameter metadata
            m_query.m_params[m_query.nparams() - 1].metadata.push_back(p);
        else  // global shader metadata
            m_query.m_meta.push_back(p);
        return;
    }
    if (m_reading_param && Strutil::parse_prefix(hintstring, "structfields{")) {
        OSLQuery::Parameter& param(m_query.m_params[m_query.nparams() - 1]);
        while (1) {
            string_view ident = Strutil::parse_identifier(hintstring);
            if (ident.length()) {
                param.fields.emplace_back(ident);
                Strutil::parse_char(hintstring, ',');
            } else {
                break;
            }
        }
        Strutil::parse_char(hintstring, '}');
        return;
    }
    if (m_reading_param && Strutil::parse_prefix(hintstring, "struct{")) {
        string_view str;
        Strutil::parse_string(hintstring, str);
        m_query.m_params[m_query.nparams() - 1].structname = str;
        Strutil::parse_char(hintstring, '}');
        return;
    }
    if (m_reading_param && Strutil::parse_prefix(hintstring, "initexpr")) {
        m_query.m_params[m_query.nparams() - 1].validdefault = false;
        return;
    }

    // std::cerr << "Hint '" << hintstring << "'\n";
}



void
OSOReaderQuery::codemarker(const char* /*name*/)
{
    m_reading_param = false;
}


};  // namespace pvt



OSLQuery::Parameter::Parameter(const Parameter& src)
    : name(src.name)
    , type(src.type)
    , isoutput(src.isoutput)
    , validdefault(src.validdefault)
    , varlenarray(src.varlenarray)
    , isstruct(src.isstruct)
    , isclosure(src.isclosure)
    , idefault(src.idefault)
    , fdefault(src.fdefault)
    , sdefault(src.sdefault)
    , spacename(src.spacename)
    , fields(src.fields)
    , structname(src.structname)
    , metadata(src.metadata)
{
    if (type.basetype == TypeDesc::INT)
        data = idefault.data();
    else if (type.basetype == TypeDesc::FLOAT)
        data = fdefault.data();
    else if (type.basetype == TypeDesc::STRING)
        data = sdefault.data();
}



OSLQuery::Parameter::Parameter(Parameter&& src)
    : name(src.name)
    , type(src.type)
    , isoutput(src.isoutput)
    , validdefault(src.validdefault)
    , varlenarray(src.varlenarray)
    , isstruct(src.isstruct)
    , isclosure(src.isclosure)
    , idefault(src.idefault)
    , fdefault(src.fdefault)
    , sdefault(src.sdefault)
    , spacename(src.spacename)
    , fields(src.fields)
    , structname(src.structname)
    , metadata(src.metadata)
{
    if (type.basetype == TypeDesc::INT)
        data = idefault.data();
    else if (type.basetype == TypeDesc::FLOAT)
        data = fdefault.data();
    else if (type.basetype == TypeDesc::STRING)
        data = sdefault.data();
}



const OSLQuery::Parameter&
OSLQuery::Parameter::operator=(const Parameter& src)
{
    name         = src.name;
    type         = src.type;
    isoutput     = src.isoutput;
    validdefault = src.validdefault;
    varlenarray  = src.varlenarray;
    isstruct     = src.isstruct;
    isclosure    = src.isclosure;
    idefault     = src.idefault;
    fdefault     = src.fdefault;
    sdefault     = src.sdefault;
    spacename    = src.spacename;
    fields       = src.fields;
    structname   = src.structname;
    metadata     = src.metadata;
    if (type.basetype == TypeDesc::INT)
        data = idefault.data();
    else if (type.basetype == TypeDesc::FLOAT)
        data = fdefault.data();
    else if (type.basetype == TypeDesc::STRING)
        data = sdefault.data();
    return *this;
}



const OSLQuery::Parameter&
OSLQuery::Parameter::operator=(Parameter&& src)
{
    name         = src.name;
    type         = src.type;
    isoutput     = src.isoutput;
    validdefault = src.validdefault;
    varlenarray  = src.varlenarray;
    isstruct     = src.isstruct;
    isclosure    = src.isclosure;
    idefault     = std::move(src.idefault);
    fdefault     = std::move(src.fdefault);
    sdefault     = std::move(src.sdefault);
    spacename    = src.spacename;
    fields       = src.fields;
    structname   = src.structname;
    metadata     = std::move(src.metadata);
    if (type.basetype == TypeDesc::INT)
        data = idefault.data();
    else if (type.basetype == TypeDesc::FLOAT)
        data = fdefault.data();
    else if (type.basetype == TypeDesc::STRING)
        data = sdefault.data();
    return *this;
}



OSLQuery::OSLQuery() {}



OSLQuery::~OSLQuery() {}



bool
OSLQuery::open(string_view shadername, string_view searchpath)
{
    OSOReaderQuery oso(*this);
    std::string filename = shadername;

    // Add file extension if not already there
    if (Filesystem::extension(filename) != std::string(".oso"))
        filename += ".oso";

    // Apply search paths
    if (!searchpath.empty()) {
        std::vector<std::string> dirs;
        Filesystem::searchpath_split(searchpath, dirs);
        filename = Filesystem::searchpath_find(filename, dirs);
    }
    if (filename.empty()) {
        errorf("File \"%s\" could not be found.", shadername);
        return false;
    }

    bool ok = oso.parse_file(filename);
    return ok;
}

bool
OSLQuery::open_bytecode(string_view buffer)
{
    OSOReaderQuery oso(*this);
    bool ok = oso.parse_memory(buffer);
    return ok;
}

OSL_NAMESPACE_EXIT
