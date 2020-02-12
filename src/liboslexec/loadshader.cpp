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
#include <cmath> // FIXME: used by timer.h - should be included there

#include "oslexec_pvt.h"
#include "osoreader.h"

#include <OpenImageIO/strutil.h>
#include <OpenImageIO/timer.h>
#include <OpenImageIO/thread.h>
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/hash.h>



OSL_NAMESPACE_ENTER

namespace pvt {   // OSL::pvt


/// Custom subclass of OSOReader that provide callbacks that set all the
/// right fields in the ShaderMaster.
class OSOReaderToMaster : public OSOReader
{
public:
    OSOReaderToMaster (ShadingSystemImpl &shadingsys)
        : OSOReader (&shadingsys.errhandler()), m_shadingsys (shadingsys),
          m_master (new ShaderMaster (shadingsys)),
          m_reading_instruction(false), m_errors(false)
      { }
    virtual ~OSOReaderToMaster () { }
    virtual bool parse_file (const std::string &filename);
    virtual bool parse_memory (const std::string &oso);
    virtual void version (const char *specid, int major, int minor);
    virtual void shader (const char *shadertype, const char *name);
    virtual void symbol (SymType symtype, TypeSpec typespec, const char *name);
    virtual void symdefault (int def);
    virtual void symdefault (float def);
    virtual void symdefault (const char *def);
    virtual void parameter_done();
    virtual void hint (string_view hintstring);
    virtual void codemarker (const char *name);
    virtual void codeend ();
    virtual void instruction (int label, const char *opcode);
    virtual void instruction_arg (const char *name);
    virtual void instruction_jump (int target);
    virtual void instruction_end ();

    ShaderMaster::ref master () const { return m_master; }

    void add_param_default (float def, size_t offset, const Symbol& sym);
    void add_param_default (int def, size_t offset, const Symbol& sym);
    void add_param_default (const char *def, size_t offset, const Symbol& sym);

private:
    ShadingSystemImpl &m_shadingsys;  ///< Reference to the shading system
    ShaderMaster::ref m_master;       ///< Reference to our master
    size_t m_firstarg;                ///< First argument in current op
    size_t m_nargs;                   ///< Number of args so far in current op
    bool m_reading_instruction;       ///< Are we reading an op?
    ustring m_sourcefile;             ///< Current source file parsed
    int m_sourceline;                 ///< Current source code line parsed
    ustring m_codesection;            ///< Which entry point are the ops for?
    int m_codesym;                    ///< Which param is being initialized?
    int m_oso_major, m_oso_minor;     ///< oso file format version
    int m_sym_default_index;          ///< Next sym default value to fill in
    bool m_errors;                    ///< Did we hit any errors?
    typedef std::unordered_map<ustring,int,ustringHash> UstringIntMap;
    UstringIntMap m_symmap;           ///< map sym name to index
};



bool
OSOReaderToMaster::parse_file (const std::string &filename)
{
    m_master->m_osofilename = filename;
    m_master->m_maincodebegin = 0;
    m_master->m_maincodeend = 0;
    m_codesection.clear ();
    m_codesym = -1;
    return OSOReader::parse_file (filename) && ! m_errors;
}



bool
OSOReaderToMaster::parse_memory (const std::string &oso)
{
    m_master->m_osofilename = "<none>";
    m_master->m_maincodebegin = 0;
    m_master->m_maincodeend = 0;
    m_codesection.clear ();
    m_codesym = -1;
    return OSOReader::parse_memory (oso) && ! m_errors;
}




void
OSOReaderToMaster::version (const char* /*specid*/, int major, int minor)
{
    m_oso_major = major;
    m_oso_minor = minor;
}



void
OSOReaderToMaster::shader (const char *shadertype, const char *name)
{
    m_master->m_shadername = name; //ustring(name);
    m_master->m_shadertype = shadertype_from_name (shadertype);
}



void
OSOReaderToMaster::symbol (SymType symtype, TypeSpec typespec, const char *name_)
{
    ustring name(name_);
    Symbol sym (name, typespec, symtype);
    TypeDesc t = typespec.simpletype();
    int nvals = t.aggregate * (t.is_unsized_array() ? 1 : t.numelements());
    if (sym.symtype() == SymTypeParam || sym.symtype() == SymTypeOutputParam) {
        // Skip structs for now, they're just placeholders
        if (typespec.is_structure()) {
        }
        else if (typespec.simpletype().basetype == TypeDesc::FLOAT) {
            sym.dataoffset ((int) m_master->m_fdefaults.size());
            expand (m_master->m_fdefaults, nvals);
        } else if (typespec.simpletype().basetype == TypeDesc::INT) {
            sym.dataoffset ((int) m_master->m_idefaults.size());
            expand (m_master->m_idefaults, nvals);
        } else if (typespec.simpletype().basetype == TypeDesc::STRING) {
            sym.dataoffset ((int) m_master->m_sdefaults.size());
            expand (m_master->m_sdefaults, nvals);
        } else if (typespec.is_closure_based()) {
            // Closures are pointers, so we allocate a string default taking
            // adventage of their default being NULL as well.
            sym.dataoffset ((int) m_master->m_sdefaults.size());
            expand (m_master->m_sdefaults, nvals);
        } else {
            OSL_DASSERT (0 && "unexpected type");
        }
    }
    if (sym.symtype() == SymTypeConst) {
        if (typespec.simpletype().basetype == TypeDesc::FLOAT) {
            sym.dataoffset ((int) m_master->m_fconsts.size());
            expand (m_master->m_fconsts, nvals);
        } else if (typespec.simpletype().basetype == TypeDesc::INT) {
            sym.dataoffset ((int) m_master->m_iconsts.size());
            expand (m_master->m_iconsts, nvals);
        } else if (typespec.simpletype().basetype == TypeDesc::STRING) {
            sym.dataoffset ((int) m_master->m_sconsts.size());
            expand (m_master->m_sconsts, nvals);
        } else {
            OSL_DASSERT (0 && "unexpected type");
        }
    }
#if 0
    // FIXME -- global_heap_offset is quite broken.  But also not necessary.
    // We made need to fix this later.
    if (sym.symtype() == SymTypeGlobal) {
        sym.dataoffset (m_shadingsys.global_heap_offset (sym.name()));
    }
#endif
    sym.lockgeom (m_shadingsys.lockgeom_default());
    m_master->m_symbols.push_back (sym);
    m_symmap[name] = int(m_master->m_symbols.size()) - 1;
    // Start the index at which we add specified defaults
    m_sym_default_index = 0;
}



void
OSOReaderToMaster::add_param_default (float def, size_t offset, const Symbol& sym)
{
  if (sym.typespec().is_unsized_array() && offset >= m_master->m_fdefaults.size())
      m_master->m_fdefaults.push_back(def);
  else
      m_master->m_fdefaults[offset] = def;
}



void
OSOReaderToMaster::add_param_default (int def, size_t offset, const Symbol& sym)
{
  if (sym.typespec().is_unsized_array() && offset >= m_master->m_idefaults.size())
      m_master->m_idefaults.push_back(def);
  else
      m_master->m_idefaults[offset] = def;
}



void
OSOReaderToMaster::add_param_default (const char *def, size_t offset, const Symbol& sym)
{
  if (sym.typespec().is_unsized_array() && offset >= m_master->m_sdefaults.size())
      m_master->m_sdefaults.emplace_back(def);
  else
      m_master->m_sdefaults[offset] = ustring(def);
}



void
OSOReaderToMaster::symdefault (int def)
{
    OSL_DASSERT (m_master->m_symbols.size() && "symdefault but no sym");
    Symbol &sym (m_master->m_symbols.back());
    size_t offset = sym.dataoffset() + m_sym_default_index;
    ++m_sym_default_index;

    if (sym.symtype() == SymTypeParam || sym.symtype() == SymTypeOutputParam) {
        if (sym.typespec().simpletype().basetype == TypeDesc::FLOAT)
            add_param_default ((float)def, offset, sym);
        else if (sym.typespec().simpletype().basetype == TypeDesc::INT)
            add_param_default (def, offset, sym);
        else {
            OSL_DASSERT_MSG (0, "unexpected type: %s (%s)",
                             sym.typespec().c_str(), sym.name().c_str());
        }
    } else if (sym.symtype() == SymTypeConst) {
        if (sym.typespec().simpletype().basetype == TypeDesc::FLOAT)
            m_master->m_fconsts[offset] = (float)def;
        else if (sym.typespec().simpletype().basetype == TypeDesc::INT)
            m_master->m_iconsts[offset] = def;
        else {
            OSL_DASSERT_MSG (0, "unexpected type: %s (%s)",
                             sym.typespec().c_str(), sym.name().c_str());
        }
    }
}



void
OSOReaderToMaster::symdefault (float def)
{
    OSL_DASSERT (m_master->m_symbols.size() && "symdefault but no sym");
    Symbol &sym (m_master->m_symbols.back());
    size_t offset = sym.dataoffset() + m_sym_default_index;
    ++m_sym_default_index;
    if (sym.symtype() == SymTypeParam || sym.symtype() == SymTypeOutputParam) {
        if (sym.typespec().simpletype().basetype == TypeDesc::FLOAT)
            add_param_default (def, offset, sym);
        else {
            OSL_DASSERT_MSG (0, "unexpected type: %s (%s)",
                             sym.typespec().c_str(), sym.name().c_str());
        }
    } else if (sym.symtype() == SymTypeConst) {
        if (sym.typespec().simpletype().basetype == TypeDesc::FLOAT)
            m_master->m_fconsts[offset] = def;
        else {
            OSL_DASSERT_MSG (0, "unexpected type: %s (%s)",
                             sym.typespec().c_str(), sym.name().c_str());
        }
    }
}



void
OSOReaderToMaster::symdefault (const char *def)
{
    OSL_DASSERT (m_master->m_symbols.size() && "symdefault but no sym");
    Symbol &sym (m_master->m_symbols.back());
    size_t offset = sym.dataoffset() + m_sym_default_index;
    ++m_sym_default_index;
    if (sym.symtype() == SymTypeParam || sym.symtype() == SymTypeOutputParam) {
        if (sym.typespec().simpletype().basetype == TypeDesc::STRING)
            add_param_default (def, offset, sym);
        else {
            OSL_DASSERT_MSG (0, "unexpected type: %s (%s)",
                             sym.typespec().c_str(), sym.name().c_str());
        }
    } else if (sym.symtype() == SymTypeConst) {
        if (sym.typespec().simpletype().basetype == TypeDesc::STRING)
            m_master->m_sconsts[offset] = ustring(def);
        else {
            OSL_DASSERT_MSG (0, "unexpected type: %s (%s)",
                             sym.typespec().c_str(), sym.name().c_str());
        }
    }
}



void
OSOReaderToMaster::parameter_done ()
{
  OSL_DASSERT (m_master->m_symbols.size() && "parameter_done but no sym");
  Symbol &sym (m_master->m_symbols.back());

  // set length of unsized array parameters
  if (sym.symtype() == SymTypeParam && sym.typespec().is_unsized_array())
      sym.initializers (m_sym_default_index / sym.typespec().aggregate());
}



void
OSOReaderToMaster::hint (string_view hintstring)
{
    string_view h (hintstring);

    if (Strutil::parse_prefix (h, "%filename{\"")) {
        m_sourcefile = Strutil::parse_until (h, "\"");
        return;
    }
    if (Strutil::parse_prefix (h, "%line{")) {
        Strutil::parse_int (h, m_sourceline);
        return;
    }
    if (Strutil::parse_prefix (h, "%structfields{") && m_master->m_symbols.size()) {
        Symbol &sym (m_master->m_symbols.back());
        StructSpec *structspec = sym.typespec().structspec();
        if (structspec->numfields() == 0) {
            while (1) {
                std::string afield = Strutil::parse_until (h, ",}");
                Strutil::parse_char (h, ','); // skip the separator
                if (! afield.length())
                    break;
                structspec->add_field (TypeSpec(), ustring(afield));
            }
        }
        return;
    }
    if (Strutil::parse_prefix (h, "%mystructfield{") && m_master->m_symbols.size()) {
        int ival = -1;
        if (Strutil::parse_int (h, ival) && ival >= 0)
            m_master->m_symbols.back().fieldid (ival);
        return;
    }
    if (Strutil::parse_prefix (h, "%read{") && m_master->m_symbols.size()) {
        Symbol &sym (m_master->m_symbols.back());
        int first, last;
        if (Strutil::parse_int (h, first) && Strutil::parse_char(h, ',')
                && Strutil::parse_int (h, last))
            sym.set_read (first, last);
        return;
    }
    if (Strutil::parse_prefix (h, "%write{") && m_master->m_symbols.size()) {
        Symbol &sym (m_master->m_symbols.back());
        int first, last;
        if (Strutil::parse_int (h, first) && Strutil::parse_char(h, ',')
                && Strutil::parse_int (h, last))
            sym.set_write (first, last);
        return;
    }
    if (Strutil::parse_prefix(h, "%argrw{") && m_master->m_ops.size()) {
        Opcode &op (m_master->m_ops.back());
        string_view str = Strutil::parse_until (h, "}");
        Strutil::parse_string (str, str, false, Strutil::DeleteQuotes);
        if (str.size() != m_nargs) {
            m_shadingsys.errorf("Parsing shader %s: malformed hint '%s' on op %s line %d",
                                m_master->shadername(), hintstring,
                                m_master->m_ops.back().opname(), m_sourceline);
            m_errors = true;
        }
        for (size_t i = 0; str.size() && i < m_nargs; i++, str.remove_prefix(1)) {
            char c = str.front();
            op.argwrite (i, c == 'w' || c =='W');
            op.argread (i, c == 'r' || c =='W');
        }
        // Fix old bug where oslc forgot to mark getmatrix last arg as write
        ustring opname = m_master->m_ops.back().opname();
        static ustring getmatrix("getmatrix");
        if (opname == getmatrix)
            m_master->m_ops.back().argwriteonly (m_nargs-1);
        // Fix old bug where oslc forgot to mark regex results as write.
        // This was a bug prior to 1.10.
        static ustring regex_search("regex_search");
        static ustring regex_match("regex_match");
        if (opname == regex_search || opname == regex_search)
            m_master->m_ops.back().argwriteonly (2);
        return;
    }
    if (Strutil::parse_prefix(h, "%argderivs{")) {
        while (1) {
            string_view afield = Strutil::parse_until (h, ",}");
            Strutil::parse_char (h, ','); // skip the separator
            if (! afield.length())
                break;
            int arg = -1;
            if (Strutil::parse_int (afield, arg) && arg >= 0)
                m_master->m_ops.back().argtakesderivs (arg, true);
        }
        return;
    }
    if (Strutil::parse_prefix (h, "%meta{")) {
        // parse type and name
        int ival = -1;
        TypeDesc type(Strutil::parse_identifier(h, "", true));
        Strutil::parse_char(h, ',');
        string_view ident = Strutil::parse_identifier(h, "", true);
        Strutil::parse_char(h, ',');
        if (m_master->m_symbols.size()) {
            // metadata is attached to a particular symbol
            Symbol& sym(m_master->m_symbols.back());
            if (type == TypeDesc::TypeInt && ident == "lockgeom"
                && Strutil::parse_int(h, ival) && ival >= 0)
                sym.lockgeom(ival);
            else if (type == TypeDesc::TypeInt && ident == "allowconnect"
                     && Strutil::parse_int(h, ival) && ival >= 0)
                sym.allowconnect(ival);
        } else {
            // metadata is attached at the shader level
            if (type == TypeDesc::TypeInt && ident == "range_checking"
                && Strutil::parse_int(h, ival) && ival >= 0)
                m_master->range_checking(ival != 0);
        }
        return;
    }
}



void
OSOReaderToMaster::codemarker (const char *name)
{
    m_sourcefile.clear();
    int nextop = (int) m_master->m_ops.size();

    codeend ();   // Mark the end spot, if we were parsing ops before

    m_codesection = ustring (name);
    m_codesym = m_master->findsymbol (m_codesection);
    if (m_codesym >= 0)
        m_master->symbol(m_codesym)->initbegin (nextop);
#if 0
    std::cerr << "Read code marker " << m_codesection
              << " at instruction " << nextop
              << ", sym " << m_codesym
              << " (" << (m_codesym >= 0 ? m_master->symbol(m_codesym)->name() : ustring()) << ")"
              << "\n";
#endif
    if (m_codesection == "___main___") {
        m_master->m_maincodebegin = nextop;
    } else if (m_codesym < 0) {
        m_shadingsys.errorf("Parsing shader %s: don't know what to do with code section \"%s\"",
                            m_master->shadername(), name);
        m_errors = true;
    }
}



void
OSOReaderToMaster::codeend ()
{
    int nextop = (int) m_master->m_ops.size();
    if (m_codesym >= 0) {
        // If we were previously chalking up the code to init ops for a
        // symbol, mark the end.
        m_master->symbol(m_codesym)->initend (nextop);
    } else if (m_codesection == "___main___") {
        // If we were previously reading ops for the ___main___ entry
        // point, mark its end properly.
        m_master->m_maincodeend = nextop;
    }
}



void
OSOReaderToMaster::instruction (int /*label*/, const char *opcode)
{
    ustring uopcode (opcode);
    Opcode op (uopcode, m_codesection);
    m_master->m_ops.push_back (op);
    m_firstarg = m_master->m_args.size();
    m_nargs = 0;
    m_reading_instruction = true;
    const OpDescriptor *od = m_shadingsys.op_descriptor (uopcode);
    if (od) {
        // Replace the name in case it was aliased for compatibility
        uopcode = od->name;
    } else {
        m_shadingsys.errorf("Parsing shader \"%s\": instruction \"%s\" is not known. Maybe compiled with a too-new oslc?",
                            m_master->shadername(), opcode);
        m_errors = true;
    }
}



void
OSOReaderToMaster::instruction_arg (const char *name)
{
    ustring argname (name);
    UstringIntMap::const_iterator found = m_symmap.find (argname);
    if (found != m_symmap.end()) {
        m_master->m_args.push_back (found->second);
        ++m_nargs;
        return;
    }
    m_shadingsys.errorf("Parsing shader %s: unknown arg %s",
                        m_master->shadername(), name);
    m_errors = true;
}



void
OSOReaderToMaster::instruction_jump (int target)
{
    m_master->m_ops.back().add_jump (target);
}



void
OSOReaderToMaster::instruction_end ()
{
    m_master->m_ops.back().set_args (m_firstarg, m_nargs);
    m_master->m_ops.back().source (m_sourcefile, m_sourceline);
    m_reading_instruction = false;
}



ShaderMaster::ref
ShadingSystemImpl::loadshader (string_view cname)
{
    if (Strutil::ends_with (cname, ".oso"))
        cname.remove_suffix (4);   // strip superfluous .oso
    if (! cname.size()) {
        error ("Attempt to load shader with empty name \"\".");
        return NULL;
    }
    ++m_stat_shaders_requested;
    ustring name (cname);
    lock_guard guard (m_mutex);  // Thread safety
    ShaderNameMap::const_iterator found = m_shader_masters.find (name);
    if (found != m_shader_masters.end()) {
        // if (debug())
        //     infof("Found %s in shader_masters", name);
        // Already loaded this shader, return its reference
        return (*found).second;
    }

    // Not found in the map
    OSOReaderToMaster oso (*this);
    bool testcwd = m_searchpath_dirs.empty();  // test "." if there's no searchpath
    std::string filename = OIIO::Filesystem::searchpath_find (name.string() + ".oso",
                                                        m_searchpath_dirs,
                                                        testcwd);
    if (filename.empty ()) {
        errorf("No .oso file could be found for shader \"%s\"", name);
        return NULL;
    }
    OIIO::Timer timer;
    bool ok = oso.parse_file (filename);
    ShaderMaster::ref r = ok ? oso.master() : nullptr;
    m_shader_masters[name] = r;
    double loadtime = timer();
    {
        spin_lock lock (m_stat_mutex);
        m_stat_master_load_time += loadtime;
    }
    if (ok) {
        ++m_stat_shaders_loaded;
        infof("Loaded \"%s\" (took %s)", filename,
              Strutil::timeintervalformat(loadtime, 2));
        OSL_DASSERT (r);
        r->resolve_syms ();
        // if (debug()) {
        //     std::string s = r->print ();
        //     if (s.length())
        //         infof("%s", s);
        // }
    } else {
        errorf("Unable to read \"%s\"", filename);
    }

    return r;
}



bool
ShadingSystemImpl::LoadMemoryCompiledShader (string_view shadername,
                                             string_view buffer)
{
    if (! shadername.size()) {
        error ("Attempt to load shader with empty name \"\".");
        return false;
    }
    if (! buffer.size()) {
        errorf ("Attempt to load shader \"%s\" with empty OSO data.", shadername);
        return false;
    }

    ustring name (shadername);
    lock_guard guard (m_mutex);  // Thread safety
    ShaderNameMap::const_iterator found = m_shader_masters.find (name);
    if (found != m_shader_masters.end() && ! allow_shader_replacement()) {
        if (debug())
            infof("Preload shader %s already exists in shader_masters", name);
        return false;
    }

    // Not found in the map
    OSOReaderToMaster reader (*this);
    OIIO::Timer timer;
    bool ok = reader.parse_memory (buffer);
    ShaderMaster::ref r = ok ? reader.master() : nullptr;
    m_shader_masters[name] = r;
    double loadtime = timer();
    {
        spin_lock lock (m_stat_mutex);
        m_stat_master_load_time += loadtime;
    }
    if (ok) {
        ++m_stat_shaders_loaded;
        infof("Loaded \"%s\" (took %s)", shadername,
              Strutil::timeintervalformat(loadtime, 2));
        OSL_DASSERT (r);
        r->resolve_syms ();
        // if (debug()) {
        //     std::string s = r->print ();
        //     if (s.length())
        //         infof ("%s", s);
        // }
    } else {
        errorf("Unable to parse preloaded shader \"%s\"", shadername);
    }

    return true;
}



}; // namespace pvt
OSL_NAMESPACE_EXIT
