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

#include <boost/algorithm/string.hpp>

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/dassert.h"
#include "OpenImageIO/timer.h"
#include "OpenImageIO/thread.h"
#include "OpenImageIO/filesystem.h"
#ifdef OIIO_NAMESPACE
namespace Filesystem = OIIO::Filesystem;
using OIIO::Timer;
#endif




OSL_NAMESPACE_ENTER

namespace pvt {   // OSL::pvt


/// Custom subclass of OSOReader that provide callbacks that set all the
/// right fields in the ShaderMaster.
class OSOReaderToMaster : public OSOReader
{
public:
    OSOReaderToMaster (ShadingSystemImpl &shadingsys)
        : OSOReader (&shadingsys.errhandler()), m_shadingsys (shadingsys),
          m_master (new ShaderMaster (shadingsys)), m_reading_instruction(false)
      { }
    virtual ~OSOReaderToMaster () { }
    virtual bool parse (const std::string &filename);
    virtual void version (const char *specid, int major, int minor);
    virtual void shader (const char *shadertype, const char *name);
    virtual void symbol (SymType symtype, TypeSpec typespec, const char *name);
    virtual void symdefault (int def);
    virtual void symdefault (float def);
    virtual void symdefault (const char *def);
    virtual void hint (const char *hintstring);
    virtual void codemarker (const char *name);
    virtual void codeend ();
    virtual void instruction (int label, const char *opcode);
    virtual void instruction_arg (const char *name);
    virtual void instruction_jump (int target);
    virtual void instruction_end ();

    ShaderMaster::ref master () const { return m_master; }

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
};



bool
OSOReaderToMaster::parse (const std::string &filename)
{
    m_master->m_osofilename = filename;
    m_master->m_maincodebegin = 0;
    m_master->m_maincodeend = 0;
    m_codesection.clear ();
    m_codesym = -1;
    return OSOReader::parse (filename);
}



void
OSOReaderToMaster::version (const char *specid, int major, int minor)
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



// Helper function to expand vec by 'size' elements, initializing them to 0.
template<class T>
inline void
expand (std::vector<T> &vec, size_t size)
{
    vec.resize (vec.size() + size, T(0));
}



void
OSOReaderToMaster::symbol (SymType symtype, TypeSpec typespec, const char *name)
{
    Symbol sym (ustring(name), typespec, symtype);
    TypeDesc t = typespec.simpletype();
    if (sym.symtype() == SymTypeParam || sym.symtype() == SymTypeOutputParam) {
        // Skip structs for now, they're just placeholders
        if (typespec.is_structure()) {
        }
        else if (typespec.simpletype().basetype == TypeDesc::FLOAT) {
            sym.dataoffset ((int) m_master->m_fdefaults.size());
            expand (m_master->m_fdefaults, t.aggregate * t.numelements());
        } else if (typespec.simpletype().basetype == TypeDesc::INT) {
            sym.dataoffset ((int) m_master->m_idefaults.size());
            expand (m_master->m_idefaults, t.aggregate * t.numelements());
        } else if (typespec.simpletype().basetype == TypeDesc::STRING) {
            sym.dataoffset ((int) m_master->m_sdefaults.size());
            expand (m_master->m_sdefaults, t.aggregate * t.numelements());
        } else if (typespec.is_closure()) {
            // Closures are pointers, so we allocate a string default taking
            // adventage of their default being NULL as well.
            sym.dataoffset ((int) m_master->m_sdefaults.size());
            expand (m_master->m_sdefaults, t.aggregate * t.numelements());
        } else {
            ASSERT (0 && "unexpected type");
        }
    }
    if (sym.symtype() == SymTypeConst) {
        if (typespec.simpletype().basetype == TypeDesc::FLOAT) {
            sym.dataoffset ((int) m_master->m_fconsts.size());
            expand (m_master->m_fconsts, t.aggregate * t.numelements());
        } else if (typespec.simpletype().basetype == TypeDesc::INT) {
            sym.dataoffset ((int) m_master->m_iconsts.size());
            expand (m_master->m_iconsts, t.aggregate * t.numelements());
        } else if (typespec.simpletype().basetype == TypeDesc::STRING) {
            sym.dataoffset ((int) m_master->m_sconsts.size());
            expand (m_master->m_sconsts, t.aggregate * t.numelements());
        } else {
            ASSERT (0 && "unexpected type");
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
    // Start the index at which we add specified defaults
    m_sym_default_index = 0;
}



void
OSOReaderToMaster::symdefault (int def)
{
    ASSERT (m_master->m_symbols.size() && "symdefault but no sym");
    Symbol &sym (m_master->m_symbols.back());
    size_t offset = sym.dataoffset() + m_sym_default_index;
    ++m_sym_default_index;
    if (sym.symtype() == SymTypeParam || sym.symtype() == SymTypeOutputParam) {
        if (sym.typespec().simpletype().basetype == TypeDesc::FLOAT)
            m_master->m_fdefaults[offset] = (float)def;
        else if (sym.typespec().simpletype().basetype == TypeDesc::INT)
            m_master->m_idefaults[offset] = def;
        else {
            ASSERT (0 && "unexpected type");
        }
    } else if (sym.symtype() == SymTypeConst) {
        if (sym.typespec().simpletype().basetype == TypeDesc::FLOAT)
            m_master->m_fconsts[offset] = (float)def;
        else if (sym.typespec().simpletype().basetype == TypeDesc::INT)
            m_master->m_iconsts[offset] = def;
        else {
            ASSERT (0 && "unexpected type");
        }
    }
}



void
OSOReaderToMaster::symdefault (float def)
{
    ASSERT (m_master->m_symbols.size() && "symdefault but no sym");
    Symbol &sym (m_master->m_symbols.back());
    size_t offset = sym.dataoffset() + m_sym_default_index;
    ++m_sym_default_index;
    if (sym.symtype() == SymTypeParam || sym.symtype() == SymTypeOutputParam) {
        if (sym.typespec().simpletype().basetype == TypeDesc::FLOAT)
            m_master->m_fdefaults[offset] = def;
        else {
            ASSERT (0 && "unexpected type");
        }
    } else if (sym.symtype() == SymTypeConst) {
        if (sym.typespec().simpletype().basetype == TypeDesc::FLOAT)
            m_master->m_fconsts[offset] = def;
        else {
            ASSERT (0 && "unexpected type");
        }
    }
}



void
OSOReaderToMaster::symdefault (const char *def)
{
    ASSERT (m_master->m_symbols.size() && "symdefault but no sym");
    Symbol &sym (m_master->m_symbols.back());
    size_t offset = sym.dataoffset() + m_sym_default_index;
    ++m_sym_default_index;
    if (sym.symtype() == SymTypeParam || sym.symtype() == SymTypeOutputParam) {
        if (sym.typespec().simpletype().basetype == TypeDesc::STRING)
            m_master->m_sdefaults[offset] = ustring(def);
        else {
            ASSERT (0 && "unexpected type");
        }
    } else if (sym.symtype() == SymTypeConst) {
        if (sym.typespec().simpletype().basetype == TypeDesc::STRING)
            m_master->m_sconsts[offset] = ustring(def);
        else {
            ASSERT (0 && "unexpected type");
        }
    }
}



inline bool
starts_with (const std::string &source, const std::string &pattern)
{
    return ! strncmp (source.c_str(), pattern.c_str(), pattern.length());
}



// If the string 'source' begins with 'pattern', erase the pattern from
// the start of source and return true.  Otherwise, do not alter source
// and return false.
inline bool
extract_prefix (std::string &source, const std::string &pattern)
{
    if (starts_with (source, pattern)) {
        source.erase (0, pattern.length());
        return true;
    }
    return false;
}



// Return the prefix of source that doesn't contain any characters in
// 'stop', erase that prefix from source (up to and including the stop
// character.  Also, the returned string is trimmed of leading and trailing
// spaces if 'do_trim' is true.
static std::string
readuntil (std::string &source, const std::string &stop, bool do_trim=false)
{
    size_t e = source.find_first_of (stop);
    if (e == source.npos)
        return std::string ();
    std::string r (source, 0, e);
    source.erase (0, e == source.npos ? e : e+1);
    if (do_trim)
        boost::trim (r);
    return r;
}



void
OSOReaderToMaster::hint (const char *hintstring)
{
    std::string h (hintstring);
    if (extract_prefix (h, "%filename{\"")) {
        m_sourcefile = readuntil (h, "\"");
        return;
    }
    if (extract_prefix (h, "%line{")) {
        m_sourceline = atoi (h.c_str());
        return;
    }
    if (extract_prefix (h, "%structfields{")) {
        ASSERT (m_master->m_symbols.size() && "structfields hint but no sym");
        Symbol &sym (m_master->m_symbols.back());
        StructSpec *structspec = sym.typespec().structspec();
        if (structspec->numfields() == 0) {
            while (1) {
                std::string afield = readuntil (h, ",}", true);
                if (! afield.length())
                    break;
//                std::cerr << " struct field " << afield << "\n";
                structspec->add_field (TypeSpec(), ustring(afield));
            }
        }
        return;
    }
    if (extract_prefix (h, "%mystructfield{")) {
        ASSERT (m_master->m_symbols.size() && "mystructfield hint but no sym");
        Symbol &sym (m_master->m_symbols.back());
        sym.fieldid (atoi(h.c_str()+15));
        return;
    }
    if (extract_prefix (h, "%read{")) {
        ASSERT (m_master->m_symbols.size() && "read hint but no sym");
        Symbol &sym (m_master->m_symbols.back());
        int first, last;
        sscanf (h.c_str(), "%d,%d", &first, &last);
        sym.set_read (first, last);
        return;
    }
    if (extract_prefix (h, "%write{")) {
        ASSERT (m_master->m_symbols.size() && "write hint but no sym");
        Symbol &sym (m_master->m_symbols.back());
        int first, last;
        sscanf (h.c_str(), "%d,%d", &first, &last);
        sym.set_write (first, last);
        return;
    }
    if (extract_prefix(h, "%argrw{")) {
        const char* str = h.c_str();
        ASSERT(*str == '\"');
        str++; // skip open quote
        size_t i = 0;
        for (; *str != '\"'; i++, str++) {
            ASSERT(*str == 'r' || *str == 'w' || *str == 'W' || *str == '-');
            m_master->m_ops.back().argwrite(i, *str == 'w' || *str =='W');
            m_master->m_ops.back().argread(i, *str == 'r' || *str =='W');
        }
        ASSERT(m_nargs == i);
    }
    if (extract_prefix(h, "%argderivs{")) {
        while (1) {
            std::string afield = readuntil (h, ",}", true);
            if (! afield.length())
                break;
            int arg = atoi (afield.c_str());
            if (arg >= 0)
                m_master->m_ops.back().argtakesderivs (arg, true);
        }
    }
    if (extract_prefix (h, "%meta{") && m_master->m_symbols.size()) {
        Symbol &sym (m_master->m_symbols.back());
        int lockval = -1;
        int ok = sscanf (h.c_str(), " int , lockgeom , %d", &lockval);
        if (ok)
            sym.lockgeom (lockval);
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
        m_shadingsys.error ("Parsing shader %s: don't know what to do with code section \"%s\"",
                            m_master->shadername().c_str(), name);
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
    } else if (m_codesection && m_codesection == "___main___") {
        // If we were previously reading ops for the ___main___ entry
        // point, mark its end properly.
        m_master->m_maincodeend = nextop;
    }
}



void
OSOReaderToMaster::instruction (int label, const char *opcode)
{
    Opcode op (ustring(opcode), m_codesection);
    m_master->m_ops.push_back (op);
    m_firstarg = m_master->m_args.size();
    m_nargs = 0;
    m_reading_instruction = true;
}



void
OSOReaderToMaster::instruction_arg (const char *name)
{
    ustring argname (name);
    for (size_t i = 0;  i < m_master->m_symbols.size();  ++i) {
        if (m_master->m_symbols[i].name() == argname) {
            m_master->m_args.push_back (i);
            ++m_nargs;
            return;
        }
    }
    // ERROR! -- FIXME
//    m_master->m_args.push_back (0);  // FIXME
    m_shadingsys.error ("Parsing shader %s: unknown arg %s",
                        m_master->shadername().c_str(), name);
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
ShadingSystemImpl::loadshader (const char *cname)
{
    if (! cname || ! cname[0]) {
        error ("Attempt to load shader with empty name \"\".");
        return NULL;
    }
    ++m_stat_shaders_requested;
    ustring name (cname);
    lock_guard guard (m_mutex);  // Thread safety
    ShaderNameMap::const_iterator found = m_shader_masters.find (name);
    if (found != m_shader_masters.end()) {
        if (debug())
            info ("Found %s in shader_masters", name.c_str());
        // Already loaded this shader, return its reference
        return (*found).second;
    }

    // Not found in the map
    OSOReaderToMaster oso (*this);
    std::string filename = Filesystem::searchpath_find (name.string() + ".oso",
                                                        m_searchpath_dirs);
    if (filename.empty ()) {
        // FIXME -- error
        error ("No .oso file could be found for shader \"%s\"", name.c_str());
        return NULL;
    }
    Timer timer;
    bool ok = oso.parse (filename);
    ShaderMaster::ref r = ok ? oso.master() : NULL;
    m_shader_masters[name] = r;
    if (ok) {
        ++m_stat_shaders_loaded;
        info ("Loaded \"%s\" (took %s)", filename.c_str(), Strutil::timeintervalformat(timer(), 2).c_str());
    } else {
        error ("Unable to read \"%s\"", filename.c_str());
    }
    // FIXME -- catch errors

    if (r) {
        r->resolve_syms ();
    }

    if (r && m_debug) {
        std::string s = r->print ();
        if (s.length())
            info ("%s", s.c_str());
    }

    return r;
}



}; // namespace pvt
OSL_NAMESPACE_EXIT
