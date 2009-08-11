/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
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

#include <boost/algorithm/string.hpp>

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/dassert.h"
#include "OpenImageIO/thread.h"
#include "OpenImageIO/filesystem.h"

#include "oslexec_pvt.h"
using namespace OSL;
using namespace OSL::pvt;



#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {


ShadingSystem *
ShadingSystem::create (RendererServices *renderer,
                       TextureSystem *texturesystem)
{
    // Doesn't need a shared cache
    ShadingSystemImpl *ts = new ShadingSystemImpl (renderer, texturesystem);
#ifdef DEBUG
    std::cout << "creating new ShadingSystem " << (void *)ts << "\n";
#endif
    return ts;
}



void
ShadingSystem::destroy (ShadingSystem *x)
{
    delete (ShadingSystemImpl *) x;
}



ShadingSystem::ShadingSystem ()
{
}



ShadingSystem::~ShadingSystem ()
{
}



namespace Strings {

ustring camera ("camera"), common ("common");
ustring object ("object"), shader ("shader");

};



namespace pvt {   // OSL::pvt



ShadingSystemImpl::ShadingSystemImpl (RendererServices *renderer,
                                      TextureSystem *texturesystem)
    : m_renderer(renderer), m_texturesys(texturesystem),
      m_in_group (false), m_statslevel (0), m_debug (false),
      m_global_heap_total (0)
{
    m_stat_shaders_loaded = 0;
    m_stat_shaders_requested = 0;
    init_global_heap_offsets ();

#if 0
    // If client didn't supply renderer services, create a default one
    if (! m_renderer) {
        m_renderer = NULL;
        ASSERT (m_renderer);
    }
#endif

    // If client didn't supply a texture system, create a new one
    if (! m_texturesys) {
        m_texturesys = TextureSystem::create (true /* shared */);
        ASSERT (m_texturesys);
        // Make some good guesses about default options
        m_texturesys->attribute ("automip",  1);
        m_texturesys->attribute ("autotile", 1);
    }
}



ShadingSystemImpl::~ShadingSystemImpl ()
{
    printstats ();
    // N.B. just let m_texsys go -- if we asked for one to be created,
    // we asked for a shared one.
}



bool
ShadingSystemImpl::attribute (const std::string &name, TypeDesc type,
                              const void *val)
{
    lock_guard guard (m_mutex);  // Thread safety
    if (name == "searchpath:shader" && type == TypeDesc::STRING) {
        m_searchpath = ustring (*(const char **)val);
        Filesystem::searchpath_split (m_searchpath, m_searchpath_dirs);
        return true;
    }
    if (name == "statistics:level" && type == TypeDesc::INT) {
        m_statslevel = *(const int *)val;
        return true;
    }
    if (name == "debug" && type == TypeDesc::INT) {
        m_debug = *(const int *)val;
        return true;
    }
    return false;
}



bool
ShadingSystemImpl::getattribute (const std::string &name, TypeDesc type,
                                 void *val)
{
    lock_guard guard (m_mutex);  // Thread safety
    if (name == "searchpath:shader" && type == TypeDesc::INT) {
        *(const char **)val = m_searchpath.c_str();
        return true;
    }
    if (name == "statistics:level" && type == TypeDesc::INT) {
        *(int *)val = m_statslevel;
        return true;
    }
    if (name == "debug" && type == TypeDesc::INT) {
        *(int *)val = m_debug;
        return true;
    }
    return false;
}



std::string
ShadingSystemImpl::geterror () const
{
    std::string e;
    std::string *errptr = m_errormessage.get ();
    if (errptr) {
        e = *errptr;
        errptr->clear ();
    }
    return e;
}



void
ShadingSystemImpl::error (const char *message, ...)
{
    std::string *errptr = m_errormessage.get ();
    if (! errptr) {
        errptr = new std::string;
        m_errormessage.reset (errptr);
    }
    ASSERT (errptr != NULL);
    if (errptr->size())
        *errptr += '\n';
    va_list ap;
    va_start (ap, message);
    *errptr += Strutil::vformat (message, ap);
    va_end (ap);
}



std::string
ShadingSystemImpl::getstats (int level) const
{
    if (level <= 0)
        return "";
    std::ostringstream out;
    out << "OSL ShadingSystem statistics (" << (void*)this << ")\n";
    out << "  Shaders:\n";
    out << "    Requested: " << m_stat_shaders_requested << "\n";
    out << "    Loaded:    " << m_stat_shaders_loaded << "\n";
    out << "    Masters:   " << m_stat_shaders_loaded << "\n";
    out << "    Instances: " << m_stat_instances << "\n";
    out << "  Shading contexts: " << m_stat_contexts << "\n";
    // FIXME
    return out.str();
}



void
ShadingSystemImpl::printstats () const
{
    if (m_statslevel == 0)
        return;
    std::cout << getstats (m_statslevel) << "\n\n";
}



void
ShadingSystemImpl::Parameter (const char *name, TypeDesc t, const void *val)
{
    m_pending_params.push_back (ParamRef (ustring(name), t, val));
}



void
ShadingSystemImpl::ShaderGroupBegin (void)
{
    if (m_in_group) {
        error ("Nested ShaderGroupBegin() calls");
        return;
    }
    m_in_group = true;
    m_group_use = ShadUseUnknown;
}



void
ShadingSystemImpl::ShaderGroupEnd (void)
{
    m_in_group = false;
    m_group_use = ShadUseUnknown;
}



void
ShadingSystemImpl::Shader (const char *shaderusage,
                           const char *shadername,
                           const char *layername)
{
    ShaderMaster::ref master = loadshader (shadername);
    if (! master) {
        // FIXME -- some kind of error return?
        return;
    }

    ShaderUse use = shaderuse_from_name (shaderusage);
    if (use == ShadUseUnknown) {
        error ("Unknown shader usage '%s'", shaderusage);
        return;
    }

    // Make sure we have a current attrib state
    if (! m_curattrib)
        m_curattrib.reset (new ShadingAttribState);

    ShaderInstanceRef instance (new ShaderInstance (master, layername));
    instance->parameters (m_pending_params);
    m_pending_params.clear ();

    ShaderGroup &shadergroup (m_curattrib->shadergroup (use));
    if (! m_in_group || m_group_use == ShadUseUnknown) {
        // A singleton, or the first in a group
        shadergroup.clear ();
    }
    if (m_in_group) {
        if (m_group_use == ShadUseUnknown) {  // First shader in group
            m_group_use = use;
        } else if (use != m_group_use) {
            error ("Shader usage '%s' does not match current group (%s)",
                   shaderusage, shaderusename (m_group_use));
            return;
        }
    }

    shadergroup.append (instance);
    m_curattrib->calc_heapsize ();
    // FIXME -- check for duplicate layer name within the group?
}



void
ShadingSystemImpl::ConnectShaders (const char *srclayer, const char *srcparam,
                                   const char *dstlayer, const char *dstparam)
{
    if (! m_in_group) {
        error ("ConectShaders can only be called within ShaderGroupBegin/End");
        return;
    }
    // FIXME
}



ShadingAttribStateRef
ShadingSystemImpl::state () const
{
    return m_curattrib;
}



void
ShadingSystemImpl::clear_state ()
{
    m_curattrib.reset (new ShadingAttribState);
}



shared_ptr<ShadingContext>
ShadingSystemImpl::get_context ()
{
    return shared_ptr<ShadingContext> (new ShadingContext (*this));
}




void
ShadingSystemImpl::init_global_heap_offsets ()
{
    lock_guard lock (m_mutex);
    if (m_global_heap_total > 0)
        return;   // Already initialized

    const int triple_size = sizeof (Vec3);
    m_global_heap_offsets[ustring("P")] = m_global_heap_total;
    m_global_heap_total += triple_size;
    m_global_heap_offsets[ustring("I")] = m_global_heap_total;
    m_global_heap_total += triple_size;
    m_global_heap_offsets[ustring("N")] = m_global_heap_total;
    m_global_heap_total += triple_size;
    m_global_heap_offsets[ustring("Ng")] = m_global_heap_total;
    m_global_heap_total += triple_size;
    m_global_heap_offsets[ustring("dPdu")] = m_global_heap_total;
    m_global_heap_total += triple_size;
    m_global_heap_offsets[ustring("dPdv")] = m_global_heap_total;
    m_global_heap_total += triple_size;

    m_global_heap_offsets[ustring("u")] = m_global_heap_total;
    m_global_heap_total += sizeof (float);
    m_global_heap_offsets[ustring("v")] = m_global_heap_total;
    m_global_heap_total += sizeof (float);
    m_global_heap_offsets[ustring("time")] = m_global_heap_total;
    m_global_heap_total += sizeof (float);
    m_global_heap_offsets[ustring("dtime")] = m_global_heap_total;
    m_global_heap_total += sizeof (float);
}



int
ShadingSystemImpl::global_heap_offset (ustring name)
{
    std::map<ustring,int>::const_iterator f = m_global_heap_offsets.find (name);
    return f != m_global_heap_offsets.end() ? f->second : -1;
}


}; // namespace pvt
}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
