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
#include <sstream>

#include "boost/foreach.hpp"

#include "OpenImageIO/dassert.h"
#include "OpenImageIO/thread.h"
#include "OpenImageIO/strutil.h"
#include "OpenImageIO/sysutil.h"

#include "oslexec_pvt.h"
#include "dual.h"
#include "oslops.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {   // OSL::pvt



ShadingExecution::ShadingExecution ()
    : m_context(NULL), m_instance(NULL), m_master(NULL),
      m_npoints_bound(0),
      m_last_instance_id(-1)
{
}



ShadingExecution::~ShadingExecution ()
{
}



void
ShadingExecution::error (const char *message, ...)
{
    va_list ap;
    va_start (ap, message);
    std::string e = Strutil::vformat (message, ap);
    m_shadingsys->error (e);
    va_end (ap);
}



void
ShadingExecution::warning (const char *message, ...)
{
    va_list ap;
    va_start (ap, message);
    std::string e = Strutil::vformat (message, ap);
    m_shadingsys->warning (e);
    va_end (ap);
}



void
ShadingExecution::info (const char *message, ...)
{
    va_list ap;
    va_start (ap, message);
    std::string e = Strutil::vformat (message, ap);
    m_shadingsys->info (e);
    va_end (ap);
}



void
ShadingExecution::message (const char *message, ...)
{
    va_list ap;
    va_start (ap, message);
    std::string e = Strutil::vformat (message, ap);
    m_shadingsys->message (e);
    va_end (ap);
}



std::string
ShadingExecution::format_symbol (const std::string &format,
                                 Symbol &sym, int whichpoint)
{
    if (sym.typespec().is_closure()) {
        // Special case for printing closures
        std::stringstream stream;
        print_closure (stream, ((const ClosureColor **)sym.data())[whichpoint], shadingsys());
        return stream.str ();
    }
    TypeDesc type = sym.typespec().simpletype();
    const char *data = (const char *)sym.data() + whichpoint * sym.step();
    std::string s;
    int n = type.numelements() * type.aggregate;
    for (int i = 0;  i < n;  ++i) {
        // FIXME -- type checking here!!!!
        if (type.basetype == TypeDesc::FLOAT)
            s += Strutil::format (format.c_str(), ((const float *)data)[i]);
        else if (type.basetype == TypeDesc::INT)
            s += Strutil::format (format.c_str(), ((const int *)data)[i]);
        else if (type.basetype == TypeDesc::STRING)
            s += Strutil::format (format.c_str(), ((const ustring *)data)[i].c_str());
        if (n > 1 && i < n-1)
            s += ' ';
    }
    if (m_debug && sym.has_derivs() && // sym.is_varying() &&
            type.basetype == TypeDesc::FLOAT) {
        s += " {dx=";
        data += sym.deriv_step ();
        for (int i = 0;  i < n;  ++i)
            s += Strutil::format ("%g%c", ((const float *)data)[i],
                                  i < n-1 ? ' ' : ',');
        s += " dy=";
        data += sym.deriv_step ();
        for (int i = 0;  i < n;  ++i)
            s += Strutil::format ("%g%c", ((const float *)data)[i],
                                  i < n-1 ? ' ' : '}');
    }
    return s;
}



void
ShadingExecution::get_matrix (Matrix44 &result, ustring from, int whichpoint)
{
    if (from == Strings::common || from == m_shadingsys->commonspace_synonym()) {
        result.makeIdentity ();
        return;
    }
    ShaderGlobals *globals = m_context->m_globals;
    if (from == Strings::shader) {
        m_renderer->get_matrix (result, globals->shader2common[whichpoint],
                                globals->time[whichpoint]);
        return;
    }
    if (from == Strings::object) {
        m_renderer->get_matrix (result, globals->object2common[whichpoint],
                                globals->time[whichpoint]);
        return;
    }
    bool ok = m_renderer->get_matrix (result, from, globals->time[whichpoint]);
    if (! ok) {
        result.makeIdentity ();
        error ("Could not get matrix '%s'", from.c_str());
    }
}



bool 
ShadingExecution::get_renderer_array_attribute(void *renderstate, bool derivatives, ustring object, 
                                               TypeDesc type, ustring name, 
                                               int index, void *val)
{
    return m_renderer->get_array_attribute(renderstate, derivatives, object, type, name, index, val);
}



bool 
ShadingExecution::get_renderer_attribute(void *renderstate, bool derivatives, ustring object, 
                                         TypeDesc type, ustring name, void *val)
{
    return m_renderer->get_attribute(renderstate, derivatives, object, type, name, val);
}



bool
ShadingExecution::get_renderer_userdata(Runflag *runflags, int npoints, bool derivatives, 
                                        ustring name, TypeDesc type, 
                                        void *renderstate, int renderstate_stepsize, 
                                        void *val, int val_stepsize)
{
   return m_renderer->get_userdata(runflags, npoints, derivatives, name, type, 
                                   renderstate, renderstate_stepsize,
                                   val, val_stepsize);
}



bool
ShadingExecution::renderer_has_userdata(ustring name, TypeDesc type, void *renderstate)
{
    return m_renderer->has_userdata(name, type, renderstate);
}



void
ShadingExecution::get_inverse_matrix (Matrix44 &result,
                                      ustring to, int whichpoint)
{
    if (to == Strings::common || to == m_shadingsys->commonspace_synonym()) {
        result.makeIdentity ();
        return;
    }
    ShaderGlobals *globals = m_context->m_globals;
    if (to == Strings::shader) {
        m_renderer->get_inverse_matrix (result, globals->shader2common[whichpoint],
                                        globals->time[whichpoint]);
        return;
    }
    if (to == Strings::object) {
        m_renderer->get_inverse_matrix (result, globals->object2common[whichpoint],
                                        globals->time[whichpoint]);
        return;
    }
    bool ok = m_renderer->get_inverse_matrix (result, to, globals->time[whichpoint]);
    if (! ok) {
        result.makeIdentity ();
        error ("Could not get matrix '%s'", to.c_str());
    }
}



void
ShadingExecution::get_matrix (Matrix44 &result, ustring from,
                              ustring to, int whichpoint)
{
    Matrix44 Mfrom, Mto;
    get_matrix (Mfrom, from, whichpoint);
    get_inverse_matrix (Mto, to, whichpoint);
    result = Mfrom * Mto;
}




}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
