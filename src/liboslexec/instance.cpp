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

#include <boost/foreach.hpp>

#include "OpenImageIO/dassert.h"

#include "oslexec_pvt.h"



namespace OSL {

namespace pvt {   // OSL::pvt


ShaderInstance::ShaderInstance (ShaderMaster::ref master,
                                const char *layername) 
    : m_master(master), m_symbols(m_master->m_symbols),
      m_layername(layername), m_heapsize(-1 /*uninitialized*/),
      m_heapround(0), m_numclosures(-1), m_heap_size_calculated(0),
      m_writes_globals(false)
{
    static int next_id = 0; // We can statically init an int, not an atomic
    m_id = ++(*(atomic_int *)&next_id);
}



void
ShaderInstance::parameters (const ParamValueList &params)
{
    m_iparams = m_master->m_idefaults;
    m_fparams = m_master->m_fdefaults;
    m_sparams = m_master->m_sdefaults;
    m_symbols = m_master->m_symbols;
    BOOST_FOREACH (const ParamValue &p, params) {
        if (shadingsys().debug())
            shadingsys().info (" PARAMETER %s %s",
                               p.name().c_str(), p.type().c_str());
        int i = m_master->findparam (p.name());
        if (i >= 0) {
            Symbol *s = symbol(i);
            // don't allow assignment of closures
            if (s->typespec().is_closure()) {
                shadingsys().warning ("skipping assignment of closure: %s", s->name().c_str());
                continue;
            }
            // check type of parameter and matching symbol
            if (s->typespec().simpletype() != p.type()) {
                shadingsys().warning ("attempting to set parameter with wrong type: %s (exepected '%s', received '%s')", s->name().c_str(), s->typespec().c_str(), p.type().c_str());
                continue;
            }
            s->valuesource (Symbol::InstanceVal);
            if (s->typespec().simpletype().basetype == TypeDesc::INT) {
                memcpy (&m_iparams[s->dataoffset()], p.data(),
                        s->typespec().simpletype().size());
            } else if (s->typespec().simpletype().basetype == TypeDesc::FLOAT) {
                memcpy (&m_fparams[s->dataoffset()], p.data(),
                        s->typespec().simpletype().size());
            } else if (s->typespec().simpletype().basetype == TypeDesc::STRING) {
                memcpy (&m_sparams[s->dataoffset()], p.data(),
                        s->typespec().simpletype().size());
            }
            if (shadingsys().debug())
                shadingsys().info ("    sym %s offset %llu address %p",
                        s->name().c_str(),
                        (unsigned long long)s->dataoffset(), s->data());
        }
        else {
            shadingsys().warning ("attempting to set nonexistent parameter: %s", p.name().c_str());
        }
    }
}



void
ShaderInstance::calc_heap_size ()
{
    // Because heap size may not be computed until mid-bind, we must
    // protect against multiple threads computing it on an Instance
    // simultaneously.
    static spin_mutex heap_size_mutex;
    spin_lock lock (heap_size_mutex);

    if (m_heap_size_calculated)
        return;   // Another thread did it before we got the lock

    if (shadingsys().debug())
        shadingsys().info ("calc_heapsize on %s", m_master->shadername().c_str());
    m_heapsize = 0;
    m_numclosures = 0;
    m_heapround = 0;
    m_writes_globals = false;
    BOOST_FOREACH (/*const*/ Symbol &s, m_symbols) {
        // Skip if the symbol is a type that doesn't need heap space
        if (s.symtype() == SymTypeConst /* || s.symtype() == SymTypeGlobal */)
            continue;

        // assume globals have derivs
        if (s.symtype() == SymTypeGlobal) {
            s.has_derivs (true);
            m_writes_globals |= s.everwritten ();
        }

#if 1
        // FIXME -- test code by assuming all locals, temps, and params
        // carry derivs
        if ((s.symtype() == SymTypeLocal || s.symtype() == SymTypeTemp ||
             s.symtype() == SymTypeParam || s.symtype() == SymTypeOutputParam) &&
                !s.typespec().is_closure() &&
                s.typespec().elementtype().is_floatbased())
            s.has_derivs (true);
#endif

        const TypeSpec &t (s.typespec());
        size_t size = s.size ();
        if (t.is_closure())
            ++m_numclosures;
        if (s.has_derivs())
            size *= 3;

        int pad = (int) shadingsys().align_padding (size);
        if (pad)
            m_heapround += pad;
        m_heapsize += size + pad;

        if (shadingsys().debug())
            shadingsys().info (" sym %s given %llu bytes on heap (including %llu padding)",
                               s.mangled().c_str(),
                               (unsigned long long)size,
                               (unsigned long long)pad);
    }
    if (shadingsys().debug()) {
        shadingsys().info (" Heap needed %llu, %d closures on the heap",
                           (unsigned long long)m_heapsize, m_numclosures);
        shadingsys().info (" Padding for alignment = %d", m_heapround);
        shadingsys().info (" Writes globals: %d", m_writes_globals);
    }

    m_heap_size_calculated = 1;
}



size_t
ShaderInstance::heapsize ()
{
    if (! heap_size_calculated ())
        calc_heap_size ();
    return (size_t) m_heapsize;
}



size_t
ShaderInstance::heapround ()
{
    if (! heap_size_calculated ())
        calc_heap_size ();
    return (size_t) m_heapround;
}



size_t
ShaderInstance::numclosures ()
{
    if (! heap_size_calculated ())
        calc_heap_size ();
    return (size_t) m_numclosures;
}


}; // namespace pvt


void
ShadingAttribState::calc_heap_size ()
{
    // Because heap size may not be computed until mid-bind, we must
    // protect against multiple threads computing it on a
    // ShadingAttribState simultaneously.
    static spin_mutex heap_size_mutex;
    spin_lock lock (heap_size_mutex);

    if (m_heap_size_calculated)
        return;   // Another thread did it before we got the lock

    m_heapsize = 0;
    m_heapround = 0;
    m_numclosures = 0;
    for (int i = 0;  i < (int)OSL::pvt::ShadUseLast;  ++i) {
        for (int lay = 0;  lay < m_shaders[i].nlayers();  ++lay) {
            m_heapsize += m_shaders[i][lay]->heapsize ();
            m_heapround += m_shaders[i][lay]->heapround ();
            m_numclosures += m_shaders[i][lay]->numclosures ();
        }
    }
    
    m_heap_size_calculated = 1;
}



size_t
ShadingAttribState::heapsize ()
{
    if (! heap_size_calculated ())
        calc_heap_size ();
    return (size_t) m_heapsize;
}


size_t
ShadingAttribState::heapround ()
{
    if (! heap_size_calculated ())
        calc_heap_size ();
    return (size_t) m_heapround;
}


size_t
ShadingAttribState::numclosures ()
{
    if (! heap_size_calculated ())
        calc_heap_size ();
    return (size_t) m_numclosures;
}



}; // namespace OSL
