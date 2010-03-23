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
      m_bound(false), m_debug(false), m_last_instance_id(-1)
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



void
ShadingExecution::error_arg_types ()
{
    std::stringstream out;
    const Opcode &op (this->op());
    const int *args = &(instance()->args()[0]);
    out << "Don't know how to compute "
        << sym(args[op.firstarg()]).typespec().string() << " " << op.opname() << " (";
    for (int i = 1;  i < op.nargs();  ++i) {
        if (i > 1)
            out << ", ";
        out << sym(args[op.firstarg()+i]).typespec().string();
    }
    out << ")";
    error ("%s", out.str().c_str());
}



void
ShadingExecution::bind (ShaderInstance *instance)
{
    ASSERT (! m_bound);  // avoid double-binding
    ASSERT (instance != NULL);

    m_shadingsys = &m_context->shadingsys ();
    m_debug = shadingsys()->debug();
    bool debugnan = m_shadingsys->debug_nan ();
    if (m_debug)
        shadingsys()->info ("bind ctx %p use %s layer %d", m_context,
                            shaderusename(shaderuse()), layer());
    ++m_context->m_binds;

    // Take various shortcuts if we are re-binding the same instance as
    // last time.
    bool rebind = (shadingsys()->allow_rebind() &&
                   instance->id() == m_last_instance_id &&
                   m_npoints_bound >= m_context->npoints());
    if (rebind) {
        ++m_context->m_rebinds;
    } else {
        m_instance = instance;
        m_master = instance->master ();
        m_renderer = m_shadingsys->renderer ();
        m_last_instance_id = instance->id ();
        m_npoints_bound = m_context->npoints ();
        ASSERT (m_master && m_context && m_shadingsys && m_renderer);
        // FIXME -- if the number of points we need now is <= last time
        // we bound to this context, we can even skip much of the work
        // below, and reuse all the heap offsets and pointers.  We can
        // do that optimization later.

        // Make a fresh copy of symbols from the instance.  Don't copy the
        // whole vector, which may do an element-by-element copy of each
        // Symbol.  We humans know that the definition of Symbol has no
        // elements that can't be memcpy'd, there is no allocated memory
        // that can leak, so we go the fast route and memcpy.
        m_symbols.resize (m_instance->m_instsymbols.size());
        if (m_symbols.size() /* && !rebind */)
            memcpy (&m_symbols[0], &m_instance->m_instsymbols[0], 
                    m_instance->m_instsymbols.size() * sizeof(Symbol));
    }

    m_npoints = m_context->npoints ();

    ShaderGlobals *globals (m_context->m_globals);

    // FIXME: bind the symbols -- get the syms ready and pointing to the
    // right place in the heap,, interpolate primitive variables, handle
    // connections, initialize all parameters
    BOOST_FOREACH (Symbol &sym, m_symbols) {
#if 0
        if (m_debug)
            m_shadingsys->info ("  bind %s, offset %d",
                                sym.mangled().c_str(), sym.dataoffset());
#endif
        if (sym.symtype() == SymTypeGlobal) {
            // FIXME -- reset sym's data pointer?

            // Instead of duplicating the logic for each possible global
            // variable, we just decode the name and store pointers to
            // the VaringRef's (for value, and for some cases,
            // derivatives) into valref, dxref, and dyref, then handle
            // them with a single logic block below.  Note that rather
            // than worry about the float and triple cases separately,
            // we just cast them all to VR<float> for now, since it's
            // just a pointer underneath anyway.
            VaryingRef<float> * valref = NULL, *dxref = NULL, *dyref = NULL;
            if (sym.name() == Strings::P) {
                valref = (VaryingRef<float>*) &globals->P;
                dxref = (VaryingRef<float>*) &globals->dPdx;
                dyref = (VaryingRef<float>*) &globals->dPdy;
            } else if (sym.name() == Strings::I) {
                valref = (VaryingRef<float>*) &globals->I;
                dxref = (VaryingRef<float>*) &globals->dIdx;
                dyref = (VaryingRef<float>*) &globals->dIdy;
            } else if (sym.name() == Strings::N) {
                valref = (VaryingRef<float>*) &globals->N;
            } else if (sym.name() == Strings::u) {
                valref = &globals->u;
                dxref = &globals->dudx;
                dyref = &globals->dudy;
            } else if (sym.name() == Strings::v) {
                valref = &globals->v;
                dxref = &globals->dvdx;
                dyref = &globals->dvdy;
            } else if (sym.name() == Strings::Ps) {
                valref = (VaryingRef<float>*) &globals->Ps;
                dxref = (VaryingRef<float>*) &globals->dPsdx;
                dyref = (VaryingRef<float>*) &globals->dPsdy;
            } else if (sym.name() == Strings::Ci) {
                valref = (VaryingRef<float>*) &globals->Ci;
            } else if (sym.name() == Strings::Ng) {
                valref = (VaryingRef<float>*) &globals->Ng;
            } else if (sym.name() == Strings::dPdu) {
                valref = (VaryingRef<float>*) &globals->dPdu;
            } else if (sym.name() == Strings::dPdv) {
                valref = (VaryingRef<float>*) &globals->dPdv;
            } else if (sym.name() == Strings::time) {
                valref = (VaryingRef<float>*) &globals->time;
            } else if (sym.name() == Strings::dtime) {
                valref = (VaryingRef<float>*) &globals->dtime;
            } else if (sym.name() == Strings::dPdtime) {
                valref = (VaryingRef<float>*) &globals->dPdtime;
            }

            if (valref && valref->ptr()) {
                if (dxref && dxref->ptr() && dyref && dyref->ptr()) {
                    // Derivs supplied
                    sym.has_derivs (true);
                    if (rebind) {
                        // Rebinding -- addr is ok, just reset the step
                        sym.step (sym.derivsize());
                    } else {
                        m_context->heap_allot (sym, true);
                    }
                    if (sym.typespec().is_float()) {
                        // It's a float -- use the valref,dxref,dyref
                        // directly.
                        VaryingRef<Dual2<float> > P ((Dual2<float> *)sym.data(), sym.step());
                        for (int i = 0;  i < npoints();  ++i)
                            P[i].set ((*valref)[i], (*dxref)[i], (*dyref)[i]);
                    } else {
                        DASSERT (sym.typespec().is_triple());
                        // It's a triple, so cast our
                        // VaryingRef<float>'s back to Vec3's.
                        VaryingRef<Dual2<Vec3> > P ((Dual2<Vec3> *)sym.data(), sym.step());
                        for (int i = 0;  i < npoints();  ++i)
                            P[i].set (*(Vec3 *)&((*valref)[i]),
                                      *(Vec3 *)&((*dxref)[i]),
                                      *(Vec3 *)&((*dyref)[i]));
                    }
                    // FIXME -- what if the user has already passed valref,
                    // dxref, and dyref with just the right layout that it
                    // could be a Dual<Vec3>?  Should we just point to it,
                    // and avoid the data copy?
                } else {
                    // No derivs anyway -- don't copy the user's data
                    sym.has_derivs (false);
                    sym.data (valref->ptr());  sym.step (valref->step());
                    // FIXME -- Hmmm... which is better, to avoid the copy
                    // here but possibly have the data spread out strangely,
                    // or to copy but then end up with the data contiguous
                    // and in cache?  Experiment at some point.
                }
                ASSERT (sym.data() != NULL);
            }
            float badval;
            bool badderiv;
            int point;
            if (debugnan && check_nan (sym, badval, badderiv, point))
                m_shadingsys->warning ("Found %s%g in shader \"%s\" when binding %s",
                                       badderiv ? "bad derivative " : "",
                                       badval, shadername().c_str(),
                                       sym.name().c_str());                    

        } else if (sym.symtype() == SymTypeParam ||
                   sym.symtype() == SymTypeOutputParam) {
            m_context->m_paramstobind++;
            if (sym.typespec().is_closure()) {
                // Special case -- closures store pointers in the heap
                sym.dataoffset (m_context->closure_allot (m_npoints));
                sym.data (m_context->heapaddr (sym.dataoffset()));
                sym.step (sizeof (ClosureColor *));
            } else if (sym.typespec().simpletype().basetype != TypeDesc::UNKNOWN) {
                size_t addr = m_context->heap_allot (sym.derivsize() * m_npoints);
                sym.data (m_context->heapaddr (addr));
                sym.step (0);
            } else {
                sym.data ((void*) 0); // reset data ptr -- this symbol should never be used
                sym.step (0);
            }
        } else if (sym.symtype() == SymTypeLocal ||
                   sym.symtype() == SymTypeTemp) {
            ASSERT (sym.dataoffset() < 0);
            if (sym.typespec().is_closure()) {
                // Special case -- closures store pointers in the heap, and
                // they are always varying.
                sym.dataoffset (m_context->closure_allot (m_npoints));
                sym.data (m_context->heapaddr (sym.dataoffset()));
                sym.step (sizeof (ClosureColor *));
            } else {
                m_context->heap_allot (sym);
            }
            if (sym.typespec().simpletype() == TypeDesc::TypeString) {
                // Uninitialized strings in the heap can really screw
                // things up, since they are a pointer underneath.
                // Clear and uniformize just in case.  Another stretegy
                // we might try someday is to have the compiler generate
                // initializations for all string vars (or ALL vars?)
                // that aren't unconditionally assigned by the user, but
                // for now we're just taking care of it here.
                sym.step (0);
                ((ustring *)sym.data())->clear ();
            }
        } else if (sym.symtype() == SymTypeConst) {
            ASSERT (sym.data() != NULL &&
                    "Const symbol should already have valid data address");
        } else {
            ASSERT (0 && "Should never get here");
        }
#if 0
        if (m_debug)
            m_shadingsys->info ("  bound %s to address %p, step %d, size %d %s",
                                sym.mangled().c_str(), sym.data(),
                                sym.step(), sym.size(),
                                sym.has_derivs() ? "(derivs)" : "(no derivs)");
#endif
    }

    // Mark the parameters that are driven by connections
    for (int i = 0;  i < m_instance->nconnections();  ++i) {
        const Connection &con = m_instance->connection (i);
        sym (con.dst.param).valuesource (Symbol::ConnectedVal);
    }
    // FIXME -- you know, the connectivity is fixed for the whole group
    // and its instances.  We *could* mark them as connected and possibly
    // do some of the other connection work once per instance rather than
    // once per execution.  Come back to this later and investigate.

    // OK, we're successfully bound.
    m_bound = true;

#ifdef DEBUG_ADJUST_VARYING
    m_adjust_calls = 0;
    m_keep_varying = 0;
    m_keep_uniform = 0;
    m_make_varying = 0;
    m_make_uniform = 0;
#endif
}



void
ShadingExecution::bind_initialize_param (Symbol &sym, int symindex)
{
    ASSERT (! sym.initialized ());
    ASSERT (m_runstate_stack.size() > 0);
    m_context->m_paramsbound++;
    sym.initialized (true);

    // Lazy parameter binding: we figure out the value for this parameter based
    // on the following priority:
    //    1) connection(s) from an earlier layer
    //    2) geometry attribute on the surface of the same name
    //    3) instance values
    //    4) default values (may include "init-ops")

    if (sym.valuesource() == Symbol::ConnectedVal) {
        // Run through all connections for this layer
        // NOTE: more than one connection may contribute to this value if we have
        //       partial connections (into individual array elements or components)
        ExecutionLayers &execlayers (m_context->execlayer (shaderuse()));
        for (int c = 0;  c < m_instance->nconnections();  ++c) {
            const Connection &con (m_instance->connection (c));
            // If the connection gives a value to this param
            if (con.dst.param == symindex) {
                // If the earlier layer it comes from has not yet been executed, do so now.
                if (! execlayers[con.srclayer].executed())
                   run_connected_layer (con.srclayer);
                // Now bind the connection for this param (this copies the actual value)
                bind_connection(con);
            }
        }
    } else {
        // Resolve symbols that map to user-data on the geometry
        // FIXME: call only one method here
        if (renderer_has_userdata (sym.name(), sym.typespec().simpletype(),
                                   &m_context->m_globals->renderstate[0])) {
            // This value came from geometry
            sym.valuesource(Symbol::GeomVal);

            // Mark as not having diverged so that adjust_varying will _never_
            // copy values (since they are not valid at this point, it would
            // just be a waste of time)
            int old_conditional_level = m_conditional_level;
            m_conditional_level = 0;
            adjust_varying(sym, true, false /* don't keep old values */);
            m_conditional_level = old_conditional_level;

            bool wants_derivatives = (sym.typespec().is_float() || sym.typespec().is_triple());
            ShaderGlobals *globals = m_context->m_globals;
            // FIXME: runflags required here even if we are using something else
            if (!get_renderer_userdata(m_runstate_stack.front().runflags, m_npoints, wants_derivatives,
                                       sym.name(), sym.typespec().simpletype(),
                                       &globals->renderstate[0], globals->renderstate.step(),
                                       sym.data(), sym.step())) {
                m_shadingsys->error ("could not find previously found userdata: %s", sym.name().c_str());
                ASSERT(0);
            } 
        } else if (sym.valuesource() == Symbol::DefaultVal &&
                   sym.initbegin() != sym.initend()) {
            // If it's still a default value and it's determined by init
            // ops, run them now
            int old_ip = m_ip;  // Save the instruction pointer
            run (m_runstate_stack.front().runflags,
                 m_runstate_stack.front().indices, m_runstate_stack.front().nindices,
                 sym.initbegin(), sym.initend());
            m_ip = old_ip;
        } else if (!sym.typespec().is_closure() && !sym.typespec().is_structure()) {
            // Otherwise, if it's a normal data type (non-closure,
            // non-struct) copy its instance and/or default value now.
            DASSERT (sym.step() == 0);
            if (sym.typespec().simpletype().basetype == TypeDesc::FLOAT)
                memcpy (sym.data(), &m_instance->m_fparams[sym.dataoffset()],
                        sym.typespec().simpletype().size());
            else if (sym.typespec().simpletype().basetype == TypeDesc::INT)
                memcpy (sym.data(), &m_instance->m_iparams[sym.dataoffset()],
                        sym.typespec().simpletype().size());
            else if (sym.typespec().simpletype().basetype == TypeDesc::STRING)
                memcpy (sym.data(), &m_instance->m_sparams[sym.dataoffset()],
                        sym.typespec().simpletype().size());
            else {
                std::cerr << "Type is " << sym.typespec().c_str() << "\n";
                ASSERT (0 && "unrecognized type -- no default value");
            }
            if (sym.has_derivs ())
                zero_derivs (sym);
            // FIXME -- is there anything to be gained by just pointing
            // to the parameter data, not copying it?
        }
    }
    float badval;
    bool badderiv;
    int point;
    // FIXME: uses wrong runstate (only the currently active points will be
    //        checked for nan, even though we evaluated all of them)
    if (m_shadingsys->debug_nan() &&
        check_nan (sym, badval, badderiv, point))
        m_shadingsys->warning ("Found %s%g in shader \"%s\" when interpolating %s",
                               badderiv ? "bad derivative " : "",
                               badval, shadername().c_str(),
                               sym.name().c_str());
}



void
ShadingExecution::bind_connection (const Connection &con)
{
    int symindex = con.dst.param;
    Symbol &dstsym (sym (symindex));
    ExecutionLayers &execlayers (context()->execlayer (shaderuse()));
    ShadingExecution &srcexec (execlayers[con.srclayer]);
    ASSERT (srcexec.m_bound);
    ASSERT (srcexec.m_executed);
    Symbol &srcsym (srcexec.sym (con.src.param));
    if (!srcsym.initialized()) {
        // Due to lazy-param binding, it is possible to have run the source
        // layer but not initialized all its parameters. Also, since we have
        // already run this layer, its runstate stack is now empty. So we must
        // push a fresh copy of the original state
        srcexec.push_runstate (m_runstate_stack.front().runflags,
                               m_runstate_stack.front().beginpoint,
                               m_runstate_stack.front().endpoint,
                               m_runstate_stack.front().indices,
                               m_runstate_stack.front().nindices);
        srcexec.bind_initialize_param (srcsym, con.src.param);
        srcexec.pop_runstate ();
    }
#if 0
    std::cerr << " bind_connection: layer " << con.srclayer << ' '
              << srcexec.instance()->layername() << ' ' << srcsym.name()
              << " to " << m_instance->layername() << ' '
              << dstsym.name() << "\n";
#endif

    // Try to identify the simple case where we can just alias the
    // variable, with no copying.
    bool simple = (equivalent(srcsym.typespec(), dstsym.typespec()) &&
                   srcsym.symtype() != SymTypeGlobal &&
                   dstsym.symtype() != SymTypeGlobal);
    if (simple) {
#if 0
        std::cerr << "  simple: setting " << dstsym.name() << " to " 
                  << srcsym.name() << ' ' << (void *)srcsym.data() 
                  << "/" << srcsym.step() << ", was " 
                  << (void *)dstsym.data() << "/" << dstsym.step() << "\n";
#endif
        dstsym.data (srcsym.data ());
        dstsym.step (srcsym.step ());
        // dstsym.dataoffset (srcsym->dataoffset ());  needed?
        ASSERT (dstsym.valuesource () == Symbol::ConnectedVal);
    } else {
        // More complex case -- casting is involved, or only a
        // partial copy (such as just cone component).
        error ("Unimplemented connection type: %s %s -> %s %s\n"
               "\tPartial copies not yet supported.",
               srcsym.typespec().c_str(), srcsym.name().c_str(),
               dstsym.typespec().c_str(), dstsym.name().c_str());
        return;
    }
    dstsym.connected (true);
}



void
ShadingExecution::run (Runflag *runflags, int *indices, int nindices, int beginop, int endop)
{
    if (m_debug)
        m_shadingsys->info ("Running ShadeExec %p, shader %s",
                            this, shadername().c_str());

    push_runstate (runflags, 0, m_context->npoints(), indices, nindices);

    if (beginop >= 0) {
        ASSERT (m_bound);
        ASSERT (m_npoints == m_context->npoints());
        /*
         * Run just the op range supplied (this is used for on-demand init ops)
         * We might be running the init ops from inside a conditional, but the
         * init ops themselves should act as if they were run from outside.
         *
         * So we backup the current conditional level and reset it after we are
         * done.
         */

        int prev_conditional_level = m_conditional_level;
        m_conditional_level = 0;
        run (beginop, endop);
        m_conditional_level = prev_conditional_level;
    } else {
        ASSERT (!m_bound);
        ASSERT (!m_executed);
        /*
         * Default (<0) means run main code block. This should happen only once
         * per shader.
         */
        // Bind symbols before running the shader
        ShaderGroup &sgroup (context()->attribs()->shadergroup (shaderuse()));
        bind (sgroup[layer()]);

        m_conditional_level = 0;
        run (m_instance->maincodebegin(), m_instance->maincodeend());
        m_executed = true;
        ASSERT(m_conditional_level == 0); // make sure we nested our loops and ifs correctly
    }
    pop_runstate ();
}



void
ShadingExecution::run (int beginop, int endop)
{
    if (m_debug)
        m_shadingsys->info ("Running ShadeExec %p, shader %s ops [%d,%d)",
                            this, shadername().c_str(),
                            beginop, endop);
    const int *args = &(instance()->args()[0]);
    bool debugnan = m_shadingsys->debug_nan ();
    OpcodeVec &code (m_instance->ops());
    int instructions_run = 0;
    for (m_ip = beginop; m_ip < endop && m_runstate.beginpoint < m_runstate.endpoint;  ++m_ip) {
        ++instructions_run;
        DASSERT (m_ip >= 0 && m_ip < (int)instance()->ops().size());
        Opcode &op (code[m_ip]);

#if 0
        // Debugging tool -- sample the run flags
        static atomic_ll count;
        if (((++count) % 172933ul /*1142821ul*/) == 0) {  //   12117689ul
            std::cerr << "rf ";
            for (int i = 0;  i < npoints();  ++i)
                std::cerr << (int)(m_runstate.runflags[i] ? 1 : 0) << ' ';
            std::cerr << std::endl;
        }
#endif

#if 0
        if (m_debug) {
            m_shadingsys->info ("Before running op %d %s, values are:",
                                m_ip, op.opname().c_str());
            for (int i = 0;  i < op.nargs();  ++i) {
                Symbol &s (sym (args[op.firstarg()+i]));
                m_shadingsys->info ("    %s\n%s", s.mangled().c_str(),
                                    printsymbolval(s).c_str());
            }
        }
#endif
        ASSERT (op.implementation() && "Unimplemented op!");
        op (this, op.nargs(), args+op.firstarg());

#if 0
        if (m_debug) {
            m_shadingsys->info ("After running %s, new values are:",
                                op.opname().c_str());
            for (int i = 0;  i < op.nargs();  ++i) {
                Symbol &s (sym (args[op.firstarg()+i]));
                m_shadingsys->info ("    %s\n%s", s.mangled().c_str(),
                                    printsymbolval(s).c_str());
            }
        }
#endif
        if (debugnan)
            check_nan (op);
    }
    m_context->m_instructions_run += instructions_run;
}



void
ShadingExecution::check_nan (Opcode &op)
{
    // Check every writable argument of this op, at every shading point
    // that's turned on, check for NaN and Inf values, and print a
    // warning if found.
    for (int a = 0;  a < op.nargs();  ++a) {
        if (! op.argwrite (a))
            continue;  // Skip args that weren't written
        Symbol &s (sym (instance()->args()[op.firstarg()+a]));
        float badval;
        bool badderiv;
        int point;
        if (check_nan (s, badval, badderiv, point))
            m_shadingsys->warning ("Generated %s%g in shader \"%s\",\n"
                                   "\tsource \"%s\", line %d (instruction %s, arg %d)\n"
                                   "\tsymbol \"%s\", %s (step %d), point %d of %d",
                                   badderiv ? "bad derivative " : "",
                                   badval, shadername().c_str(),
                                   op.sourcefile().c_str(),
                                   op.sourceline(), op.opname().c_str(), a,
                                   s.name().c_str(), s.is_uniform() ? "uniform" : "varying", s.step(), point, npoints() );
    }
}



bool
ShadingExecution::check_nan (Symbol &sym, float &badval,
                             bool &badderiv, int &point)
{
    if (sym.typespec().is_closure())
        return false;
    TypeDesc t (sym.typespec().simpletype());
    badval = 0;
    badderiv = false;
    if (t.basetype == TypeDesc::FLOAT) {
        int agg = t.aggregate * t.numelements();
        ShadingExecution *exec = this;
        SHADE_LOOP_BEGIN
            float *f = (float *)((char *)sym.data()+sym.step()*i);
            for (int d = 0;  d < 3;  ++d)  {  // for each of val, dx, dy
                for (int c = 0;  c < agg;  ++c)
                    if (! std::isfinite (f[c])) {
                        badval = f[c];
                        badderiv = (d > 0);
                        point = i;
                        return true;
                    }
                if (! sym.has_derivs())
                    break;    // don't advance to next deriv if no derivs
                // Step to next derivative
                f = (float *)((char *)f + sym.deriv_step());
            }
        SHADE_LOOP_END
    }
    return false;
}



void
ShadingExecution::run_connected_layer (int layer)
{
    ExecutionLayers &execlayers (m_context->execlayer (shaderuse()));
    ShadingExecution &connected (execlayers[layer]);
    ASSERT (! connected.executed ());

#if 0
    std::cerr << "Lazy running layer " << layer << ' ' << "\n";
#endif
    // Run the earlier layer using the runflags we were originally
    // called with.
    connected.run (m_runstate_stack.front().runflags, 
                   m_runstate_stack.front().indices,
                   m_runstate_stack.front().nindices);
    m_context->m_lazy_evals += 1;
}



void
ShadingExecution::adjust_varying_makevarying (Symbol &sym, bool preserve_value)
{
    // sym is uniform, but we're either assigning a new varying
    // value or we're inside a conditional.  Promote sym to varying.
    size_t size = sym.derivsize();
    sym.step (size);
    if (preserve_value || diverged()) {
        // Propagate the value from slot 0 to other slots
        char *data = (char *) sym.data();
#if USE_RUNFLAGS
        SHADE_LOOP_RUNFLAGS_BEGIN (m_runstate_stack.front().runflags,
                                   m_runstate_stack.front().beginpoint,
                                   m_runstate_stack.front().endpoint)
#elif USE_RUNINDICES
        SHADE_LOOP_INDICES_BEGIN (m_runstate_stack.front().indices,
                                  m_runstate_stack.front().nindices)
#elif USE_RUNSPANS
        SHADE_LOOP_SPANS_BEGIN (m_runstate_stack.front().indices,
                                m_runstate_stack.front().nindices)
#endif
            if (i != 0) memcpy (data + i*size, data, size);
        SHADE_LOOP_END
    }
}


void
ShadingExecution::adjust_varying_full (Symbol &sym, bool varying_assignment,
                                  bool preserve_value)
{
    // TODO: move comments to adjust_varying (this version is no longer called from anywhere)

    // This is tricky.  To make sure we're catching all the cases, let's
    // enumerate them by the current symbol varyingness, the assignent
    // varyingness, and whether all points in the grid are active:
    //   case   sym    assignment   diverged     action
    //    0      v         v            y           v (leave alone)
    //    1      v         v            n           v (leave alone)
    //    2      v         u            y           v (leave alone)
    //    3      v         u            n           u (demote)
    //    4      u         v            y           v (promote)
    //    5      u         v            n           v (promote)
    //    6      u         u            y           v (promote)
    //    7      u         u            n           u (leave alone)

    // If we're inside a conditional of any kind, even a uniform assignment
    // makes the result varying.  
    varying_assignment |= diverged();

    // This reduces us to just four cases:
    //   case   sym    assignment   action
    //    0/1/2  v         v          v (leave alone)
    //    3      v         u          u (demote)
    //    4/5/6  u         v          v (promote)
    //    7      u         u          u (leave alone)

#ifdef DEBUG_ADJUST_VARYING
    ++m_adjust_calls;
    if (sym.is_varying() == varying_assignment) {
        if (varying_assignment)
            ++m_keep_varying;
        else
            ++m_keep_uniform;
    } else {
        if (varying_assignment)
            ++m_make_varying;
        else
            ++m_make_uniform;
    }
#endif

    // Trivial case: we need it varying and it already is, or we need it
    // uniform and it already is.
    if (sym.is_varying() == varying_assignment)
        return;

    if (varying_assignment) {
        // sym is uniform, but we're either assigning a new varying
        // value or we're inside a conditional.  Promote sym to varying.
        size_t size = sym.has_derivs() ? 3*sym.deriv_step() : sym.size();
        sym.step (size);
        if (preserve_value || diverged()) {
            // Propagate the value from slot 0 to other slots
            char *data = (char *) sym.data();
#if USE_RUNFLAGS
            SHADE_LOOP_RUNFLAGS_BEGIN (context()->m_original_runflags,
                                       0, m_npoints)
#elif USE_RUNINDICES
            SHADE_LOOP_INDICES_BEGIN (context()->m_original_indices,
                                      context()->m_original_nindices)
#elif USE_RUNSPANS
            SHADE_LOOP_SPANS_BEGIN (context()->m_original_indices,
                                    context()->m_original_nindices)
#endif
                memcpy (data + i*size, data, size);
            SHADE_LOOP_END
        }
    } else {
        // sym is varying, but we're assigning a new uniform value AND
        // we're not inside a conditional.  Safe to demote sym to uniform.
        if (sym.symtype() != SymTypeGlobal) { // DO NOT demote a global
            sym.step (0);
            if (sym.has_derivs()) {
                size_t deriv_step = sym.deriv_step();
                memset ((char *)sym.data()+deriv_step, 0, 2*deriv_step);
            }
        }
    }
}



void
ShadingExecution::zero (Symbol &sym)
{
    size_t size = sym.has_derivs() ? sym.deriv_step()*3 : sym.size();
    if (sym.is_uniform ()) {
        memset (sym.data(), 0, size);
#if 0
    } else if (sym.is_varying() && all_points_on()) {
        // Varying, but we can do one big memset
        memset (sym.data(), 0, size * m_npoints);
#endif
    } else {
        // Varying, with some points on and some off
        ShadingExecution *exec = this;
        SHADE_LOOP_BEGIN
            memset ((char *)sym.data() + i * sym.step(), 0, size);
        SHADE_LOOP_END
    }
}



void
ShadingExecution::zero_derivs (Symbol &sym)
{
    DASSERT (sym.has_derivs ());
    size_t deriv_step = sym.deriv_step ();
    size_t deriv_size = 2 * deriv_step;
    char *data = (char *)sym.data() + deriv_step;
    if (sym.is_uniform ())
        memset (data, 0, deriv_size);
    else {
        ShadingExecution *exec = this;
        SHADE_LOOP_BEGIN
            memset (data + i*sym.step(), 0, deriv_size);
        SHADE_LOOP_END
    }
}



void
ShadingExecution::push_runstate (Runflag *runflags,
                                 int beginpoint, int endpoint,
                                 int *indices, int nindices)
{
#if USE_RUNFLAGS
    ASSERT (runflags != NULL && beginpoint < endpoint);
#else
    ASSERT (indices != NULL && nindices > 0);
#endif
    m_runstate.init (runflags, beginpoint, endpoint, //m_runstate.allpointson,
                     indices, nindices);
#if USE_RUNFLAGS
    m_runstate.beginpoint = INT_MAX;
    m_runstate.endpoint = -1;
//    m_runstate.allpointson = (begin == 0 && end == m_npoints);

    // FIXME: this might be done redundantly (init-ops for example)
    for (int i = beginpoint;  i < endpoint;  ++i) {
        if (m_runstate.runflags[i]) {
            if (i < m_runstate.beginpoint)
                m_runstate.beginpoint = i;
            if (i >= m_runstate.endpoint)
                m_runstate.endpoint = i+1;
        } else {
//            m_runstate.allpointson = false;
        }
    }
#endif
    m_runstate_stack.push_back (m_runstate);
}



void
ShadingExecution::pop_runstate ()
{
    DASSERT(m_runstate_stack.size() > 0);
    m_runstate_stack.pop_back ();
    if (m_runstate_stack.size()) {
        m_runstate = m_runstate_stack.back();
    } else {
        m_runstate.runflags = NULL;
        m_runstate.indices = NULL;
        m_runstate.nindices = 0;
    }
}



std::string
ShadingExecution::format_symbol (const std::string &format,
                                 Symbol &sym, int whichpoint)
{
    if (sym.typespec().is_closure()) {
        // Special case for printing closures
        std::stringstream stream;
        stream << *(((const ClosureColor **)sym.data())[whichpoint]);
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



std::string
ShadingExecution::printsymbolval (Symbol &sym)
{
    std::stringstream out;
    TypeDesc type = sym.typespec().simpletype();
    ShadingExecution *exec = this;
    SHADE_LOOP_BEGIN
        if (sym.is_uniform())
            out << "\tuniform";
        else if (i == beginpoint() || (i%8) == 0)
            out << "\t" << i << ": ";
        if (sym.typespec().is_closure())
            out << format_symbol (" %s", sym, i);
        else if (type.basetype == TypeDesc::FLOAT)
            out << format_symbol (" %g", sym, i);
        else if (type.basetype == TypeDesc::INT)
            out << format_symbol (" %d", sym, i);
        else if (type.basetype == TypeDesc::STRING)
            out << format_symbol (" \"%s\"", sym, i);
        if (i == endpoint()-1 || (i%8) == 7 ||
                sym.is_uniform() || sym.typespec().is_closure())
            out << "\n";
        if (sym.is_uniform())
            SHADE_LOOP_EXIT
    SHADE_LOOP_END
    return out.str ();
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
