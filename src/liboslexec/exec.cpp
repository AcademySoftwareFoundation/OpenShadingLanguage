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
    const int *args = &m_master->m_args[0];
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
ShadingExecution::bind (ShadingContext *context, ShaderUse use,
                        int layerindex, ShaderInstance *instance)
{
    ASSERT (! m_bound);  // avoid double-binding
    ASSERT (context != NULL && instance != NULL);

    m_shadingsys = &context->shadingsys ();
    m_debug = shadingsys()->debug();
    if (m_debug)
        shadingsys()->info ("bind ctx %p use %s layer %d", context,
                            shaderusename(use), layerindex);
    m_use = use;

    // Take various shortcuts if we are re-binding the same instance as
    // last time.
    bool rebind = (shadingsys()->allow_rebind() &&
                   instance->id() == m_last_instance_id &&
                   m_npoints_bound >= m_context->npoints());
    if (rebind) {
        ++context->m_rebinds;
    } else {
        m_context = context;
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
        m_symbols.resize (m_instance->m_symbols.size());
        if (m_symbols.size() /* && !rebind */)
            memcpy (&m_symbols[0], &m_instance->m_symbols[0], 
                    m_instance->m_symbols.size() * sizeof(Symbol));
    }

    m_npoints = m_context->npoints ();

    ShaderGlobals *globals (m_context->m_globals);

    // FIXME: bind the symbols -- get the syms ready and pointing to the
    // right place in the heap,, interpolate primitive variables, handle
    // connections, initialize all parameters
    BOOST_FOREACH (Symbol &sym, m_symbols) {
        if (m_debug)
            m_shadingsys->info ("  bind %s, offset %d",
                                sym.mangled().c_str(), sym.dataoffset());
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
            } else if (sym.name() == Strings::I) {
                valref = (VaryingRef<float>*) &globals->I;
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
                        VaryingRef<Dual2<float> > P ((Dual2<Vec3> *)sym.data(), sym.step());
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

        } else if (sym.symtype() == SymTypeParam ||
                   sym.symtype() == SymTypeOutputParam) {
            if (sym.typespec().is_closure()) {
                // Special case -- closures store pointers in the heap
                sym.dataoffset (m_context->closure_allot (m_npoints));
                sym.data (m_context->heapaddr (sym.dataoffset()));
                sym.step (sizeof (ClosureColor *));
            } else {
                size_t addr = context->heap_allot (sym.derivsize() * m_npoints);
                sym.data (m_context->heapaddr (addr));
                sym.step (0);  // FIXME
                // Copy the parameter value
                // FIXME -- if the parameter is not being overridden and is
                // not writeable, I think we should just point to the parameter
                // data, not copy it?  Or does it matter?
                if (sym.typespec().simpletype().basetype == TypeDesc::FLOAT)
                    memcpy (sym.data(), &instance->m_fparams[sym.dataoffset()],
                            sym.typespec().simpletype().size());
                else if (sym.typespec().simpletype().basetype == TypeDesc::INT)
                    memcpy (sym.data(), &instance->m_iparams[sym.dataoffset()],
                            sym.typespec().simpletype().size());
                else if (sym.typespec().simpletype().basetype == TypeDesc::STRING)
                    memcpy (sym.data(), &instance->m_sparams[sym.dataoffset()],
                            sym.typespec().simpletype().size());
                if (sym.has_derivs())
                   zero_derivs (sym);
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
        if (m_debug)
            m_shadingsys->info ("  bound %s to address %p, step %d, size %d %s",
                                sym.mangled().c_str(), sym.data(),
                                sym.step(), sym.size(),
                                sym.has_derivs() ? "(derivs)" : "(no derivs)");
    }

    // Handle all of the symbols that are connected to earlier layers.
    bind_connections ();

    // Mark symbols that map to user-data on the geometry
    bind_mark_geom_variables (m_instance);

    // OK, we're successfully bound.
    m_bound = true;
}



void
ShadingExecution::bind_initialize_params (ShaderInstance *inst)
{
    ShaderMaster *master = inst->master();
    for (int i = master->m_firstparam;  i <= master->m_lastparam;  ++i) {
        Symbol *sym = symptr (i);
        if (sym->valuesource() == Symbol::DefaultVal) {
            // Execute init ops, if there are any
            if (sym->initbegin() != sym->initend()) {
                run (context()->m_original_runflags,
                     sym->initbegin(), sym->initend());
            }
        } else if (sym->valuesource() == Symbol::InstanceVal) {
            // FIXME -- eventually, copy the instance values here,
            // rather than above in bind(), so that we skip the
            // unnecessary copying if the values came from geom or
            // connections.  As it stands now, there is some redundancy.
        } else if (sym->valuesource() == Symbol::GeomVal) {
            adjust_varying(*sym, true, false /* don't keep old values */);
            bool wants_derivatives = (sym->typespec().is_float() || sym->typespec().is_triple());
            ShaderGlobals *globals = m_context->m_globals;
            if (!get_renderer_userdata (m_npoints, wants_derivatives,
                                        sym->name(), sym->typespec().simpletype(),
                                        &globals->renderstate[0], globals->renderstate.step(),
                                        sym->data(), sym->step())) {
#ifdef DEBUG
                std::cerr << "could not find previously found userdata '" << sym->name() << "'\n";
#endif
            }

        } else if (sym->valuesource() == Symbol::ConnectedVal) {
            // Nothing to do if it fully came from an earlier layer
        }
    }
}



void
ShadingExecution::bind_mark_geom_variables (ShaderInstance *inst)
{
    ShaderGlobals *globals (m_context->m_globals);
    ShaderMaster *master = inst->master();
    for (int i = master->m_firstparam;  i <= master->m_lastparam;  ++i) {
        Symbol *sym = symptr (i);
        if (sym->valuesource() != Symbol::ConnectedVal) {
            if (renderer_has_userdata (sym->name(), sym->typespec().simpletype(), &globals->renderstate[0])) {
               sym->valuesource(Symbol::GeomVal);
            }
        }
    }
}

void
ShadingExecution::bind_connections ()
{
    for (int i = 0;  i < m_instance->nconnections();  ++i)
        bind_connection (m_instance->connection (i));
    // FIXME -- you know, the connectivity is fixed for the whole group
    // and its instances.  We *could* mark them as connected and possibly
    // do some of the other connection work once per instance rather than
    // once per execution.  Come back to this later and investigate.
}



void
ShadingExecution::bind_connection (const Connection &con)
{
    int symindex = con.dst.param;
    Symbol &dstsym (sym (symindex));
    ExecutionLayers &execlayers (context()->execlayer (shaderuse()));
    ShadingExecution &srcexec (execlayers[con.srclayer]);
    ASSERT (srcexec.m_bound);
    Symbol &srcsym (srcexec.sym (con.src.param));
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
        dstsym.valuesource (Symbol::ConnectedVal);
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
ShadingExecution::unbind ()
{
    m_bound = false;
    m_executed = false;
}



void
ShadingExecution::run (Runflag *rf, int beginop, int endop)
{
    if (m_executed)
        return;       // Already executed

    if (m_debug)
        m_shadingsys->info ("Running ShadeExec %p, shader %s",
                            this, m_master->shadername().c_str());

    ASSERT (m_bound);  // We'd better be bound at this point

    // Make space for new runflags
    Runflag *runflags = ALLOCA (Runflag, m_npoints);
    if (rf) {
        // Passed runflags -- copy those
        memcpy (runflags, rf, m_npoints*sizeof(Runflag));
    } else {
        // If not passed runflags, make new ones
        for (int i = 0;  i < m_npoints;  ++i)
            runflags[i] = RunflagOn;
    }

    push_runflags (runflags, 0, m_npoints);
    if (beginop >= 0)   // Run just the op range supplied, and no init ops
        run (beginop, endop);
    else {              // Default (<0) means run param init ops + main code
        bind_initialize_params (m_instance);  // run param init code
        run (m_master->m_maincodebegin, m_master->m_maincodeend);
        m_executed = true;
    }
    pop_runflags ();
}



void
ShadingExecution::run (int beginop, int endop)
{
    if (m_debug)
        m_shadingsys->info ("Running ShadeExec %p, shader %s ops [%d,%d)",
                            this, m_master->shadername().c_str(),
                            beginop, endop);
    const int *args = &m_master->m_args[0];
    bool debugnan = m_shadingsys->debug_nan ();
    for (m_ip = beginop; m_ip < endop && m_beginpoint < m_endpoint;  ++m_ip) {
        DASSERT (m_ip >= 0 && m_ip < (int)m_master->m_ops.size());
        Opcode &op (this->op ());
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
        op (this, op.nargs(), args+op.firstarg(),
            m_runflags, m_beginpoint, m_endpoint);

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
}



void
ShadingExecution::check_nan (Opcode &op)
{
    // Check every writable argument of this op, at every shading point
    // that's turned on, check for NaN and Inf values, and print a
    // warning if found.
    const int *args = &m_master->m_args[0];
    for (int a = 0;  a < op.nargs();  ++a) {
        if (! op.argwrite (a))
            continue;  // Skip args that weren't written
        Symbol &s (sym (args[op.firstarg()+a]));
        TypeDesc t (s.typespec().simpletype());
        bool found_nan = false;
        float badval = 0;
        if (t.basetype == TypeDesc::FLOAT) {
            int agg = t.aggregate;
            for (int i = m_beginpoint;  i <= m_endpoint;  ++i) {
                if (m_runflags[i]) {
                    float *f = (float *)((char *)s.data()+s.step()*i);
                    for (int c = 0;  c < agg;  ++c)
                        if (! std::isfinite (f[c])) {
                            badval = f[c];
                            found_nan = true;
                        }
                }
            }
        }
        if (found_nan)
            m_shadingsys->warning ("Generated %g at %s, line %d (instruction %s, arg %d)",
                                   badval, op.sourcefile().c_str(),
                                   op.sourceline(), op.opname().c_str(), a);
    }
}



void
ShadingExecution::run_connected_layer (int layer)
{
    ShadingContext *ctx = context();
    ShaderUse use = ctx->use();
    ExecutionLayers &execlayers (ctx->execlayer (use));
    ShadingExecution &connected (execlayers[layer]);
    ASSERT (! connected.executed ());

    // Run the earlier layer using the runflags we were originally
    // called with.
    ShaderGroup &sgroup (ctx->attribs()->shadergroup (use));
    size_t nlayers = (int) sgroup.nlayers ();
    if (! connected.m_bound)
        connected.bind (ctx, use, layer, sgroup[layer]);
    connected.run (ctx->m_original_runflags);
    ctx->m_lazy_evals += 1;

    // Now re-bind the connections between that layer and all other
    // later layers that have not yet executed.
    for (int i = layer+1;  i < (int)nlayers;  ++i) {
        ShadingExecution &exec (execlayers[i]);
        if (exec.m_bound && ! exec.m_executed) {
            ShaderInstance *inst = exec.instance();
            for (int c = 0;  c < inst->nconnections();  ++c) {
                const Connection &con (inst->connection (c));
                if (con.srclayer == layer)
                    exec.bind_connection (con);
            }
        }
    }
}



void
ShadingExecution::adjust_varying (Symbol &sym, bool varying_assignment,
                                  bool preserve_value)
{
    // This is tricky.  To make sure we're catching all the cases, let's
    // enumerate them by the current symbol varyingness, the assignent
    // varyingness, and whether all points in the grid are active:
    //   case   sym    assignment   all_pts_on     action
    //    0      v         v            n           v (leave alone)
    //    1      v         v            y           v (leave alone)
    //    2      v         u            n           v (leave alone)
    //    3      v         u            y           u (demote)
    //    4      u         v            n           v (promote)
    //    5      u         v            y           v (promote)
    //    6      u         u            n           v (promote)
    //    7      u         u            y           u (leave alone)

    // If we're inside a conditional of any kind, even a uniform assignment
    // makes the result varying.  
    varying_assignment |= ! all_points_on();

    // This reduces us to just four cases:
    //   case   sym    assignment   action
    //    0/1/2  v         v          v (leave alone)
    //    3      v         u          u (demote)
    //    4/5/6  u         v          v (promote)
    //    7      u         u          u (leave alone)

    // Trivial case: we need it varying and it already is, or we need it
    // uniform and it already is.
    if (sym.is_varying() == varying_assignment)
        return;

    if (varying_assignment) {
        // sym is uniform, but we're either assigning a new varying
        // value or we're inside a conditional.  Promote sym to varying.
        size_t size = sym.has_derivs() ? 3*sym.deriv_step() : sym.size();
        sym.step (size);
        if (preserve_value || ! all_points_on()) {
            // Propagate the value from slot 0 to other slots
            char *data = (char *) sym.data();
            for (int i = 1;  i < m_npoints;  ++i)
                memcpy (data + i*size, data, size);
        }
    } else {
        // sym is varying, but we're assigning a new uniform value AND
        // we're not inside a conditional.  Safe to demote sym to uniform.
        if (sym.symtype() != SymTypeGlobal) { // DO NOT demote a global
            sym.step (0);
            if (sym.has_derivs())
                zero_derivs (sym);
        }
    }
}



void
ShadingExecution::zero (Symbol &sym)
{
    size_t size = sym.has_derivs() ? sym.deriv_step()*3 : sym.size();
    if (sym.is_uniform ())
        memset (sym.data(), 0, size);
    else if (sym.is_varying() && all_points_on()) {
        // Varying, but we can do one big memset
        memset (sym.data(), 0, size * m_npoints);
    } else {
        // Varying, with some points on and some off
        char *data = (char *)sym.data() + m_beginpoint * sym.step();
        for (int i = m_beginpoint;  i < m_endpoint;  ++i, data += sym.step())
            if (m_runflags[i])
                memset (data, 0, size);
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
        data += m_beginpoint * sym.step();
        for (int i = m_beginpoint;  i < m_endpoint;  ++i, data += sym.step())
            if (m_runflags[i])
                memset (data, 0, deriv_size);
    }
}



void
ShadingExecution::push_runflags (Runflag *runflags,
                                 int beginpoint, int endpoint)
{
    ASSERT (runflags != NULL);
    m_runflags = runflags;
    new_runflag_range (beginpoint, endpoint);
    m_runflag_stack.push_back (Runstate (m_runflags, m_beginpoint,
                                         m_endpoint, m_all_points_on));
}



void
ShadingExecution::pop_runflags ()
{
    m_runflag_stack.pop_back ();
    if (m_runflag_stack.size()) {
        const Runstate &r (m_runflag_stack.back());
        m_runflags = r.runflags;
        m_beginpoint = r.beginpoint;
        m_endpoint = r.endpoint;
        m_all_points_on = r.allpointson;
    } else {
        m_runflags = NULL;
    }
}



void
ShadingExecution::new_runflag_range (int begin, int end)
{
    m_beginpoint = INT_MAX;
    m_endpoint = -1;
    m_all_points_on = (begin == 0 && end == m_npoints);
    for (int i = begin;  i < end;  ++i) {
        if (m_runflags[i]) {
            if (i < m_beginpoint)
                m_beginpoint = i;
            if (i >= m_endpoint)
                m_endpoint = i+1;
        } else {
            m_all_points_on = false;
        }
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
    const char *data = (const char *) sym.data ();
    data += m_beginpoint * sym.step();
    for (int i = m_beginpoint;  i < m_endpoint;  ++i, data += sym.step()) {
        if (sym.is_uniform())
            out << "\tuniform";
        else if (i == m_beginpoint || (i%8) == 0)
            out << "\t" << i << ": ";
        if (sym.typespec().is_closure())
            out << format_symbol (" %s", sym, i);
        else if (type.basetype == TypeDesc::FLOAT)
            out << format_symbol (" %g", sym, i);
        else if (type.basetype == TypeDesc::INT)
            out << format_symbol (" %d", sym, i);
        else if (type.basetype == TypeDesc::STRING)
            out << format_symbol (" \"%s\"", sym, i);
        if (i == m_endpoint-1 || (i%8) == 7 ||
                sym.is_uniform() || sym.typespec().is_closure())
            out << "\n";
        if (sym.is_uniform())
            break;
    }
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
ShadingExecution::get_renderer_userdata(int npoints, bool derivatives, 
                                        ustring name, TypeDesc type, 
                                        void *renderstate, int renderstate_stepsize, 
                                        void *val, int val_stepsize)
{
   return m_renderer->get_userdata(npoints, derivatives, name, type, 
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
