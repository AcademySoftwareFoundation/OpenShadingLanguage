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

#include "boost/foreach.hpp"

#include "OpenImageIO/dassert.h"
#include "OpenImageIO/thread.h"

#include "oslexec_pvt.h"

namespace {
// Define static ustring symbols for very fast comparison
static ustring P_("P"), I_("I"), N_("N"), Ng_("Ng");
static ustring dPdu_("dPdu"), dPdv_("dPdv"), u_("u"), v_("v");
static ustring time_("time"), dtime_("dtime"), dPdtime_("dPdtime");
};


namespace OSL {
namespace pvt {   // OSL::pvt



ShadingExecution::ShadingExecution ()
    : m_context(NULL), m_instance(NULL), m_master(NULL),
      m_bound(false), m_debug(false)
{
}



ShadingExecution::~ShadingExecution ()
{
}



void
ShadingExecution::bind (ShadingContext *context, ShaderUse use,
                        int layerindex, ShaderInstance *instance)
{
    ASSERT (! m_bound);  // avoid double-binding
    ASSERT (context != NULL && instance != NULL);

    m_debug = context->shadingsys().debug();
    if (m_debug)
        std::cout << "bind ctx " << (void *)context << " use " 
                  << shaderusename(use) << " layer " << layerindex << "\n";
    m_use = use;

    // Take various shortcuts if we are re-binding the same instance as
    // last time.
    bool rebind = (m_context == context && m_instance == instance);
    if (! rebind) {
        m_context = context;
        m_instance = instance;
        m_master = instance->master ();
        ASSERT (m_master);
    }

    m_npoints = m_context->npoints ();
    m_symbols = m_instance->m_symbols;  // fresh copy of symbols from instance
    ShaderGlobals *globals (m_context->m_globals);

    // FIXME: bind the symbols -- get the syms ready and pointing to the
    // right place in the heap,, interpolate primitive variables, handle
    // connections, initialize all parameters
    BOOST_FOREACH (Symbol &sym, m_symbols) {
        if (m_debug)
            std::cout << "  bind " << sym.mangled() 
                      << ", offset " << sym.dataoffset() << "\n";
        if (sym.symtype() == SymTypeGlobal) {
            // FIXME -- is this too wasteful here?
            if (sym.name() == P_) {
                sym.data (globals->P.ptr());  sym.step (globals->P.step());
            } else if (sym.name() == I_) {
                sym.data (globals->I.ptr());  sym.step (globals->I.step());
            } else if (sym.name() == N_) {
                sym.data (globals->N.ptr());  sym.step (globals->N.step());
            } else if (sym.name() == Ng_) {
                sym.data (globals->Ng.ptr());  sym.step (globals->Ng.step());
            } else if (sym.name() == u_) {
                sym.data (globals->u.ptr());  sym.step (globals->u.step());
            } else if (sym.name() == v_) {
                sym.data (globals->v.ptr());  sym.step (globals->v.step());
            } else if (sym.name() == dPdu_) {
                sym.data (globals->dPdu.ptr());  sym.step (globals->dPdu.step());
            } else if (sym.name() == dPdv_) {
                sym.data (globals->dPdv.ptr());  sym.step (globals->dPdv.step());
            } else if (sym.name() == time_) {
                sym.data (globals->time.ptr());  sym.step (globals->time.step());
            } else if (sym.name() == dtime_) {
                sym.data (globals->dtime.ptr());  sym.step (globals->dtime.step());
            } else if (sym.name() == dPdtime_) {
                sym.data (globals->dPdtime.ptr());  sym.step (globals->dPdtime.step());
            }
            if (sym.data() == NULL) {
                if (sym.dataoffset() >= 0) {
                    // Not specified where it lives, put it in the heap
                    sym.data (m_context->heapaddr (sym.dataoffset()));
                    sym.step (0);
                    std::cout << "Global " << sym.name() << " at address " << sym.dataoffset() << "\n";
                } else {
                    // ASSERT (sym.dataoffset() >= 0 &&
                    //         "Global ought to already have a dataoffset");
                    // Skip this for now -- it includes L, Cl, etc.
                    sym.step (0);   // FIXME
                }
            }
        } else if (sym.symtype() == SymTypeParam ||
                   sym.symtype() == SymTypeOutputParam) {
//            ASSERT (sym.dataoffset() < 0 &&
//                    "Param should not yet have a data offset");
//            sym.dataoffset (m_context->heap_allot (sym.typespec().simpletype().size()));
            size_t addr = context->heap_allot (sym.typespec().simpletype().size());
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
        } else if (sym.symtype() == SymTypeLocal ||
                   sym.symtype() == SymTypeTemp) {
            ASSERT (sym.dataoffset() < 0);
            sym.dataoffset (m_context->heap_allot (sym.typespec().simpletype().size()));
            sym.data (m_context->heapaddr (sym.dataoffset()));
            sym.step (0);  // FIXME
        } else if (sym.symtype() == SymTypeConst) {
            ASSERT (sym.data() != NULL &&
                    "Const symbol should already have valid data address");
        } else {
            ASSERT (0 && "Should never get here");
        }
        if (m_debug)
            std::cout << "  bound " << sym.mangled() << " to address " 
                      << (void *)sym.data() << ", step " << sym.step() 
                      << ", size " << sym.size() << "\n";
    }

    m_bound = true;
    m_executed = false;
}



void
ShadingExecution::run (Runflag *rf)
{
    if (m_executed)
        return;       // Already executed

    if (m_debug)
        std::cout << "Running ShadeExec " << (void *)this << ", shader " 
                  << m_master->shadername() << "\n";

    ASSERT (m_bound);  // We'd better be bound at this point

    // Make space for new runflags
    m_runflags = (Runflag *) alloca (m_npoints * sizeof(Runflag));
    if (rf) {
        // Passed runflags -- copy those
        memcpy (m_runflags, rf, m_npoints*sizeof(Runflag));
        new_runflag_range (0, m_npoints);
    } else {
        // If not passed runflags, make new ones
        for (int i = 0;  i < m_npoints;  ++i)
            m_runflags[i] = 1;
        m_beginpoint = 0;
        m_endpoint = m_npoints;
        m_all_points_on = true;
    }

    // FIXME -- push the runflags, begin, end

    // FIXME -- this runs every op.  Really, we just want the main code body.
    run (0, (int)m_master->m_ops.size());

    // FIXME -- pop the runflags, begin, end

    m_executed = true;
}



void
ShadingExecution::run (int beginop, int endop)
{
    if (m_debug)
        std::cout << "Running ShadeExec " << (void *)this 
                  << ", shader " << m_master->shadername() 
                  << " ops [" << beginop << "," << endop << ")\n";
    for (m_ip = beginop; m_ip < endop && m_beginpoint < m_endpoint;  ++m_ip) {
        Opcode &op (this->op ());
        if (m_debug)
            std::cout << "  instruction " << m_ip << ": " << op.opname() << " ";
        for (int i = 0;  i < op.nargs();  ++i) {
            int arg = m_master->m_args[op.firstarg()+i];
            if (m_debug)
                std::cout << m_instance->symbol(arg)->mangled() << " ";
        }
        if (m_debug)
            std::cout << "\n";
        ASSERT (op.implementation() && "Unimplemented op!");
        op (this, op.nargs(), &m_master->m_args[op.firstarg()],
            m_runflags, m_beginpoint, m_endpoint);
    }
    // FIXME -- this is a good place to do all sorts of other sanity checks,
    // like seeing if any nans have crept in from each op.
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
        sym.step (sym.size());
        if (preserve_value || ! all_points_on()) {
            // Propagate the value from slot 0 to other slots
            size_t size = sym.size();
            char *data = (char *) sym.data();
            for (int i = 1;  i < m_npoints;  ++i)
                memcpy (data + i*size, data, size);
        }
    } else {
        // sym is varying, but we're assigning a new uniform value AND
        // we're not inside a conditional.  Safe to demote sym to uniform.
        if (sym.symtype() != SymTypeGlobal) // DO NOT demote a global
            sym.step (0);
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


}; // namespace pvt
}; // namespace OSL
