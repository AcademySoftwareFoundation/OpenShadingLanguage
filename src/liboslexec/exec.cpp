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



namespace OSL {
namespace pvt {   // OSL::pvt



ShadingExecution::ShadingExecution ()
    : m_context(NULL), m_instance(NULL), m_master(NULL),
      m_bound(false)
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

    std::cerr << "bind ctx " << (void *)context << " use " 
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
    m_symbols = m_instance->m_symbols;

    // FIXME: bind the symbols -- get the syms ready and pointing to the
    // right place in the heap,, interpolate primitive variables, handle
    // connections, initialize all parameters
    BOOST_FOREACH (Symbol &sym, m_symbols) {
        std::cerr << "  bind " << sym.mangled() 
                  << ", offset " << sym.dataoffset() << "\n";
        if (sym.symtype() == SymTypeGlobal) {
            if (sym.dataoffset() >= 0) {
                sym.data (m_context->heapaddr (sym.dataoffset()));
            } else {
                // ASSERT (sym.dataoffset() >= 0 &&
                //         "Global ought to already have a dataoffset");
                // Skip this for now -- it includes L, Cl, etc.
            }
            sym.step (0);  // FIXME
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
        std::cerr << "  bound " << sym.mangled() << " to address " 
                  << (void *)sym.data() << ", step " << sym.step() << "\n";
    }

    m_bound = true;
    m_executed = false;
}



void
ShadingExecution::run (Runflag *rf)
{
    if (m_executed)
        return;       // Already executed

    std::cerr << "Running ShadeExec " << (void *)this << ", shader " 
              << m_master->shadername() << "\n";

    ASSERT (m_bound);  // We'd better be bound at this point

    // Make space for new runflags
    m_runflags = (Runflag *) alloca (m_npoints * sizeof(Runflag));
    if (rf) {
        // Passed runflags -- copy those
        memcpy (m_runflags, rf, m_npoints*sizeof(Runflag));
        // FIXME -- restrict begin/end
        m_beginpoint = 0;
        m_endpoint = m_npoints;
        m_allpointson = true;
    } else {
        // If not passed runflags, make new ones
        for (int i = 0;  i < m_npoints;  ++i)
            m_runflags[i] = 1;
        m_beginpoint = 0;
        m_endpoint = m_npoints;
        m_allpointson = true;
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
    std::cerr << "Running ShadeExec " << (void *)this 
              << ", shader " << m_master->shadername() 
              << " ops [" << beginop << "," << endop << ")\n";
    for (m_ip = beginop; m_ip < endop && m_beginpoint < m_endpoint;  ++m_ip) {
        Opcode &op (this->op ());
        std::cerr << "  instruction " << m_ip << ": " << op.opname() << " ";
        for (int i = 0;  i < op.nargs();  ++i) {
            int arg = m_master->m_args[op.firstarg()+i];
            std::cerr << m_instance->symbol(arg)->mangled() << " ";
        }
        std::cerr << "\n";
        ASSERT (op.implementation() && "Unimplemented op!");
        op (this, op.nargs(), &m_master->m_args[op.firstarg()],
            m_runflags, m_beginpoint, m_endpoint);
    }
    // FIXME -- this is a good place to do all sorts of other sanity checks,
    // like seeing if any nans have crept in from each op.
}



}; // namespace pvt
}; // namespace OSL
