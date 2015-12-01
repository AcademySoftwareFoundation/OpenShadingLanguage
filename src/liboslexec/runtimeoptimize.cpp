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
#include <cstdio>
#include <cmath>

#include <boost/foreach.hpp>

#include <OpenImageIO/sysutil.h>
#include <OpenImageIO/timer.h>
#include <OpenImageIO/thread.h>

#include "oslexec_pvt.h"
#include "runtimeoptimize.h"
#include "../liboslcomp/oslcomp_pvt.h"
using namespace OSL;
using namespace OSL::pvt;


// names of ops we'll be using frequently
static ustring u_nop    ("nop"),
               u_assign ("assign"),
               u_add    ("add"),
               u_sub    ("sub"),
               u_mul    ("mul"),
               u_if     ("if"),
               u_for    ("for"),
               u_while  ("while"),
               u_dowhile("dowhile"),
               u_functioncall ("functioncall"),
               u_break ("break"),
               u_continue ("continue"),
               u_return ("return"),
               u_useparam ("useparam"),
               u_closure ("closure"),
               u_pointcloud_write ("pointcloud_write"),
               u_isconnected ("isconnected"),
               u_setmessage ("setmessage"),
               u_getmessage ("getmessage"),
               u_getattribute ("getattribute");


OSL_NAMESPACE_ENTER

namespace pvt {   // OSL::pvt

using OIIO::spin_lock;
using OIIO::Timer;

DECLFOLDER(constfold_assign);  // forward decl



/// Wrapper that erases elements of c for which predicate p is true.
/// (Unlike std::remove_if, it resizes the container so that it contains
/// ONLY elements for which the predicate is true.)
template<class Container, class Predicate>
void erase_if (Container &c, const Predicate &p)
{
    c.erase (std::remove_if (c.begin(), c.end(), p), c.end());
}



OSOProcessorBase::OSOProcessorBase (ShadingSystemImpl &shadingsys,
                                    ShaderGroup &group, ShadingContext *ctx)
    : m_shadingsys(shadingsys),
      m_group(group),
      m_context(ctx),
      m_debug(shadingsys.debug()),
      m_inst(NULL)
{
    set_debug ();
}



OSOProcessorBase::~OSOProcessorBase ()
{
}



RuntimeOptimizer::RuntimeOptimizer (ShadingSystemImpl &shadingsys,
                                    ShaderGroup &group, ShadingContext *ctx)
    : OSOProcessorBase(shadingsys, group, ctx),
      m_optimize(shadingsys.optimize()),
      m_opt_simplify_param(shadingsys.m_opt_simplify_param),
      m_opt_constant_fold(shadingsys.m_opt_constant_fold),
      m_opt_stale_assign(shadingsys.m_opt_stale_assign),
      m_opt_elide_useless_ops(shadingsys.m_opt_elide_useless_ops),
      m_opt_elide_unconnected_outputs(shadingsys.m_opt_elide_unconnected_outputs),
      m_opt_peephole(shadingsys.m_opt_peephole),
      m_opt_coalesce_temps(shadingsys.m_opt_coalesce_temps),
      m_opt_assign(shadingsys.m_opt_assign),
      m_opt_mix(shadingsys.m_opt_mix),
      m_opt_middleman(shadingsys.m_opt_middleman),
      m_pass(0),
      m_next_newconst(0), m_next_newtemp(0),
      m_stat_opt_locking_time(0), m_stat_specialization_time(0),
      m_stop_optimizing(false)
{
    memset (&m_shaderglobals, 0, sizeof(ShaderGlobals));
    m_shaderglobals.context = shadingcontext();
}



RuntimeOptimizer::~RuntimeOptimizer ()
{
}



void
OSOProcessorBase::set_inst (int newlayer)
{
    m_layer = newlayer;
    m_inst = group()[m_layer];
    ASSERT (m_inst != NULL);
    set_debug ();
}



void
RuntimeOptimizer::set_inst (int newlayer)
{
    OSOProcessorBase::set_inst (newlayer);
    m_all_consts.clear ();
    m_symbol_aliases.clear ();
    m_block_aliases.clear ();
    m_param_aliases.clear ();
}



void
OSOProcessorBase::set_debug ()
{
    // start with the shading system's idea of debugging level
    m_debug = shadingsys().debug();

    // If either group or layer was specified for debug, surely they want
    // debugging turned on.
    if (shadingsys().debug_groupname() || shadingsys().debug_layername())
        m_debug = std::max (m_debug, 1);

    // Force debugging off if a specific group was selected for debug
    // and we're not it, or a specific layer was selected for debug and
    // we're not it.
    bool wronggroup = (shadingsys().debug_groupname() && 
                       shadingsys().debug_groupname() != group().name());
    bool wronglayer = (shadingsys().debug_layername() && inst() &&
                       shadingsys().debug_layername() != inst()->layername());
    if (wronggroup || wronglayer)
        m_debug = 0;
}



void
RuntimeOptimizer::set_debug ()
{
    OSOProcessorBase::set_debug ();

    // If a specific group is isolated for debugging and  the
    // 'optimize_dondebug' flag is on, fully optimize all other groups.
    if (shadingsys().debug_groupname() &&
        shadingsys().debug_groupname() != group().name()) {
        if (shadingsys().m_optimize_nondebug) {
            // Debugging trick: if user said to only debug one group, turn
            // on full optimization for all others!  This prevents
            // everything from running 10x slower just because you want to
            // debug one shader.
            m_optimize = 3;
            m_opt_simplify_param = true;
            m_opt_constant_fold = true;
            m_opt_stale_assign = true;
            m_opt_elide_useless_ops = true;
            m_opt_elide_unconnected_outputs = true;
            m_opt_peephole = true;
            m_opt_coalesce_temps = true;
            m_opt_assign = true;
            m_opt_mix = true;
            m_opt_middleman = true;
        }
    }
}



int
RuntimeOptimizer::find_constant (const TypeSpec &type, const void *data)
{
    for (int i = 0;  i < (int)m_all_consts.size();  ++i) {
        const Symbol &s (*inst()->symbol(m_all_consts[i]));
        ASSERT (s.symtype() == SymTypeConst);
        if (equivalent (s.typespec(), type) &&
              !memcmp (s.data(), data, s.typespec().simpletype().size())) {
            return m_all_consts[i];
        }
    }
    return -1;
}



int
RuntimeOptimizer::add_constant (const TypeSpec &type, const void *data,
                                TypeDesc datatype)
{
    int ind = find_constant (type, data);
    if (ind < 0) {
        // support varlen arrays
        TypeSpec newtype = type;
        if (type.is_unsized_array())
            newtype.make_array (datatype.numelements());

        Symbol newconst (ustring::format ("$newconst%d", m_next_newconst++),
                         newtype, SymTypeConst);
        void *newdata;
        TypeDesc t (newtype.simpletype());
        size_t n = t.aggregate * t.numelements();
        if (datatype == TypeDesc::UNKNOWN)
            datatype = t;
        size_t datan = datatype.aggregate * datatype.numelements();
        if (t.basetype == TypeDesc::INT &&
                datatype.basetype == TypeDesc::INT && n == datan) {
            newdata = inst()->shadingsys().alloc_int_constants (n);
            memcpy (newdata, data, t.size());
        } else if (t.basetype == TypeDesc::FLOAT &&
                   datatype.basetype == TypeDesc::FLOAT) {
            newdata = inst()->shadingsys().alloc_float_constants (n);
            if (n == datan)
                for (size_t i = 0;  i < n;  ++i)
                    ((float *)newdata)[i] = ((const float *)data)[i];
            else if (datan == 1)
                for (size_t i = 0;  i < n;  ++i)
                    ((float *)newdata)[i] = ((const float *)data)[0];
            else {
                ASSERT (0 && "unsupported type for add_constant");
            }
        } else if (t.basetype == TypeDesc::FLOAT &&
                   datatype.basetype == TypeDesc::INT) {
            newdata = inst()->shadingsys().alloc_float_constants (n);
            if (n == datan)
                for (size_t i = 0;  i < n;  ++i)
                    ((float *)newdata)[i] = ((const int *)data)[i];
            else if (datan == 1)
                for (size_t i = 0;  i < n;  ++i)
                    ((float *)newdata)[i] = ((const int *)data)[0];
            else {
                ASSERT (0 && "unsupported type for add_constant");
            }
        } else if (t.basetype == TypeDesc::STRING &&
                   datatype.basetype == TypeDesc::STRING && n == datan) {
            newdata = inst()->shadingsys().alloc_string_constants (n);
            memcpy (newdata, data, t.size());
        } else {
            ASSERT (0 && "unsupported type for add_constant");
        }
        newconst.data (newdata);
        ind = add_symbol (newconst);
        m_all_consts.push_back (ind);
    }
    return ind;
}



int
RuntimeOptimizer::add_temp (const TypeSpec &type)
{
    return add_symbol (Symbol (ustring::format ("$opttemp%d", m_next_newtemp++),
                               type, SymTypeTemp));
}



int
RuntimeOptimizer::add_global (ustring name, const TypeSpec &type)
{
    int index = inst()->findsymbol (name);
    if (index < 0)
        index = add_symbol (Symbol (name, type, SymTypeGlobal));
    return index;
}



int
RuntimeOptimizer::add_symbol (const Symbol &sym)
{
    size_t index = inst()->symbols().size ();
    ASSERT (inst()->symbols().capacity() > index &&
            "we shouldn't have to realloc here");
    inst()->symbols().push_back (sym);
    // Mark the symbol as always read.  Next time we recompute symbol
    // lifetimes, it'll get the correct range for when it's read and
    // written.  But for now, just make sure it doesn't accidentally
    // look entirely unused.
    inst()->symbols().back().mark_always_used ();
    return (int) index;
}



void
RuntimeOptimizer::debug_opt_impl (string_view message) const
{
    static OIIO::spin_mutex mutex;
    OIIO::spin_lock lock (mutex);
    std::cout << message;
}



void
RuntimeOptimizer::debug_opt_ops (int opbegin, int opend, string_view message) const
{
    const Opcode &op (inst()->ops()[opbegin]);
    std::string oprange;
    if (opbegin >= 0 && opend-opbegin > 1)
        oprange = Strutil::format ("ops %d-%d ", opbegin, opend);
    else if (opbegin >= 0)
        oprange = Strutil::format ("op %d ", opbegin);
    debug_opt ("%s%s (@ %s:%d)\n", oprange, message,
               op.sourcefile(), op.sourceline());
}



void
RuntimeOptimizer::debug_turn_into (const Opcode &op, int numops,
                                   string_view newop,
                                   int newarg0, int newarg1, int newarg2,
                                   string_view why)
{
    int opnum = &op - &(inst()->ops()[0]);
    std::string msg;
    if (numops == 1)
        msg = Strutil::format ("turned '%s' to '%s", op_string(op), newop);
    else
        msg = Strutil::format ("turned to '%s", newop);
    if (newarg0 >= 0)
        msg += Strutil::format (" %s", inst()->symbol(newarg0)->name());
    if (newarg1 >= 0)
        msg += Strutil::format (" %s", inst()->symbol(newarg1)->name());
    if (newarg2 >= 0)
        msg += Strutil::format (" %s", inst()->symbol(newarg2)->name());
    msg += "'";
    if (why.size())
        msg += Strutil::format (" : %s", why);
    debug_opt_ops (opnum, opnum+numops, msg);
}



void
RuntimeOptimizer::turn_into_new_op (Opcode &op, ustring newop, int newarg0,
                                    int newarg1, int newarg2, string_view why)
{
    int opnum = &op - &(inst()->ops()[0]);
    DASSERT (opnum >= 0 && opnum < (int)inst()->ops().size());
    if (debug() > 1)
        debug_turn_into (op, 1, newop, newarg0, newarg1, newarg2, why);
    op.reset (newop, newarg2<0 ? 2 : 3);
    inst()->args()[op.firstarg()+0] = newarg0;
    op.argwriteonly (0);
    opargsym(op, 0)->mark_rw (opnum, false, true);
    inst()->args()[op.firstarg()+1] = newarg1;
    op.argreadonly (1);
    opargsym(op, 1)->mark_rw (opnum, true, false);
    if (newarg2 >= 0) {
        inst()->args()[op.firstarg()+2] = newarg2;
        op.argreadonly (2);
        opargsym(op, 2)->mark_rw (opnum, true, false);
    }
}



void
RuntimeOptimizer::turn_into_assign (Opcode &op, int newarg, string_view why)
{
    // We don't know the op num here, so we subtract the pointers
    int opnum = &op - &(inst()->ops()[0]);
    if (debug() > 1)
        debug_turn_into (op, 1, "assign", oparg(op,0), newarg, -1, why);
    op.reset (u_assign, 2);
    inst()->args()[op.firstarg()+1] = newarg;
    op.argwriteonly (0);
    op.argread (1, true);
    op.argwrite (1, false);
    // Need to make sure the symbol we're assigning is marked as read
    // for this op.
    DASSERT (opnum >= 0 && opnum < (int)inst()->ops().size());
    Symbol *arg = opargsym (op, 1);
    arg->mark_rw (opnum, true, false);
}



// Turn the current op into a simple assignment to zero (of the first arg).
void
RuntimeOptimizer::turn_into_assign_zero (Opcode &op, string_view why)
{
    static float zero[16] = { 0, 0, 0, 0,  0, 0, 0, 0,
                              0, 0, 0, 0,  0, 0, 0, 0 };
    Symbol &R (*(inst()->argsymbol(op.firstarg()+0)));
    int cind = add_constant (R.typespec(), &zero);
    turn_into_assign (op, cind, why);
}



// Turn the current op into a simple assignment to one (of the first arg).
void
RuntimeOptimizer::turn_into_assign_one (Opcode &op, string_view why)
{
    Symbol &R (*(inst()->argsymbol(op.firstarg()+0)));
    if (R.typespec().is_int()) {
        int one = 1;
        int cind = add_constant (R.typespec(), &one);
        turn_into_assign (op, cind, why);
    } else {
        ASSERT (R.typespec().is_triple() || R.typespec().is_float());
        static float one[3] = { 1, 1, 1 };
        int cind = add_constant (R.typespec(), &one);
        turn_into_assign (op, cind, why);
    }
}



// Turn the op into a no-op
int
RuntimeOptimizer::turn_into_nop (Opcode &op, string_view why)
{
    if (op.opname() != u_nop) {
        if (debug() > 1)
            debug_turn_into (op, 1, "nop", -1, -1, -1, why);
        op.reset (u_nop, 0);
        return 1;
    }
    return 0;
}



int
RuntimeOptimizer::turn_into_nop (int begin, int end, string_view why)
{
    int changed = 0;
    for (int i = begin;  i < end;  ++i) {
        Opcode &op (inst()->ops()[i]);
        if (op.opname() != u_nop) {
            op.reset (u_nop, 0);
            ++changed;
        }
    }
    if (debug() > 1 && changed)
        debug_turn_into (inst()->ops()[begin], end-begin, "nop", -1, -1, -1, why);
    return changed;
}



void
RuntimeOptimizer::insert_code (int opnum, ustring opname,
                               const int *argsbegin, const int *argsend,
                               RecomputeRWRangesOption recompute_rw_ranges,
                               InsertRelation relation)
{
    OpcodeVec &code (inst()->ops());
    std::vector<int> &opargs (inst()->args());
    ustring method = (opnum < (int)code.size()) ? code[opnum].method() : OSLCompilerImpl::main_method_name();
    int nargs = argsend - argsbegin;
    Opcode op (opname, method, opargs.size(), nargs);
    code.insert (code.begin()+opnum, op);
    opargs.insert (opargs.end(), argsbegin, argsend);
    if (opnum < inst()->m_maincodebegin)
        ++inst()->m_maincodebegin;
    ++inst()->m_maincodeend;
    if ((relation == -1 && opnum > 0) ||
        (relation == 1 && opnum < (int)code.size()-1)) {
        code[opnum].method (code[opnum+relation].method());
        code[opnum].source (code[opnum+relation].sourcefile(),
                            code[opnum+relation].sourceline());
    }

    // Unless we were inserting at the end, we may need to adjust
    // the jump addresses of other ops and the param init ranges.
    if (opnum < (int)code.size()-1) {
        // Adjust jump offsets
        for (size_t n = 0;  n < code.size();  ++n) {
            Opcode &c (code[n]);
            for (int j = 0; j < (int)Opcode::max_jumps && c.jump(j) >= 0; ++j) {
                if (c.jump(j) > opnum) {
                    c.jump(j) = c.jump(j) + 1;
                    // std::cerr << "Adjusting jump target at op " << n << "\n";
                }
            }
        }
        // Adjust param init ranges
        FOREACH_PARAM (Symbol &s, inst()) {
            if (s.initbegin() > opnum)
                s.initbegin (s.initbegin()+1);
            if (s.initend() > opnum)
                s.initend (s.initend()+1);
        }
    }

    // Inserting the instruction may change the read/write ranges of
    // symbols.  Not adjusting this can throw off other optimizations.
    if (recompute_rw_ranges) {
        BOOST_FOREACH (Symbol &s, inst()->symbols()) {
            if (s.everread()) {
                int first = s.firstread(), last = s.lastread();
                if (first >= opnum)
                    ++first;
                if (last >= opnum)
                    ++last;
                s.set_read (first, last);
            }
            if (s.everwritten()) {
                int first = s.firstwrite(), last = s.lastwrite();
                if (first >= opnum)
                    ++first;
                if (last >= opnum)
                    ++last;
                s.set_write (first, last);
            }
        }
    }

    // Adjust the basic block IDs and which instructions are inside
    // conditionals.
    if (m_bblockids.size()) {
        ASSERT (m_bblockids.size() == code.size()-1);
        m_bblockids.insert (m_bblockids.begin()+opnum, 1, m_bblockids[opnum]);
    }
    if (m_in_conditional.size()) {
        ASSERT (m_in_conditional.size() == code.size()-1);
        m_in_conditional.insert (m_in_conditional.begin()+opnum, 1,
                                 m_in_conditional[opnum]);
    }
    if (m_in_loop.size()) {
        ASSERT (m_in_loop.size() == code.size()-1);
        m_in_loop.insert (m_in_loop.begin()+opnum, 1,
                          m_in_loop[opnum]);
    }
    // If the first return happened after this, bump it up
    if (m_first_return >= opnum)
        ++m_first_return;

    if (opname == u_if) {
        // special case for 'if' -- the arg is read, not written
        inst()->symbol(argsbegin[0])->mark_rw (opnum, true, false);
    }
    else if (opname != u_useparam) {
        // Mark the args as being used for this op (assume that the
        // first is written, the others are read).  Enforce that with an
        // DASSERT to be sure we only use insert_code for the couple of
        // instructions that we think it is used for.
        for (int a = 0;  a < nargs;  ++a)
            inst()->symbol(argsbegin[a])->mark_rw (opnum, a>0, a==0);
    }
}



void
RuntimeOptimizer::insert_code (int opnum, ustring opname,
                               const std::vector<int> &args_to_add,
                               RecomputeRWRangesOption recompute_rw_ranges,
                               InsertRelation relation)
{
    const int *argsbegin = (args_to_add.size())? &args_to_add[0]: NULL;
    const int *argsend = argsbegin + args_to_add.size();

    insert_code (opnum, opname, argsbegin, argsend,
                 recompute_rw_ranges, relation);
}



void
RuntimeOptimizer::insert_code (int opnum, ustring opname,
                               InsertRelation relation,
                               int arg0, int arg1, int arg2, int arg3)
{
    int args[4];
    int nargs = 0;
    if (arg0 >= 0) args[nargs++] = arg0;
    if (arg1 >= 0) args[nargs++] = arg1;
    if (arg2 >= 0) args[nargs++] = arg2;
    if (arg3 >= 0) args[nargs++] = arg3;
    insert_code (opnum, opname, args, args+nargs, RecomputeRWRanges, relation);
}



/// Insert a 'useparam' instruction in front of instruction 'opnum', to
/// reference the symbols in 'params'.
void
RuntimeOptimizer::insert_useparam (size_t opnum,
                                   const std::vector<int> &params_to_use)
{
    ASSERT (params_to_use.size() > 0);
    OpcodeVec &code (inst()->ops());
    insert_code (opnum, u_useparam, params_to_use,
                 RecomputeRWRanges, GroupWithNext);

    // All ops are "read"
    code[opnum].argwrite (0, false);
    code[opnum].argread (0, true);
    if (opnum < code.size()-1) {
        // We have no parse node, but we set the new instruction's
        // "source" to the one of the statement right after.
        code[opnum].source (code[opnum+1].sourcefile(),
                            code[opnum+1].sourceline());
        // Set the method id to the same as the statement right after
        code[opnum].method (code[opnum+1].method());
    } else {
        // If there IS no "next" instruction, just call it main
        code[opnum].method (OSLCompilerImpl::main_method_name());
    }
}



void
RuntimeOptimizer::add_useparam (SymbolPtrVec &allsyms)
{
    OpcodeVec &code (inst()->ops());
    std::vector<int> &opargs (inst()->args());

    // Mark all symbols as un-initialized
    BOOST_FOREACH (Symbol &s, inst()->symbols())
        s.initialized (false);

    if (inst()->m_maincodebegin < 0)
        inst()->m_maincodebegin = (int)code.size();

    // Take care of the output params right off the bat -- as soon as the
    // shader starts running 'main'.
    std::vector<int> outputparams;
    for (int i = 0;  i < (int)inst()->symbols().size();  ++i) {
        Symbol *s = inst()->symbol(i);
        if (s->symtype() == SymTypeOutputParam &&
            (s->connected() || (s->valuesource() == Symbol::DefaultVal && s->has_init_ops()))) {
            outputparams.push_back (i);
            s->initialized (true);
        }
    }
    if (outputparams.size())
        insert_useparam (inst()->m_maincodebegin, outputparams);

    // Figure out which statements are inside conditional states
    find_conditionals ();

    // Loop over all ops...
    for (int opnum = 0;  opnum < (int)code.size();  ++opnum) {
        Opcode &op (code[opnum]);  // handy ref to the op
        if (op.opname() == u_useparam)
            continue;  // skip useparam ops themselves, if we hit one
        bool simple_assign = is_simple_assign(op);
        bool in_main_code = (opnum >= inst()->m_maincodebegin);
        std::vector<int> params;   // list of params referenced by this op
        // For each argument...
        for (int a = 0;  a < op.nargs();  ++a) {
            int argind = op.firstarg() + a;
            SymbolPtr s = inst()->argsymbol (argind);
            DASSERT (s->dealias() == s);
            // If this arg is a param and is read, remember it
            if (s->symtype() != SymTypeParam && s->symtype() != SymTypeOutputParam)
                continue;  // skip non-params
            // skip if we've already 'usedparam'ed it unconditionally
            if (s->initialized() && in_main_code)
                continue;

            bool inside_init = (opnum >= s->initbegin() && opnum < s->initend());
            if (op.argread(a) || (op.argwrite(a) && !inside_init)) {
                // Don't add it more than once
                if (std::find (params.begin(), params.end(), opargs[argind]) == params.end()) {
                    // If this arg is the one being written to by a
                    // "simple" assignment, it doesn't need a useparam here.
                    if (! (simple_assign && a == 0))
                        params.push_back (opargs[argind]);
                    // mark as already initialized unconditionally, if we do
                    if (op_is_unconditionally_executed(opnum) &&
                            op.method() == OSLCompilerImpl::main_method_name())
                        s->initialized (true);
                }
            }
        }

        // If the arg we are examining read any params, insert a "useparam"
        // op whose arguments are the list of params we are about to use.
        if (params.size()) {
            insert_useparam (opnum, params);
            // Skip the op we just added
            ++opnum;
        }
    }

    // Mark all symbols as un-initialized
    BOOST_FOREACH (Symbol &s, inst()->symbols())
        s.initialized (false);

    // Re-track variable lifetimes, since the inserted useparam
    // instructions will have change the instruction numbers.
    track_variable_lifetimes (allsyms);
}



bool
OSOProcessorBase::is_zero (const Symbol &A)
{
    if (! A.is_constant())
        return false;
    const TypeSpec &Atype (A.typespec());
    static Vec3 Vzero (0, 0, 0);
    return (Atype.is_float() && *(const float *)A.data() == 0) ||
        (Atype.is_int() && *(const int *)A.data() == 0) ||
        (Atype.is_triple() && *(const Vec3 *)A.data() == Vzero);
}



bool
OSOProcessorBase::is_one (const Symbol &A)
{
    if (! A.is_constant())
        return false;
    const TypeSpec &Atype (A.typespec());
    static Vec3 Vone (1, 1, 1);
    static Matrix44 Mone (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    return (Atype.is_float() && *(const float *)A.data() == 1) ||
        (Atype.is_int() && *(const int *)A.data() == 1) ||
        (Atype.is_triple() && *(const Vec3 *)A.data() == Vone) ||
        (Atype.is_matrix() && *(const Matrix44 *)A.data() == Mone);
}



std::string
OSOProcessorBase::const_value_as_string (const Symbol &A)
{
    if (! A.is_constant())
        return std::string();
    TypeDesc type (A.typespec().simpletype());
    int n = type.numelements() * type.aggregate;
    std::ostringstream s;
    if (type.basetype == TypeDesc::FLOAT) {
        for (int i = 0; i < n; ++i)
            s << (i ? "," : "") << ((const float *)A.data())[i];
    } else if (type.basetype == TypeDesc::INT) {
        for (int i = 0; i < n; ++i)
            s << (i ? "," : "") << ((const int *)A.data())[i];
    } else if (type.basetype == TypeDesc::STRING) {
        for (int i = 0; i < n; ++i)
            s << (i ? "," : "") << '\"' << ((const ustring *)A.data())[i] << '\"';
    }
    return s.str();
}



void
RuntimeOptimizer::register_message (ustring name)
{
    m_local_messages_sent.push_back (name);
}



void
RuntimeOptimizer::register_unknown_message ()
{
    m_local_unknown_message_sent = true;
}



bool
RuntimeOptimizer::message_possibly_set (ustring name) const
{
    return m_local_unknown_message_sent || m_unknown_message_sent ||
        std::find (m_messages_sent.begin(), m_messages_sent.end(), name) != m_messages_sent.end() ||
        std::find (m_local_messages_sent.begin(), m_local_messages_sent.end(), name) != m_local_messages_sent.end();
}



/// For all the instance's parameters (that can't be overridden by the
/// geometry), if they can be found to be effectively constants or
/// globals, make constants for them and alias them to the constant. If
/// they are connected to an earlier layer's output, if it can determine
/// that the output will be a constant or global, then sever the
/// connection and just alias our parameter to that value.
void
RuntimeOptimizer::simplify_params ()
{
    for (int i = inst()->firstparam();  i < inst()->lastparam();  ++i) {
        Symbol *s (inst()->symbol(i));
        if (s->symtype() != SymTypeParam)
            continue;  // Skip non-params
                       // FIXME - clever things we can do for OutputParams?
        if (! s->lockgeom())
            continue;  // Don't mess with params that can change with the geom
        if (s->typespec().is_structure() || s->typespec().is_closure_based())
            continue;  // We don't mess with struct placeholders or closures

        if (s->valuesource() == Symbol::InstanceVal) {
            // Instance value -- turn it into a constant and remove init ops
            make_symbol_room (1);
            s = inst()->symbol(i);  // In case make_symbol_room changed ptrs
            int cind = add_constant (s->typespec(), s->data());
            global_alias (i, cind); // Alias this symbol to the new const
            turn_into_nop (s->initbegin(), s->initend(),
                           "instance value doesn't need init ops");
        } else if (s->valuesource() == Symbol::DefaultVal && !s->has_init_ops()) {
            // Plain default value without init ops -- turn it into a constant
            make_symbol_room (1);
            s = inst()->symbol(i);  // In case make_symbol_room changed ptrs
            int cind = add_constant (s->typespec(), s->data(), s->typespec().simpletype());
            global_alias (i, cind); // Alias this symbol to the new const
        } else if (s->valuesource() == Symbol::DefaultVal && s->has_init_ops()) {
            // Default val comes from init ops -- special cases?  Yes,
            // if it's a simple assignment from a global whose value is
            // not reassigned later, we can just alias it, and if we're
            // lucky that may eliminate all uses of the parameter.
            if (s->initbegin() == s->initend()-1) {  // just one op
                Opcode &op (inst()->ops()[s->initbegin()]);
                if (op.opname() == u_assign) {
                    // The default value has init ops, but they consist of
                    // just a single assignment op...
                    Symbol *src = inst()->argsymbol(op.firstarg()+1);
                    // Is it assigning a global, or a parameter that's
                    // got a default or instance value and isn't on the geom,
                    // and its value is never changed and the types match?
                    if ((src->symtype() == SymTypeGlobal ||
                         src->symtype() == SymTypeConst ||
                         (src->symtype() == SymTypeParam && src->lockgeom() &&
                          (src->valuesource() == Symbol::DefaultVal ||
                           src->valuesource() == Symbol::InstanceVal)))
                        && !src->everwritten()
                        && equivalent(src->typespec(), s->typespec())) {
                        // Great, so let's remember the alias.  We can't
                        // call global_alias() here, because we're still in
                        // init ops, that'll screw us up.  So we just record
                        // it in m_param_aliases and then we'll establish
                        // the global aliases when we hit the main code.
                        m_param_aliases[i] = inst()->arg(op.firstarg()+1);
                    }
                }
            }
        } else if (s->valuesource() == Symbol::ConnectedVal) {
            // It's connected to an earlier layer.  If the output var of
            // the upstream shader is effectively constant or a global,
            // then so is this variable.
            turn_into_nop (s->initbegin(), s->initend(),
                           "connected value doesn't need init ops");
            BOOST_FOREACH (Connection &c, inst()->connections()) {
                if (c.dst.param == i) {
                    // srcsym is the earlier group's output param, which
                    // is connected as the input to the param we're
                    // examining.
                    ShaderInstance *uplayer = group()[c.srclayer];
                    Symbol *srcsym = uplayer->symbol(c.src.param);
                    if (!srcsym->lockgeom())
                        continue; // Not if it can be overridden by geometry

                    // Is the source symbol known to be a global, from
                    // earlier analysis by find_params_holding_globals?
                    // If so, make sure the global is in this instance's
                    // symbol table, and alias the parameter to it.
                    ustringmap_t &g (m_params_holding_globals[c.srclayer]);
                    ustringmap_t::const_iterator f;
                    f = g.find (srcsym->name());
                    if (f != g.end()) {
                        if (debug() > 1)
                            debug_opt ("Remapping %s.%s because it's connected to "
                                       "%s.%s, which is known to be %s\n",
                                       inst()->layername(), s->name(),
                                       uplayer->layername(), srcsym->name(),
                                       f->second);
                        make_symbol_room (1);
                        s = inst()->symbol(i);  // In case make_symbol_room changed ptrs
                        int ind = add_global (f->second, srcsym->typespec());
                        global_alias (i, ind);
                        shadingsys().m_stat_global_connections += 1;
                        break;
                    }

                    if (!srcsym->everwritten() &&
                        (srcsym->valuesource() == Symbol::DefaultVal ||
                         srcsym->valuesource() == Symbol::InstanceVal) &&
                        !srcsym->has_init_ops()) {
                        make_symbol_room (1);
                        s = inst()->symbol(i);  // In case make_symbol_room changed ptrs
                        int cind = add_constant (s->typespec(), srcsym->data(),
                                                 srcsym->typespec().simpletype());
                        // Alias this symbol to the new const
                        global_alias (i, cind);
                        make_param_use_instanceval (s, "- upstream layer sets it to a constant");
                        replace_param_value (s, srcsym->data(), srcsym->typespec());
                        shadingsys().m_stat_const_connections += 1;
                        break;
                    }
                }
            }
        }
    }
}



/// For all the instance's parameters, if they are simply assigned globals,
/// record that in m_params_holding_globals.
void
RuntimeOptimizer::find_params_holding_globals ()
{
    FOREACH_PARAM (Symbol &s, inst()) {
        // Skip if this isn't a shader output parameter that's connected
        // to a later layer.
        if (s.symtype() != SymTypeParam && s.symtype() != SymTypeOutputParam)
            continue;  // Skip non-params
        if (!s.connected_down())
            continue;  // Skip unconnected params -- who cares
        if (s.valuesource() != Symbol::DefaultVal)
            continue;  // Skip -- must be connected or an instance value
        if (s.firstwrite() < 0 || s.firstwrite() != s.lastwrite())
            continue;  // Skip -- written more than once

        int opnum = s.firstwrite();
        Opcode &op (inst()->ops()[opnum]);
        if (op.opname() != u_assign || ! op_is_unconditionally_executed(opnum))
            continue;   // Not a simple assignment unconditionally performed

        // what s is assigned from (fully dealiased)
        Symbol *src = inst()->symbol (dealias_symbol (oparg (op, 1), opnum));

        if (src->symtype() != SymTypeGlobal)
            continue;   // only interested in global assignments

        if (debug() > 1)
            debug_opt ("I think that %s.%s will always be %s\n",
                       inst()->layername(), s.name(), src->name());
        m_params_holding_globals[layer()][s.name()] = src->name();
    }
}



void
OSOProcessorBase::find_conditionals ()
{
    OpcodeVec &code (inst()->ops());

    m_in_conditional.clear ();
    m_in_conditional.resize (code.size(), false);
    m_in_loop.clear ();
    m_in_loop.resize (code.size(), false);
    m_first_return = (int)code.size();
    for (int i = 0;  i < (int)code.size();  ++i) {
        if (code[i].jump(0) >= 0) {
            std::fill (m_in_conditional.begin()+i,
                       m_in_conditional.begin()+code[i].farthest_jump(), true);
            if (code[i].opname() == Strings::op_dowhile ||
                  code[i].opname() == Strings::op_for ||
                  code[i].opname() == Strings::op_while) {
                std::fill (m_in_loop.begin()+i,
                           m_in_loop.begin()+code[i].farthest_jump(), true);
            }
        }
        if (code[i].opname() == Strings::op_exit)
            m_first_return = std::min (m_first_return, i);
    }
}



void
OSOProcessorBase::find_basic_blocks ()
{
    OpcodeVec &code (inst()->ops());

    // Start by setting all basic block IDs to 0
    m_bblockids.clear ();
    m_bblockids.resize (code.size(), 0);

    // First, keep track of all the spots where blocks begin
    std::vector<bool> block_begin (code.size(), false);

    // Init ops start basic blocks
    FOREACH_PARAM (const Symbol &s, inst()) {
        if (s.has_init_ops())
            block_begin[s.initbegin()] = true;
    }

    // Main code starts a basic block
    block_begin[inst()->maincodebegin()] = true;

    for (size_t opnum = 0;  opnum < code.size();  ++opnum) {
        Opcode &op (code[opnum]);
        // Anyplace that's the target of a jump instruction starts a basic block
        for (int j = 0;  j < (int)Opcode::max_jumps;  ++j) {
            if (op.jump(j) >= 0)
                block_begin[op.jump(j)] = true;
            else
                break;
        }
        // The first instruction in a conditional or loop (which is not
        // itself a jump target) also begins a basic block.  If the op has
        // any jump targets at all, it must be a conditional or loop.
        if (op.jump(0) >= 0)
            block_begin[opnum+1] = true;
        // 'break', 'continue', and 'return' also cause the next
        // statement to begin a new basic block.
        if (op.opname() == u_break || op.opname() == u_continue ||
                op.opname() == u_return)
            block_begin[opnum+1] = true;
    }

    // Now color the blocks with unique identifiers
    int bbid = 1;  // next basic block ID to use
    for (size_t opnum = 0;  opnum < code.size();  ++opnum) {
        if (block_begin[opnum])
            ++bbid;
        m_bblockids[opnum] = bbid;
    }
}



/// For 'R = A_const' where R and A are different, but coerceable,
/// types, turn it into a constant assignment of the exact type.
/// Return true if a change was made, otherwise return false.
bool
RuntimeOptimizer::coerce_assigned_constant (Opcode &op)
{
    ASSERT (op.opname() == u_assign);
    Symbol *R (inst()->argsymbol(op.firstarg()+0));
    Symbol *A (inst()->argsymbol(op.firstarg()+1));

    if (! A->is_constant() || R->typespec().is_closure_based())
        return false;   // we don't handle those cases

    // turn 'R_float = A_int_const' into a float const assignment
    if (A->typespec().is_int() && R->typespec().is_float()) {
        float result = *(int *)A->data();
        int cind = add_constant (R->typespec(), &result);
        turn_into_assign (op, cind, "coerce to correct type");
        return true;
    }

    // turn 'R_int = A_float_const' into an int const assignment
    if (A->typespec().is_float() && R->typespec().is_int()) {
        int result = (int) *(float *)A->data();
        int cind = add_constant (R->typespec(), &result);
        turn_into_assign (op, cind, "coerce to correct type");
        return true;
    }

    // turn 'R_triple = A_int_const' into a float const assignment
    if (A->typespec().is_int() && R->typespec().is_triple()) {
        float f = *(int *)A->data();
        Vec3 result (f, f, f);
        int cind = add_constant (R->typespec(), &result);
        turn_into_assign (op, cind, "coerce to correct type");
        return true;
    }

    // turn 'R_triple = A_float_const' into a triple const assignment
    if (A->typespec().is_float() && R->typespec().is_triple()) {
        float f = *(float *)A->data();
        Vec3 result (f, f, f);
        int cind = add_constant (R->typespec(), &result);
        turn_into_assign (op, cind, "coerce to correct type");
        return true;
    }

    // Turn 'R_triple = A_other_triple_constant' into a triple const assign
    if (A->typespec().is_triple() && R->typespec().is_triple() &&
        A->typespec() != R->typespec()) {
        Vec3 *f = (Vec3 *)A->data();
        int cind = add_constant (R->typespec(), f);
        turn_into_assign (op, cind, "coerce to correct type");
        return true;
    }

    // turn 'R_matrix = A_float_const' into a matrix const assignment
    if (A->typespec().is_float() && R->typespec().is_matrix()) {
        float f = *(float *)A->data();
        Matrix44 result (f, 0, 0, 0, 0, f, 0, 0, 0, 0, f, 0, 0, 0, 0, f);
        int cind = add_constant (R->typespec(), &result);
        turn_into_assign (op, cind, "coerce to correct type");
        return true;
    }
    // turn 'R_matrix = A_int_const' into a matrix const assignment
    if (A->typespec().is_int() && R->typespec().is_matrix()) {
        float f = *(int *)A->data();
        Matrix44 result (f, 0, 0, 0, 0, f, 0, 0, 0, 0, f, 0, 0, 0, 0, f);
        int cind = add_constant (R->typespec(), &result);
        turn_into_assign (op, cind, "coerce to correct type");
        return true;
    }

    return false;
}



void
RuntimeOptimizer::clear_stale_syms ()
{
    m_stale_syms.clear ();
}



void
RuntimeOptimizer::use_stale_sym (int sym)
{
    FastIntMap::iterator i = m_stale_syms.find(sym);
    if (i != m_stale_syms.end())
        m_stale_syms.erase (i);
}



bool
RuntimeOptimizer::is_simple_assign (Opcode &op)
{
    // Simple only if arg0 is the only write, and is write only.
    if (op.argwrite_bits() != 1 || op.argread(0))
        return false;
    const OpDescriptor *opd = shadingsys().op_descriptor (op.opname());
    if (!opd || !opd->simple_assign)
        return false;   // reject all other known non-simple assignments
    // Make sure the result isn't also read
    int result = oparg(op,0);
    for (int i = 1, e = op.nargs();  i < e;  ++i)
        if (oparg(op,i) == result)
            return false;
    return true;
}



void
RuntimeOptimizer::simple_sym_assign (int sym, int opnum)
{
    if (optimize() >= 2 && m_opt_stale_assign) {
        FastIntMap::iterator i = m_stale_syms.find(sym);
        if (i != m_stale_syms.end()) {
            Opcode &uselessop (inst()->ops()[i->second]);
            if (uselessop.opname() != u_nop)
                turn_into_nop (uselessop,
                           debug() > 1 ? Strutil::format("remove stale value assignment to %s, reassigned on op %d",
                                                         opargsym(uselessop,0)->name(), opnum).c_str() : "");
        }
    }
    m_stale_syms[sym] = opnum;
}



bool
RuntimeOptimizer::unread_after (const Symbol *A, int opnum)
{
    // Try to figure out if this symbol is completely unused after this
    // op (and thus, any values written to it now will never be needed).

    // Globals may be read by later layers
    if (A->symtype() == SymTypeGlobal)
        return false;

    // Params may be read afterwards if connected to a downstream
    // layer or if "elide_unconnected_outputs" is turned off.
    if (A->symtype() == SymTypeOutputParam || A->symtype() == SymTypeParam) {
        if (! m_opt_elide_unconnected_outputs)
            return false;   // Asked not do do this optimization
        if (A->connected_down())
            return false;   // Connected to something downstream
        if (A->renderer_output())
            return false;   // This is a renderer output -- don't cull it
    }

    // For all else, check if it's either never read at all in this
    // layer or it's only read earlier and we're not part of a loop
    return !A->everread() || (A->lastread() <= opnum && !m_in_loop[opnum]);
}



void
RuntimeOptimizer::replace_param_value (Symbol *R, const void *newdata,
                                       const TypeSpec &newdata_type)
{
    ASSERT (R->symtype() == SymTypeParam || R->symtype() == SymTypeOutputParam);
    TypeDesc Rtype = R->typespec().simpletype();
    DASSERT (R->dataoffset() >= 0);
    int Rnvals = int(Rtype.aggregate * Rtype.numelements());
    TypeDesc Ntype = newdata_type.simpletype();
    if (Ntype == TypeDesc::UNKNOWN)
        Ntype = Rtype;
    int Nnvals = int(Ntype.aggregate * Ntype.numelements());
    if (Rtype.basetype == TypeDesc::FLOAT &&
          Ntype.basetype == TypeDesc::FLOAT) {
        float *Rdefault = &inst()->m_fparams[R->dataoffset()];
        DASSERT ((R->dataoffset()+Rnvals) <= (int)inst()->m_fparams.size());
        if (Rnvals == Nnvals)   // straight copy
            for (int i = 0;  i < Rnvals;  ++i)
                Rdefault[i] = ((const float *)newdata)[i];
        else if (Nnvals == 1)  // scalar -> aggregate, by replication
            for (int i = 0;  i < Rnvals;  ++i)
                Rdefault[i] = ((const float *)newdata)[0];
        else {
            ASSERT (0 && "replace_param_value: unexpected types");
        }
    }
    else if (Rtype.basetype == TypeDesc::FLOAT &&
             Ntype.basetype == TypeDesc::INT) {
        // Careful, this is an int-to-float conversion
        float *Rdefault = &inst()->m_fparams[R->dataoffset()];
        DASSERT ((R->dataoffset()+Rnvals) <= (int)inst()->m_fparams.size());
        if (Rnvals == Nnvals)   // straight copy
            for (int i = 0;  i < Rnvals;  ++i)
                Rdefault[i] = ((const int *)newdata)[i];
        else if (Nnvals == 1)  // scalar -> aggregate, by replication
            for (int i = 0;  i < Rnvals;  ++i)
                Rdefault[i] = ((const int *)newdata)[0];
        else {
            ASSERT (0 && "replace_param_value: unexpected types");
        }
    }
    else if (Rtype.basetype == TypeDesc::INT &&
             Ntype.basetype == TypeDesc::INT && Rnvals == Nnvals) {
        int *Rdefault = &inst()->m_iparams[R->dataoffset()];
        DASSERT ((R->dataoffset()+Rnvals) <= (int)inst()->m_iparams.size());
        for (int i = 0;  i < Rnvals;  ++i)
            Rdefault[i] = ((const int *)newdata)[i];
    }
    else if (Rtype.basetype == TypeDesc::STRING &&
             Ntype.basetype == TypeDesc::STRING && Rnvals == Nnvals) {
        ustring *Rdefault = &inst()->m_sparams[R->dataoffset()];
        DASSERT ((R->dataoffset()+Rnvals) <= (int)inst()->m_sparams.size());
        for (int i = 0;  i < Rnvals;  ++i)
            Rdefault[i] = ((const ustring *)newdata)[i];
    } else {
        ASSERT (0 && "replace_param_value: unexpected types");
    }
}



// Predicate to test if the connection's destination is never used
struct ConnectionDestIs
{
    ConnectionDestIs (const ShaderInstance &inst, const Symbol *sym)
        : m_inst(inst), m_sym(sym) { }
    bool operator() (const Connection &c) {
        return m_inst.symbol(c.dst.param) == m_sym;
    }
private:
    const ShaderInstance &m_inst;
    const Symbol *m_sym;
};



/// Symbol R in the current instance has a connection or init ops we
/// no longer need; turn it into a a plain old instance-value
/// parameter.
void
RuntimeOptimizer::make_param_use_instanceval (Symbol *R, string_view why)
{
    if (debug() > 1)
        std::cout << "Turning " << R->valuesourcename() << ' '
                  << R->typespec().c_str() << ' ' << R->name()
                  << " into an instance value "
                  << why << "\n";

    // Mark its source as the instance value, not connected
    R->valuesource (Symbol::InstanceVal);
    // If it isn't a connection or computed, it doesn't need derivs.
    R->has_derivs (false);

    // Point the symbol's data pointer to its instance value
    // uniform
    void *Rdefault = NULL;
    DASSERT (R->dataoffset() >= 0);
    TypeDesc Rtype = R->typespec().simpletype();
    if (Rtype.basetype == TypeDesc::FLOAT)
        Rdefault = &inst()->m_fparams[R->dataoffset()];
    else if (Rtype.basetype == TypeDesc::INT)
        Rdefault = &inst()->m_iparams[R->dataoffset()];
    else if (Rtype.basetype == TypeDesc::STRING)
        Rdefault = &inst()->m_sparams[R->dataoffset()];
    DASSERT (Rdefault != NULL);
    R->data (Rdefault);

    // Get rid of any init ops
    if (R->has_init_ops()) {
        turn_into_nop (R->initbegin(), R->initend(), "init ops not needed");
        R->initbegin (0);
        R->initend (0);
    }
    // Erase R's incoming connections
    erase_if (inst()->connections(), ConnectionDestIs(*inst(),R));
}



/// Check for conditions under which assignments to output parameters
/// can be removed.
///
/// Return true if the assignment is removed entirely.
bool
RuntimeOptimizer::outparam_assign_elision (int opnum, Opcode &op)
{
    ASSERT (op.opname() == u_assign);
    Symbol *R (inst()->argsymbol(op.firstarg()+0));
    Symbol *A (inst()->argsymbol(op.firstarg()+1));

    if (R->symtype() != SymTypeOutputParam)
        return false;    // This logic is only about output params

    // Check for assignment of output params that are written only once
    // in the whole shader -- on this statement -- and assigned a
    // constant, and the assignment is unconditional.  In that case,
    // just alias it to the constant from here on out.
    if (// R is being assigned a constant of the right type:
        A->is_constant() && R->typespec() == A->typespec()
                // FIXME -- can this be equivalent() rather than == ?
        // and it's written only on this op, and unconditionally:
        && R->firstwrite() == opnum && R->lastwrite() == opnum
        && !m_in_conditional[opnum]
        // and this is not a case of an init op for an output param that
        // actually will get an instance value or a connection:
        && ! ((R->valuesource() == Symbol::InstanceVal || R->connected())
              && R->initbegin() <= opnum && R->initend() > opnum)
        ) {
        // Alias it to the constant it's being assigned
        int cind = inst()->args()[op.firstarg()+1];
        global_alias (inst()->args()[op.firstarg()], cind);
        // If it's also never read before this assignment and isn't a
        // designated renderer output (which we obviously must write!), just
        // replace its default value entirely and get rid of the assignment.
        if (R->firstread() > opnum && ! R->renderer_output() &&
                m_opt_elide_unconnected_outputs) {
            make_param_use_instanceval (R, Strutil::format("- written once, with a constant (%s), before any reads", const_value_as_string(*A)));
            replace_param_value (R, A->data(), A->typespec());
            turn_into_nop (op, debug() > 1 ? Strutil::format("oparam %s never subsequently read or connected", R->name().c_str()).c_str() : "");
            return true;
        }
    }

    // If the output param will neither be read later in the shader nor
    // connected to a downstream layer, then we don't really need this
    // assignment at all. Note that unread_after() does take into
    // consideration whether it's a renderer output.
    if (unread_after(R,opnum)) {
        turn_into_nop (op, debug() > 1 ? Strutil::format("oparam %s never subsequently read or connected", R->name().c_str()).c_str() : "");
        return true;
    }

    return false;
}




/// If every potentially-written argument to this op is NEVER read, turn
/// it into a nop and return true.  We don't do this to ops that have no
/// written args at all, since they tend to have side effects (e.g.,
/// printf, setmessage).
bool
RuntimeOptimizer::useless_op_elision (Opcode &op, int opnum)
{
    if (op.nargs()) {
        bool writes_something = false;
        for (int a = 0;  a < op.nargs();  ++a) {
            if (op.argwrite(a)) {
                writes_something = true;
                Symbol *A = opargsym (op, a);
                if (! unread_after(A,opnum))
                    return false;
            }
        }
        // If we get this far, nothing written had any effect
        if (writes_something) {
            // Enumerate exceptions -- ops that write something, but have
            // side effects that means they shouldn't be eliminated.
            if (op.opname() == u_pointcloud_write)
                return false;
            // It's a useless op, eliminate it
            turn_into_nop (op, "eliminated op whose writes will never be read");
            return true;
        }
    }
    return false;
}



int
RuntimeOptimizer::dealias_symbol (int symindex, int opnum)
{
    do {
        int i = block_alias (symindex);
        if (i >= 0) {
            // block-specific alias for the sym
            symindex = i;
            continue;
        }
        FastIntMap::const_iterator found;
        found = m_symbol_aliases.find (symindex);
        if (found != m_symbol_aliases.end()) {
            // permanent alias for the sym
            symindex = found->second;
            continue;
        }
        if (inst()->symbol(symindex)->symtype() == SymTypeParam &&
            opnum >= inst()->maincodebegin()) {
            // Only check parameter aliases for main code
            found = m_param_aliases.find (symindex);
            if (found != m_param_aliases.end()) {
                symindex = found->second;
                continue;
            }
        }
    } while (0);
    return symindex;
}



void
RuntimeOptimizer::block_unalias (int symindex)
{
    FastIntMap::iterator i = m_block_aliases.find (symindex);
    if (i != m_block_aliases.end())
        i->second = -1;
    // In addition to the current block_aliases, unalias from any
    // saved alias lists.
    for (size_t s = 0, send = m_block_aliases_stack.size(); s < send; ++s) {
        FastIntMap::iterator i = m_block_aliases_stack[s]->find (symindex);
        if (i != m_block_aliases_stack[s]->end())
            i->second = -1;
    }
}



/// Make sure there's room for at least one more symbol, so that we can
/// add a const if we need to, without worrying about the addresses of
/// symbols changing if we add a new one soon.
void
RuntimeOptimizer::make_symbol_room (int howmany)
{
    inst()->make_symbol_room (howmany);
}




// Predicate to test if a symbol (specified by symbol index, symbol
// pointer, or by the inbound Connection record) is never used within
// the shader or passed along.  Subtlety: you can't base the test for
// params on sym->everused(), since of course it may be used within its
// own init ops, but then never subsequently used, and thus be a prime
// candidate for culling.  Instead, for params we test whether it was
// used at any point AFTER its init ops.
class SymNeverUsed
{
public:
    SymNeverUsed (const RuntimeOptimizer &rop, const ShaderInstance *inst)
        : m_rop(rop), m_inst(inst)
    { }
    bool operator() (const Symbol &sym) const {
        if (sym.symtype() == SymTypeParam)
            return (sym.lastuse() < sym.initend()) && !sym.connected_down();
        if (sym.symtype() == SymTypeOutputParam) {
            if (! m_rop.opt_elide_unconnected_outputs())
                return false;   // Asked not to do this optimization
            if (sym.connected_down())
                return false;   // Connected to something downstream
            if (sym.renderer_output())
                return false;   // This is a renderer output
            return (sym.lastuse() < sym.initend());
        }
        return ! sym.everused();  // all other symbol types
    }
    bool operator() (int symid) const {
        return (*this)(*m_inst->symbol(symid));
    }
    bool operator() (const Connection &c) const {
        return (*this)(c.dst.param);
    }
private:
    const RuntimeOptimizer &m_rop;
    const ShaderInstance *m_inst;
};



int
RuntimeOptimizer::next_block_instruction (int opnum)
{
    int end = (int)inst()->ops().size();
    for (int n = opnum+1; n < end && m_bblockids[n] == m_bblockids[opnum]; ++n)
        if (inst()->ops()[n].opname() != u_nop)
            return n;   // Found it!
    return 0;   // End of ops or end of basic block
}



int
RuntimeOptimizer::peephole2 (int opnum, int op2num)
{
    Opcode &op (inst()->ops()[opnum]);
    Opcode &next (inst()->ops()[op2num]);

    // N.B. Some of these transformations may look strange, you may
    // think "nobody will write code that does that", but (a) they do;
    // and (b) it can end up like that after other optimizations have
    // changed the code around.

    // Ping-pong assignments can eliminate the second one:
    //     assign a b
    //     assign b a    <-- turn into nop
    // But note that if a is an int and b is a float, this transformation
    // is not safe because of the intentional truncation.
    if (op.opname() == u_assign && next.opname() == u_assign) {
        Symbol *a = opargsym(op,0);
        Symbol *b = opargsym(op,1);
        Symbol *c = opargsym(next,0);
        Symbol *d = opargsym(next,1);
        if (a == d && b == c) {
            // Exclude the integer truncation case
            if (! (a->typespec().is_int() && b->typespec().is_floatbased())) {
                // std::cerr << "ping-pong assignment " << opnum << " of " 
                //           << opargsym(op,0)->mangled() << " and "
                //           << opargsym(op,1)->mangled() << "\n";
                turn_into_nop (next, "ping-pong assignments");
                return 1;
            }
        }
    }

    // Daisy chain assignments -> use common source
    //     assign a b
    //     assign c a
    // turns into:
    //     assign a b
    //     assign c b
    // This may allow a to be eliminated if it's not used elsewhere.
    // But note that this doesn't work for float = int = float,
    // which intentionally truncates before the assignment to c!
    if (op.opname() == u_assign && next.opname() == u_assign) {
        Symbol *a = opargsym(op,0);
        Symbol *b = opargsym(op,1);
        Symbol *c = opargsym(next,0);
        Symbol *d = opargsym(next,1);
        if (a == d && assignable (c->typespec(), b->typespec())) {
            // Exclude the float=int=float case
            if (! (a->typespec().is_int() && b->typespec().is_floatbased() &&
                   c->typespec().is_floatbased())) {
                turn_into_assign (next, inst()->arg(op.firstarg()+1),
                                  "daisy-chain assignments");
                return 1;
            }
        }
    }

    // Look for adjacent add and subtract of the same value:
    //     add a a b
    //     sub a a b
    // (or vice versa)
    if (((op.opname() == u_add && next.opname() == u_sub) ||
         (op.opname() == u_sub && next.opname() == u_add)) &&
          opargsym(op,0) == opargsym(next,0) &&
          opargsym(op,1) == opargsym(next,1) &&
          opargsym(op,2) == opargsym(next,2) &&
          opargsym(op,0) == opargsym(op,1)) {
        // std::cerr << "dueling add/sub " << opnum << " & " << op2num << ": " 
        //           << opargsym(op,0)->mangled() << "\n";
        turn_into_nop (op, "simplify add/sub pair");
        turn_into_nop (next, "simplify add/sub pair");
        return 2;
    }

    // Look for add of a value then subtract of the same value
    //     add a b c     or:    sub a b c
    //     sub d a c            add d a c
    // the second instruction should be changed to
    //     assign d b
    // and furthermore, if the only use of a is on these two lines or
    // if a == d, then the first instruction can be changed to a 'nop'.
    // Careful, "only used on these two lines" can be tricky if 'a' is a
    // global or output parameter, which are used after the shader finishes!
    if (((op.opname() == u_add && next.opname() == u_sub) ||
         (op.opname() == u_sub && next.opname() == u_add)) &&
        opargsym(op,0) == opargsym(next,1) &&
        opargsym(op,2) == opargsym(next,2) &&
        opargsym(op,0) != opargsym(next,2) /* a != c */) {
        Symbol *a = opargsym(op,0);
        Symbol *d = opargsym(next,0);
        turn_into_assign (next, oparg(op,1)/*b*/, "simplify add/sub pair");
        if ((a->firstuse() >= opnum && a->lastuse() <= op2num &&
             ((a->symtype() != SymTypeGlobal && a->symtype() != SymTypeOutputParam)))
            || a == d) {
            turn_into_nop (op, "simplify add/sub pair");
            return 2;
        }
        else
            return 1;
    }

    // Look for simple functions followed by an assignment:
    //    OP a b...
    //    assign c a
    // If OP is "simple" (completely overwrites its first argument, only
    // reads the rest), and a and c are the same type, and a is never
    // used again, then we can replace those two instructions with:
    //    OP c b...
    // Careful, "never used again" can be tricky if 'a' is a global or
    // output parameter, which are used after the shader finishes!
    if (next.opname() == u_assign && 
        op.nargs() >= 1 && opargsym(op,0) == opargsym(next,1) &&
        is_simple_assign(op)) {
        Symbol *a = opargsym(op,0);
        Symbol *c = opargsym(next,0);
        if (a->firstuse() >= opnum && a->lastuse() <= op2num &&
              (a->symtype() != SymTypeGlobal && a->symtype() != SymTypeOutputParam) &&
              equivalent (a->typespec(), c->typespec())) {
            if (debug() > 1)
                debug_opt ("turned '%s %s...' to '%s %s...' as part of daisy-chain\n",
                           op.opname(), a->name(), op.opname(), c->name());
            inst()->args()[op.firstarg()] = inst()->args()[next.firstarg()];
            c->mark_rw (opnum, false, true);
            // Any time we write to a variable that wasn't written to at
            // this op previously, we need to block_unalias it, or it
            // can dealias to the wrong thing when examining subsequent
            // instructions.
            block_unalias (oparg(op,0));  // clear any aliases
            turn_into_nop (next, "daisy-chain op and assignment");
            return 2;
        }
    }

    // Convert this combination
    //     closure A name arg...
    //     mul B A weight
    // into
    //     closure B C name arg...
    // That is, collapse a creation and immediate scale of a closure into
    // a single closure-with-scale constructor. (Valid if A is not used
    // elsewhere.)  Further refinement: if weight = 1, no need to do
    // the scale, and if weight == 0, eliminate the work entirely.
    // We only do this optimization on pass > 1, to give a fair chance
    // for other optimizations to be able to turn the weight into a
    // constant before we do this one (since if it's 1 or 0, we can
    // simplify further).
    if (op.opname() == u_closure && next.opname() == u_mul
          && optimization_pass() > 1) {
        Symbol *a = opargsym(op,0);
        Symbol *name = opargsym(op,1);
        Symbol *aa = opargsym(next,1);
        Symbol *weight = opargsym(next,2);
        int weightarg = 2;
        if (weight->typespec().is_closure()) {  // opposite order
            std::swap (aa, weight);
            weightarg = 1;
        }
        if (name->typespec().is_string() &&
            a->firstuse() >= opnum && a->lastuse() <= op2num &&
            a == aa && weight->typespec().is_triple()) {
            if (is_zero(*weight)) {
                turn_into_nop (op, "zero-weighted closure");
                turn_into_assign (next, add_constant(0.0f),
                                  "zero-weighted closure");
                return 1;
            }
            // FIXME - handle weight being a float as well
            std::vector<int> newargs;
            newargs.push_back (oparg(next,0)); // B
            if (! is_one(*weight))
                newargs.push_back (oparg(next,weightarg)); // weight
            for (int i = 1;  i < op.nargs();  ++i)
                newargs.push_back (oparg(op,i));
            turn_into_nop (op, "combine closure+mul");
            turn_into_nop (next, "combine closure+mul");
            insert_code (opnum, u_closure, newargs,
                         RecomputeRWRanges, GroupWithNext);
            if (debug() > 1)
                std::cout << "op " << opnum << "-" << (op2num) 
                          << " combined closure+mul\n";            
            return 1;
        }
    }

    // No changes
    return 0;
}



/// Mark our params that feed to later layers, and whether we have any
/// outgoing connections.
void
RuntimeOptimizer::mark_outgoing_connections ()
{
    ASSERT (! inst()->m_instoverrides.size() &&
            "don't call this before copy_code_from_master");
    inst()->outgoing_connections (false);
    FOREACH_PARAM (Symbol &s, inst())
        s.connected_down (false);
    for (int lay = layer()+1;  lay < group().nlayers();  ++lay) {
        BOOST_FOREACH (Connection &c, group()[lay]->m_connections)
            if (c.srclayer == layer()) {
                inst()->symbol(c.src.param)->connected_down (true);
                inst()->outgoing_connections (true);
            }
    }
}



/// Check all params and output params to find any that are neither used
/// in the shader (aside from their own init ops, which shouldn't count)
/// nor connected to downstream layers, and for those, remove their init
/// ops and connections.
/// Precondition: mark_outgoing_connections should be up to date.
int
RuntimeOptimizer::remove_unused_params ()
{
    int alterations = 0;
    SymNeverUsed param_never_used (*this, inst());  // handy predicate

    // Get rid of unused params' init ops and clear their read/write ranges
    FOREACH_PARAM (Symbol &s, inst()) {
        if (param_never_used(s) && s.has_init_ops()) {
            std::string why;
            if (debug() > 1)
                why = Strutil::format ("remove init ops of unused param %s %s", s.typespec().c_str(), s.name());
            turn_into_nop (s.initbegin(), s.initend(), why);
            s.set_initrange (0, 0);
            s.clear_rw();   // mark as totally unused
            ++alterations;
        }
    }

    // Get rid of the Connections themselves
    erase_if (inst()->connections(), param_never_used);

    return alterations;
}



void
RuntimeOptimizer::catalog_symbol_writes (int opbegin, int opend,
                                         FastIntSet &syms)
{
    for (int i = opbegin; i < opend; ++i) {
        const Opcode &op (inst()->ops()[i]);
        for (int a = 0, nargs = op.nargs();  a < nargs;  ++a) {
            if (op.argwrite(a))
                syms.insert (oparg (op, a));
        }
    }
}



/// Find situations where an output is simply a copy of a connected
/// input, and eliminate the middleman.
int
RuntimeOptimizer::eliminate_middleman ()
{
    int changed = 0;
    FOREACH_PARAM (Symbol &s, inst()) {
        // Skip if this isn't a shader output parameter that's connected
        // to a later layer.
        if (s.symtype() != SymTypeOutputParam || !s.connected_down())
            continue;
        // If it's written more than once, or has init ops, don't bother
        if (s.firstwrite() != s.lastwrite() || s.has_init_ops())
            continue;
        // Ok, s is a connected output, written only once, without init ops.

        // If the one time it's written isn't a simple assignment, never mind
        int opnum = s.firstwrite();
        Opcode &op (inst()->ops()[opnum]);
        if (op.opname() != u_assign)
            continue;   // only consider direct assignments
        // Now what's it assigned from?  If it's not a connected
        // parameter, or if it's not an equivalent data type, or if it's
        // a closure, never mind.
        int src_index = oparg (op, 1);
        Symbol *src = opargsym (op, 1);

        if (! (src->symtype() == SymTypeParam && src->connected()) ||
              ! equivalent(src->typespec(), s.typespec()) ||
              s.typespec().is_closure())
            continue;

        // Only works if the assignment is unconditional.  Needs to not
        // be in a conditional or loop, and not have any exit or return
        // statement before the assignment.
        if (! op_is_unconditionally_executed (opnum))
            continue;

        // OK, output param 's' is simply and unconditionally assigned
        // the value of the equivalently-typed input parameter 'src'.
        // Doctor downstream shaders that use s to connect directly to
        // src.

        // First, find what src is connected to.
        int upstream_layer = -1, upstream_symbol = -1;
        for (int i = 0, e = inst()->nconnections();  i < e;  ++i) {
            const Connection &c = inst()->connection(i);
            if (c.dst.param == src_index &&  // the connection we want
                c.src.is_complete() && c.dst.is_complete() &&
                equivalent(c.src.type,c.dst.type) &&
                !c.src.type.is_closure() && ! c.dst.type.is_closure()) {
                upstream_layer = c.srclayer;
                upstream_symbol = c.src.param;
                break;
            }
        }
        if (upstream_layer < 0 || upstream_symbol < 0)
            continue;  // not a complete connection, forget it
            
        ShaderInstance *upinst = group()[upstream_layer];
        if (debug() > 1)
            std::cout << "Noticing that " << inst()->layername() << "." 
                      << s.name() << " merely copied from " << src->name() 
                      << ", connected from " << upinst->layername() << "."
                      << upinst->symbol(upstream_symbol)->name() << "\n";

        // Find all the downstream connections of s, make them 
        // connections to src.
        int s_index = inst()->symbolindex(&s);
        for (int laynum = layer()+1;  laynum < group().nlayers();  ++laynum) {
            ShaderInstance *downinst = group()[laynum];
            for (int i = 0, e = downinst->nconnections();  i < e;  ++i) {
                Connection &c = downinst->connections()[i];
                if (c.srclayer == layer() && // connected to our layer
                    c.src.param == s_index && // connected to s
                    c.src.is_complete() && c.dst.is_complete() &&
                    equivalent(c.src.type,c.dst.type)) {
                    // just change the connection's referrant to the
                    // upstream source of s.
                    c.srclayer = upstream_layer;
                    c.src.param = upstream_symbol;
                    ++changed;
                    shadingsys().m_stat_middlemen_eliminated += 1;
                    if (debug() > 1) {
                        const Symbol *dsym = downinst->symbol(c.dst.param);
                        if (! dsym)
                            dsym = downinst->mastersymbol(c.dst.param);
                        const Symbol *usym = upinst->symbol(upstream_symbol);
                        if (! usym)
                            usym = upinst->mastersymbol(upstream_symbol);
                        ASSERT (dsym && usym);
                        std::cout << "Removed " << inst()->layername() << "."
                                  << s.name() << " middleman for " 
                                  << downinst->layername() << "."
                                  << dsym->name() << ", now connected to "
                                  << upinst->layername() << "."
                                  << usym->name() << "\n";
                    }
                }
            }
        }
    }
    return changed;
}



int
RuntimeOptimizer::optimize_assignment (Opcode &op, int opnum)
{
    // Various optimizations specific to assignment statements
    ASSERT (op.opname() == u_assign);
    int changed = 0;
    Symbol *R (inst()->argsymbol(op.firstarg()+0));
    Symbol *A (inst()->argsymbol(op.firstarg()+1));
    bool R_local_or_tmp = (R->symtype() == SymTypeLocal ||
                           R->symtype() == SymTypeTemp);
    if (block_alias(inst()->arg(op.firstarg())) == inst()->arg(op.firstarg()+1) ||
        block_alias(inst()->arg(op.firstarg()+1)) == inst()->arg(op.firstarg())) {
        // We're re-assigning something already aliased, skip it
        turn_into_nop (op, "reassignment of current value (2)");
        return ++changed;
    }
    if (coerce_assigned_constant (op)) {
        // A may have changed, so we need to reset it
        A = inst()->argsymbol(op.firstarg()+1);
        ++changed;
    }
    // NOW do assignment constant folding, only after we
    // have performed all the other transformations that may
    // turn this op into an assignment.
    changed += constfold_assign (*this, opnum);
    if (op.opname() != u_assign) {
        // The const fold has changed the assignment to something
        // other than assign (presumably nop), so skip the other
        // assignment transformations below.
        return 0;
    }
    if ((A->is_constant() || A->lastwrite() < opnum) &&
        equivalent(R->typespec(), A->typespec())) {
        // Safe to alias R to A for this block, if A is a
        // constant or if it's never written to again.
        block_alias (inst()->arg(op.firstarg()),
                         inst()->arg(op.firstarg()+1));
        // std::cerr << opnum << " aliasing " << R->mangled() << " to "
        //       << inst()->argsymbol(op.firstarg()+1)->mangled() << "\n";
    }
    if (A->is_constant() && R->typespec() == A->typespec() &&
        R_local_or_tmp &&
        R->firstwrite() == opnum && R->lastwrite() == opnum) {
        // This local or temp is written only once in the
        // whole shader -- on this statement -- and it's
        // assigned a constant.  So just alias it to the
        // constant.
        int cind = inst()->args()[op.firstarg()+1];
        global_alias (inst()->args()[op.firstarg()], cind);
        turn_into_nop (op, "replace symbol with constant");
        return ++changed;
    }
    if (R_local_or_tmp && ! R->everread()) {
        // This local is written but NEVER READ.  nop it.
        turn_into_nop (op, "local/tmp never read");
        return ++changed;
    }
    if (outparam_assign_elision (opnum, op)) {
        return ++changed;
    }
    if (R == A) {
        // Just an assignment to itself -- turn into NOP!
        turn_into_nop (op, "self-assignment");
        return ++changed;
    } else if (R_local_or_tmp && R->lastread() < opnum
               && ! m_in_loop[opnum]) {
        // Don't bother assigning if we never read it again
        turn_into_nop (op, "symbol never read again");
        return ++changed;
    }
    return changed;
}



void
RuntimeOptimizer::copy_block_aliases (const FastIntMap &old_block_aliases,
                                      FastIntMap &new_block_aliases,
                                      const FastIntSet *excluded,
                                      bool copy_temps)
{
    ASSERT (&old_block_aliases != &new_block_aliases &&
            "copy_block_aliases does not work in-place");
    // Find all symbols written anywhere in the instruction range
    new_block_aliases.clear ();
    new_block_aliases.reserve (old_block_aliases.size());
    for (FastIntMap::const_iterator alias = old_block_aliases.begin();
         alias != old_block_aliases.end();  ++alias) {
        if (alias->second < 0)
            continue;    // erased alias -- don't copy
        if (! copy_temps && (inst()->symbol(alias->first)->is_temp() ||
                             inst()->symbol(alias->second)->is_temp()))
            continue;    // don't copy temp aliases unless told to
        if (excluded && (excluded->find(alias->first) != excluded->end() ||
                         excluded->find(alias->second) != excluded->end()))
            continue;    // don't copy from excluded list
        new_block_aliases[alias->first] = alias->second;
    }
}



int
RuntimeOptimizer::optimize_ops (int beginop, int endop,
                                FastIntMap *seed_block_aliases)
{
    if (beginop >= endop)
        return 0;

    // Constant aliases valid for just this basic block
    clear_block_aliases ();

    // Provide a place where, if we recurse, we can save prior block
    // aliases. Register them on the block_aliases_stack so that calls to
    // block_unalias() will unalias from there, too.
    FastIntMap saved_block_aliases;
    m_block_aliases_stack.push_back (&saved_block_aliases);

    int lastblock = -1;
    int skipops = 0;   // extra inserted ops to skip over
    int changed = 0;
    size_t num_ops = inst()->ops().size();
    size_t old_num_ops = num_ops;   // track when it changes
    for (int opnum = beginop;  opnum < endop;  opnum += 1) {
        ASSERT (old_num_ops == num_ops); // better not happen unknowingly
        DASSERT (num_ops == inst()->ops().size());
        if (m_stop_optimizing)
            break;
        // Before getting a reference to this op, be sure that a space
        // is reserved at the end in case a folding routine inserts an
        // op.  That ensures that the reference won't be invalid.
        inst()->ops().reserve (num_ops+1);
        Opcode &op (inst()->ops()[opnum]);
        if (skipops) {
            // If a previous optimization inserted ops and told us
            // to skip over the new ones, we still need to unalias
            // any symbols written by this op, but otherwise skip
            // all subsequent optimizations until we run down the
            // skipops counter.
            block_unalias_written_args (op);
            ASSERT (lastblock == m_bblockids[opnum] &&
                    "this should not be a new basic block");
            --skipops;
            continue;   // Move along to the next op, no opimization here
        }
        // Things to do if we've just moved to a new basic block
        if (lastblock != m_bblockids[opnum]) {
            clear_block_aliases (seed_block_aliases);
            seed_block_aliases = NULL; // only the first time
            clear_stale_syms ();
            lastblock = m_bblockids[opnum];
        }
        // Nothing below here to do for no-ops, take early out.
        if (op.opname() == u_nop)
            continue;
        // De-alias the readable args to the op and figure out if
        // there are any constants involved.
        for (int i = 0, e = op.nargs();  i < e;  ++i) {
            if (! op.argwrite(i)) { // Don't de-alias args that are written
                int argindex = op.firstarg() + i;
                int argsymindex = dealias_symbol (inst()->arg(argindex), opnum);
                inst()->args()[argindex] = argsymindex;
            }
            if (op.argread(i))
                use_stale_sym (oparg(op,i));
        }
        // If it's a simple assignment and the lvalue is "stale", go
        // back and eliminate its last assignment.
        if (is_simple_assign(op))
            simple_sym_assign (oparg (op, 0), opnum);
        // Make sure there's room for several more symbols, so that we
        // can add a few consts if we need to, without worrying about
        // the addresses of symbols changing when we add a new one below.
        make_symbol_room (max_new_consts_per_fold);
        // For various ops that we know how to effectively
        // constant-fold, dispatch to the appropriate routine.
        if (optimize() >= 2 && m_opt_constant_fold) {
            const OpDescriptor *opd = shadingsys().op_descriptor (op.opname());
            if (opd && opd->folder) {
                int c = (*opd->folder) (*this, opnum);
                if (c) {
                    changed += c;
                    // Re-check num_ops in case the folder inserted something
                    num_ops = inst()->ops().size();
                    skipops = num_ops - old_num_ops;
                    endop += num_ops - old_num_ops; // adjust how far we loop
                    old_num_ops = num_ops;
                }
            }
        }
        // Clear local block aliases for any args that were written
        // by this op
        block_unalias_written_args (op);

        // Now we handle assignments.
        if (optimize() >= 2 && op.opname() == u_assign && m_opt_assign)
            changed += optimize_assignment (op, opnum);
        if (optimize() >= 2 && m_opt_elide_useless_ops)
            changed += useless_op_elision (op, opnum);
        if (m_stop_optimizing)
            break;
        // Peephole optimization involving pair of instructions (the second
        // instruction will be in the same basic block.
        if (optimize() >= 2 && m_opt_peephole && op.opname() != u_nop) {
            // Find the next instruction in the same basic block
            int op2num = next_block_instruction (opnum);
            if (op2num) {
                int c = peephole2 (opnum, op2num);
                if (c) {
                    changed += c;
                    // Re-check num_ops in case the folder inserted something
                    num_ops = inst()->ops().size();
                    // skipops = num_ops - old_num_ops;
                    endop += num_ops - old_num_ops; // adjust how far we loop
                    old_num_ops = num_ops;
                }
            }
        }

        // Special cases for "if", "functioncall", and loops: Optimize the
        // sequences of instructions in the bodies recursively in a way that
        // allows us to be clever about the basic block alias tracking.
        ustring opname = op.opname();
        if ((opname == u_if || opname == u_functioncall ||
             opname == u_for || opname == u_while || opname == u_dowhile)
              && shadingsys().m_opt_seed_bblock_aliases) {
            // Find all symbols written anywhere in the instruction range
            // of the bodies.
            FastIntSet symwrites;
            catalog_symbol_writes (opnum+1, op.farthest_jump(), symwrites);
            // Save the aliases from the basic block we are exiting.
            // If & function call: save all prior aliases.
            // Loops: dont save aliases involving syms written in the loop.
            // Note that for both cases, we don't copy aliases involving
            // temps, because that breaks our later assumptions (for temp
            // coalescing) that temp uses never cross basic block boundaries.
            if (opname == u_if || opname == u_functioncall)
                copy_block_aliases (m_block_aliases, saved_block_aliases);
            else
                copy_block_aliases (m_block_aliases, saved_block_aliases,
                                    &symwrites);
            // 'if' has 2 blocks (then, else), function call has just
            // one (the body), loops have 4 (init, cond, body, incr),
            int njumps = (opname == u_if) ? 2 : (opname == u_functioncall ? 1 : 4);
            // Recursively optimize each body block.
            // Don't use op after inserstions! Use inst()->op(opnum).
            for (int j = 0; j < njumps; ++j)
                changed += optimize_ops (j==0 ? opnum+1 : inst()->op(opnum).jump(j-1),
                                         inst()->op(opnum).jump(j),
                                         &saved_block_aliases);
            // Adjust optimization loop end if any instructions were added
            num_ops = inst()->ops().size();
            endop += num_ops - old_num_ops;
            old_num_ops = num_ops;
            // Now we can restore the original aliases to seed the basic
            // block that follows. For if/function, we need to remove all
            // aliases referencing syms written within the conditional or
            // function body. For loops, recall that we already excluded
            // the written syms from the saved_block_aliases.
            if (opname == u_if || opname == u_functioncall) {
                FastIntMap restored_aliases;
                restored_aliases.swap (saved_block_aliases);
                // catalog again, in case optimizations in those blocks
                // caused writes that weren't apparent before.
                catalog_symbol_writes (opnum+1, inst()->op(opnum).farthest_jump(), symwrites);
                copy_block_aliases (restored_aliases, saved_block_aliases,
                                    &symwrites);
            }
            seed_block_aliases = &saved_block_aliases;
            // Get ready to increment to the next instruction
            opnum = inst()->op(opnum).farthest_jump() - 1;
        }
    }
    m_block_aliases_stack.pop_back();  // Done with saved_block_aliases
    return changed;
}



void
RuntimeOptimizer::optimize_instance ()
{
    // If "opt_layername" attribute is set, only optimize the named layer
    if (shadingsys().m_opt_layername &&
        shadingsys().m_opt_layername != inst()->layername())
        return;

    // Make a list of the indices of all constants.
    for (int i = 0, e = (int)inst()->symbols().size();  i < e;  ++i)
        if (inst()->symbol(i)->symtype() == SymTypeConst)
            m_all_consts.push_back (i);

    // Turn all parameters with instance or default values, and which
    // cannot be overridden by geometry values, into constants or
    // aliases for globals.  Also turn connections from earlier layers'
    // outputs that are known to be constants or globals into constants
    // or global aliases without any connection.
    if (optimize() >= 2 && m_opt_simplify_param) {
        simplify_params ();
    }

#ifndef NDEBUG
    // Confirm that the symbols between [firstparam,lastparam] are all
    // input or output params.
    FOREACH_PARAM (const Symbol &s, inst()) {
        ASSERT (s.symtype() == SymTypeParam ||
                s.symtype() == SymTypeOutputParam);
    }
#endif

    // Recompute which of our params have downstream connections.
    mark_outgoing_connections ();

    // Try to fold constants.  We take several passes, until we get to
    // the point that not much is improving.  It rarely goes beyond 3-4
    // passes, but we have a hard cutoff just to be sure we don't
    // ever get into an infinite loop from an unforseen cycle where we
    // end up inadvertently transforming A => B => A => etc.
    int totalchanged = 0;
    int reallydone = 0;   // Force a few passes after we think we're done
    int npasses = shadingsys().opt_passes();
    for (m_pass = 0;  m_pass < npasses;  ++m_pass) {

        // Once we've made one pass (and therefore called
        // mark_outgoing_connections), we may notice that the layer is
        // unused, and therefore can stop doing work to optimize it.
        if (m_pass != 0 && inst()->unused())
            break;

        if (m_stop_optimizing)
            break;

        if (debug() > 1)
            debug_opt ("layer %d \"%s\", pass %d:\n",
                       layer(), inst()->layername(), m_pass);

        // Track basic blocks and conditional states
        find_conditionals ();
        find_basic_blocks ();

        // Clear local messages for this instance
        m_local_unknown_message_sent = false;
        m_local_messages_sent.clear ();

        // Figure out which params are just aliases for globals (only
        // necessary to do once, on the first pass).
        if (m_pass == 0 && optimize() >= 2)
            find_params_holding_globals ();

        // Here is the meat of the optimization, where we pass over the
        // code for this instance and make various transformations.
        int changed = optimize_ops (0, (int)inst()->ops().size());

        // Now that we've rewritten the code, we need to re-track the
        // variable lifetimes.
        track_variable_lifetimes ();

        // Recompute which of our params have downstream connections.
        mark_outgoing_connections ();

        // Find situations where an output is simply a copy of a connected
        // input, and eliminate the middleman.
        if (optimize() >= 2 && m_opt_middleman) {
            int c = eliminate_middleman ();
            if (c)
                mark_outgoing_connections ();
            changed += c;
        }

        // Elide unconnected parameters that are never read.
        if (optimize() >= 1)
            changed += remove_unused_params ();

        // FIXME -- we should re-evaluate whether writes_globals() is still
        // true for this layer.

        // If nothing changed, we're done optimizing.  But wait, it may be
        // that after re-tracking variable lifetimes, we can notice new
        // optimizations!  So force another pass, then we're really done.
        totalchanged += changed;
        if (changed < 1) {
            if (++reallydone > 3)
                break;
        } else {
            reallydone = 0;
        }
    }

    // A layer that was allowed to run lazily originally, if it no
    // longer (post-optimized) has any outgoing connections, is no
    // longer needed at all.
    if (inst()->unused()) {
        // Not needed.  Remove all its connections and ops.
        inst()->connections().clear ();
        turn_into_nop (0, (int)inst()->ops().size()-1,
                       debug() > 1 ? Strutil::format("eliminate layer %s with no outward connections", inst()->layername().c_str()).c_str() : "");
        BOOST_FOREACH (Symbol &s, inst()->symbols())
            s.clear_rw ();
    }

    // Now that we've optimized this layer, walk through the ops and
    // note which messages may have been sent, so subsequent layers will
    // know.
    for (int opnum = 0, e = (int)inst()->ops().size();  opnum < e;   ++opnum) {
        Opcode &op (inst()->ops()[opnum]);
        if (op.opname() == u_setmessage) {
            Symbol &Name (*inst()->argsymbol(op.firstarg()+0));
            if (Name.is_constant())
                m_messages_sent.push_back (*(ustring *)Name.data());
            else
                m_unknown_message_sent = true;
        }
    }
}



void
RuntimeOptimizer::resolve_isconnected ()
{
    for (int i = 0, n = (int)inst()->ops().size();  i < n;  ++i) {
        Opcode &op (inst()->ops()[i]);
        if (op.opname() == u_isconnected) {
            inst()->make_symbol_room (1);
            SymbolPtr s = inst()->argsymbol (op.firstarg() + 1);
            while (const StructSpec *structspec = s->typespec().structspec()) {
                // How to deal with structures -- just change the reference
                // to the first field in the struct.
                // FIXME -- if we ever allow separate layer connection of
                // individual struct members, this will need something more
                // sophisticated.
                ASSERT (structspec && structspec->numfields() >= 1);
                std::string fieldname = (s->name().string() + "." +
                                         structspec->field(0).name.string());
                int fieldsymid = inst()->findparam (ustring(fieldname));
                ASSERT (fieldsymid >= 0);
                s = inst()->symbol(fieldsymid);
            }
            int val = (s->connected() ? 1 : 0) + (s->connected_down() ? 2 : 0);
            turn_into_assign (op, add_constant(TypeDesc::TypeInt, &val),
                              "resolve isconnected()");
        }
    }
}



void
RuntimeOptimizer::track_variable_lifetimes (const SymbolPtrVec &allsymptrs)
{
    SymbolPtrVec oparg_ptrs;
    oparg_ptrs.reserve (inst()->args().size());
    BOOST_FOREACH (int a, inst()->args())
        oparg_ptrs.push_back (inst()->symbol (a));

    OSLCompilerImpl::track_variable_lifetimes (inst()->ops(), oparg_ptrs,
                                               allsymptrs);
}



void
RuntimeOptimizer::track_variable_lifetimes ()
{
    SymbolPtrVec allsymptrs;
    allsymptrs.reserve (inst()->symbols().size());
    BOOST_FOREACH (Symbol &s, inst()->symbols())
        allsymptrs.push_back (&s);

    track_variable_lifetimes (allsymptrs);
}


// This has O(n^2) memory usage, so only for debugging
//#define DEBUG_SYMBOL_DEPENDENCIES

// Add to the dependency map that "symbol A depends on symbol B".
void
RuntimeOptimizer::add_dependency (SymDependency &dmap, int A, int B)
{
    ASSERT (A < (int)inst()->symbols().size());
    ASSERT (B < (int)inst()->symbols().size());
    dmap[A].insert (B);

#ifdef DEBUG_SYMBOL_DEPENDENCIES
    // Unification -- make all of B's dependencies be dependencies of A.
    BOOST_FOREACH (int r, dmap[B])
        dmap[A].insert (r);
#endif
}


void
RuntimeOptimizer::syms_used_in_op (Opcode &op, std::vector<int> &rsyms,
                                   std::vector<int> &wsyms)
{
    rsyms.clear ();
    wsyms.clear ();
    for (int i = 0;  i < op.nargs();  ++i) {
        int arg = inst()->arg (i + op.firstarg());
        if (op.argread(i))
            if (std::find (rsyms.begin(), rsyms.end(), arg) == rsyms.end())
                rsyms.push_back (arg);
        if (op.argwrite(i))
            if (std::find (wsyms.begin(), wsyms.end(), arg) == wsyms.end())
                wsyms.push_back (arg);
    }
}



// Fake symbol index for "derivatives" entry in dependency map.
static const int DerivSym = -1;


// Recursively mark symbols that have derivatives from dependency map
void
RuntimeOptimizer::mark_symbol_derivatives (SymDependency &symdeps, SymIntSet &visited, int d)
{
    BOOST_FOREACH (int r, symdeps[d]) {
        if (visited.find(r) == visited.end()) {
            visited.insert(r);
            
            Symbol *s = inst()->symbol(r);

            if (! s->typespec().is_closure_based() && 
                    s->typespec().elementtype().is_floatbased())
                s->has_derivs (true);

            mark_symbol_derivatives(symdeps, visited, r);
        }
    }
}


/// Run through all the ops, for each one marking its 'written'
/// arguments as dependent upon its 'read' arguments (and performing
/// unification as we go), yielding a dependency map that lets us look
/// up any symbol and see the set of other symbols on which it ever
/// depends on during execution of the shader.
void
RuntimeOptimizer::track_variable_dependencies ()
{
    SymDependency symdeps;

    // It's important to note that this is simplistically conservative
    // in that it overestimates dependencies.  To see why this is the
    // case, consider the following code:
    //       // inputs a,b; outputs x,y; local variable t
    //       t = a;
    //       x = t;
    //       t = b;
    //       y = t;
    // We can see that x depends on a and y depends on b.  But the
    // dependency analysis we do below thinks that y also depends on a
    // (because t depended on both a and b, but at different times).
    //
    // This naivite will never miss a dependency, but it may
    // overestimate dependencies.  (Hence we call this "conservative"
    // rather than "wrong.")  We deem this acceptable for now, since
    // it's so much easer to implement the conservative dependency
    // analysis, and it's not yet clear that getting it closer to
    // optimal will have any performance impact on final shaders. Also
    // because this is probably no worse than the "dependency slop" that
    // would happen with loops and conditionals.  But we certainly may
    // revisit with a more sophisticated algorithm if this crops up
    // a legitimate issue.
    //
    // Because of this conservative approach, it is critical that this
    // analysis is done BEFORE temporaries are coalesced (which would
    // cause them to be reassigned in exactly the way that confuses this
    // analysis).

    symdeps.clear ();

    std::vector<int> read, written;
    // Loop over all ops...
    BOOST_FOREACH (Opcode &op, inst()->ops()) {
        // Gather the list of syms read and written by the op.  Reuse the
        // vectors defined outside the loop to cut down on malloc/free.
        read.clear ();
        written.clear ();
        syms_used_in_op (op, read, written);

        // FIXME -- special cases here!  like if any ops implicitly read
        // or write to globals without them needing to be arguments.

        // For each symbol w written by the op...
        BOOST_FOREACH (int w, written) {
            // For each symbol r read by the op, make w depend on r.
            // (Unless r is a constant , in which case it's not necessary.)
            BOOST_FOREACH (int r, read)
                if (inst()->symbol(r)->symtype() != SymTypeConst)
                    add_dependency (symdeps, w, r);
            // If the op takes derivs, make the pseudo-symbol DerivSym
            // depend on those arguments.
            if (op.argtakesderivs_all()) {
                for (int a = 0;  a < op.nargs();  ++a)
                    if (op.argtakesderivs(a)) {
                        Symbol &s (*opargsym (op, a));
                        // Constants can't take derivs
                        if (s.symtype() == SymTypeConst)
                            continue;
                        // Careful -- not all globals can take derivs
                        if (s.symtype() == SymTypeGlobal &&
                            ! (s.mangled() == Strings::P ||
                               s.mangled() == Strings::I ||
                               s.mangled() == Strings::u ||
                               s.mangled() == Strings::v ||
                               s.mangled() == Strings::Ps))
                            continue;
                        add_dependency (symdeps, DerivSym,
                                        inst()->arg(a+op.firstarg()));
                    }
            }
        }
    }

    // Propagate derivative dependencies for any syms already known to
    // need derivs.  It's probably marked that way because another layer
    // downstream connects to it and needs derivatives of that
    // connection.
    int snum = 0;
    BOOST_FOREACH (Symbol &s, inst()->symbols()) {
        // Globals that get written should always provide derivs.
        // Exclude N, since its derivs are unreliable anyway, so no point
        // making it cause the whole disp shader to need derivs.
        if (s.symtype() == SymTypeGlobal && s.everwritten() &&
              !s.typespec().is_closure_based() && s.mangled() != Strings::N)
            s.has_derivs(true);
        if (s.has_derivs())
            add_dependency (symdeps, DerivSym, snum);
        ++snum;
    }

    // Mark all symbols needing derivatives as such
    SymIntSet visited;
    mark_symbol_derivatives (symdeps, visited, DerivSym);

    // Only some globals are allowed to have derivatives
    BOOST_FOREACH (Symbol &s, inst()->symbols()) {
        if (s.symtype() == SymTypeGlobal &&
            ! (s.mangled() == Strings::P ||
               s.mangled() == Strings::I ||
               s.mangled() == Strings::u ||
               s.mangled() == Strings::v ||
               s.mangled() == Strings::Ps))
            s.has_derivs (false);
    }

#ifdef DEBUG_SYMBOL_DEPENDENCIES
    // Helpful for debugging

    std::cerr << "track_variable_dependencies\n";
    std::cerr << "\nDependencies:\n";
    BOOST_FOREACH (SymDependency::value_type &m, symdeps) {
        if (m.first == DerivSym)
            std::cerr << "$derivs depends on ";
        else
            std::cerr << inst->symbol(m.first)->mangled() << " depends on ";
        BOOST_FOREACH (int d, m.second) {
            if (d == DerivSym)
                std::cerr << "$derivs ";
            else
                std::cerr << inst->symbol(d)->mangled() << ' ';
        }
        std::cerr << "\n";
    }
    std::cerr << "\n\n";

    // Invert the dependency
    SymDependency influences;
    BOOST_FOREACH (SymDependency::value_type &m, symdeps)
        BOOST_FOREACH (int d, m.second)
            influences[d].insert (m.first);

    std::cerr << "\nReverse dependencies:\n";
    BOOST_FOREACH (SymDependency::value_type &m, influences) {
        if (m.first == DerivSym)
            std::cerr << "$derivs contrbutes to ";
        else
            std::cerr << inst->symbol(m.first)->mangled() << " contributes to ";
        BOOST_FOREACH (int d, m.second) {
            if (d == DerivSym)
                std::cerr << "$derivs ";
            else
                std::cerr << inst->symbol(d)->mangled() << ' ';
        }
        std::cerr << "\n";
    }
    std::cerr << "\n\n";
#endif
}



// Is the symbol coalescable?
inline bool
coalescable (const Symbol &s)
{
    return (s.symtype() == SymTypeTemp &&     // only coalesce temporaries
            s.everused() &&                   // only if they're used
            s.dealias() == &s &&              // only if not already aliased
            ! s.typespec().is_structure() &&  // only if not a struct
            s.fieldid() < 0);                 //    or a struct field
}



/// Coalesce temporaries.  During code generation, we make a new
/// temporary EVERY time we need one.  Now we examine them all and merge
/// ones of identical type and non-overlapping lifetimes.
void
RuntimeOptimizer::coalesce_temporaries ()
{
    // We keep looping until we can't coalesce any more.
    int ncoalesced = 1;
    while (ncoalesced) {
        ncoalesced = 0;   // assume we're done, unless we coalesce something

        // We use a greedy algorithm that loops over each symbol, and
        // then examines all higher-numbered symbols (in order) and
        // tries to merge the first one it can find that doesn't overlap
        // lifetimes.  The temps were created as we generated code, so
        // they are already sorted by their "first use".  Thus, for any
        // pair t1 and t2 that are merged, it is guaranteed that t2 is
        // the symbol whose first use the earliest of all symbols whose
        // lifetimes do not overlap t1.

        SymbolVec::iterator s;
        for (s = inst()->symbols().begin(); s != inst()->symbols().end(); ++s) {
            // Skip syms that can't be (or don't need to be) coalesced
            if (! coalescable(*s))
                continue;

            int sfirst = s->firstuse ();
            int slast  = s->lastuse ();

            // Loop through every other symbol
            for (SymbolVec::iterator t = s+1; t != inst()->symbols().end(); ++t) {
                // Coalesce s and t if both syms are coalescable,
                // equivalent types, have nonoverlapping lifetimes,
                // and either both do or both do not need derivatives.
                if (coalescable (*t) &&
                      equivalent (s->typespec(), t->typespec()) &&
                      s->has_derivs() == t->has_derivs() &&
                      (slast < t->firstuse() || sfirst > t->lastuse())) {
                    // Make all future t references alias to s
                    t->alias (&(*s));
                    // s gets union of the lifetimes
                    s->union_rw (t->firstread(), t->lastread(),
                                 t->firstwrite(), t->lastwrite());
                    sfirst = s->firstuse ();
                    slast  = s->lastuse ();
                    // t gets marked as unused
                    t->clear_rw ();
                    ++ncoalesced;
                }
            }
        }
        // std::cerr << "Coalesced " << ncoalesced << "\n";
    }

    // Since we may have aliased temps, now we need to make sure all
    // symbol refs are dealiased.
    BOOST_FOREACH (int &arg, inst()->args()) {
        Symbol *s = inst()->symbol(arg);
        s = s->dealias ();
        arg = s - inst()->symbol(0);
    }
}



void
RuntimeOptimizer::post_optimize_instance ()
{
    inst()->evaluate_writes_globals_and_userdata_params ();

    if (inst()->unused())
        return;    // skip the expensive stuff if we're not used anyway

    SymbolPtrVec allsymptrs;
    allsymptrs.reserve (inst()->symbols().size());
    BOOST_FOREACH (Symbol &s, inst()->symbols())
        allsymptrs.push_back (&s);

    m_bblockids.clear ();       // Keep insert_code from getting confused
    m_in_conditional.clear ();
    m_in_loop.clear ();

    add_useparam (allsymptrs);

    if (optimize() >= 1 && m_opt_coalesce_temps)
        coalesce_temporaries ();
}



void
RuntimeOptimizer::collapse_syms ()
{
    //
    // Make a new symbol table that removes all the unused symbols.
    //

    // Mark our params that feed to later layers, so that unused params
    // that aren't needed downstream can be removed.
    mark_outgoing_connections ();

    SymbolVec new_symbols;          // buffer for new symbol table
    std::vector<int> symbol_remap;  // mapping of old sym index to new
    int total_syms = 0;             // number of new symbols we'll need
    SymNeverUsed never_used (*this, inst());  // handy predicate

    // First, just count how many we need and set up the mapping
    BOOST_FOREACH (const Symbol &s, inst()->symbols()) {
        symbol_remap.push_back (total_syms);
        if (! never_used (s))
            ++total_syms;
    }

    // Now make a new table of the right (new) size, and copy the used syms
    new_symbols.reserve (total_syms);
    BOOST_FOREACH (const Symbol &s, inst()->symbols()) {
        if (! never_used (s))
            new_symbols.push_back (s);
    }

    // Remap all the function arguments to the new indices
    BOOST_FOREACH (int &arg, inst()->m_instargs)
        arg = symbol_remap[arg];

    // Fix our connections from upstream shaders
    BOOST_FOREACH (Connection &c, inst()->m_connections)
        c.dst.param = symbol_remap[c.dst.param];

    // Fix downstream connections that reference us
    for (int lay = layer()+1;  lay < group().nlayers();  ++lay) {
        BOOST_FOREACH (Connection &c, group()[lay]->m_connections)
            if (c.srclayer == layer())
                c.src.param = symbol_remap[c.src.param];
    }

    // Swap the new symbol list for the old.
    std::swap (inst()->m_instsymbols, new_symbols);
    {
        // adjust memory stats
        // Remember that they're already swapped
        off_t mem = vectorbytes(new_symbols) - vectorbytes(inst()->m_instsymbols);
        ShadingSystemImpl &ss (shadingsys());
        spin_lock lock (ss.m_stat_mutex);
        ss.m_stat_mem_inst_syms -= mem;
        ss.m_stat_mem_inst -= mem;
        ss.m_stat_memory -= mem;
    }

    // Miscellaneous cleanup of other things that used symbol indices
    inst()->m_Psym = -1;
    inst()->m_Nsym = -1;
    inst()->m_firstparam = -1;
    inst()->m_lastparam = -1;
    int i = 0;
    BOOST_FOREACH (Symbol &s, inst()->symbols()) {
        if (s.symtype() == SymTypeParam || s.symtype() == SymTypeOutputParam) {
            if (inst()->m_firstparam < 0)
                inst()->m_firstparam = i;
            inst()->m_lastparam = i+1;
        }
        if (s.name() == Strings::P)
            inst()->m_Psym = i;
        else if (s.name() == Strings::N)
            inst()->m_Nsym = i;
        ++i;
    }
#ifndef NDEBUG
    // Confirm that the symbols between [firstparam,lastparam] are all
    // input or output params.
    FOREACH_PARAM (const Symbol &s, inst()) {
        ASSERT (s.symtype() == SymTypeParam ||
                s.symtype() == SymTypeOutputParam);
    }
#endif
}



void
RuntimeOptimizer::collapse_ops ()
{
    //
    // Make new code that removes all the nops
    //
    OpcodeVec new_ops;              // buffer for new code
    std::vector<int> op_remap;      // mapping of old opcode indices to new
    int total_ops = 0;              // number of new ops we'll need

    // First, just count how many we need and set up the mapping
    BOOST_FOREACH (const Opcode &op, inst()->ops()) {
        op_remap.push_back (total_ops);
        if (op.opname() != u_nop)
            ++total_ops;
    }

    // Now make a new table of the right (new) size, copy the used ops, and
    // reset the jump addresses.
    new_ops.reserve (total_ops);
    BOOST_FOREACH (const Opcode &op, inst()->ops()) {
        if (op.opname() != u_nop) {
            new_ops.push_back (op);
            Opcode &newop (new_ops.back());
            for (int i = 0;  i < (int)Opcode::max_jumps;  ++i)
                if (newop.jump(i) >= 0)
                    newop.jump(i) = op_remap[newop.jump(i)];
        }
    }

    // Adjust 'main' code range and init op ranges
    inst()->m_maincodebegin = op_remap[inst()->m_maincodebegin];
    inst()->m_maincodeend = (int)new_ops.size();
    FOREACH_PARAM (Symbol &s, inst()) {
        if (s.has_init_ops()) {
            s.initbegin (op_remap[s.initbegin()]);
            if (s.initend() < (int)op_remap.size())
                s.initend (op_remap[s.initend()]);
            else
                s.initend ((int)new_ops.size());
        }
    }

    // Swap the new code for the old.
    std::swap (inst()->m_instops, new_ops);

    // These are no longer valid
    m_bblockids.clear ();
    m_in_conditional.clear ();
    m_in_loop.clear ();
}



std::ostream &
RuntimeOptimizer::printinst (std::ostream &out) const
{
    out << "Shader " << inst()->shadername() << "\n";
    out << (inst()->unused() ? " UNUSED" : "");
    out << " connections in=" << inst()->nconnections();
    out << " out=" << inst()->outgoing_connections();
    out << (inst()->writes_globals() ? " writes_globals" : "");
    out << (inst()->userdata_params() ? " userdata_params" : "");
    out << (inst()->run_lazily() ? " run_lazily" : " run_unconditionally");
    out << (inst()->outgoing_connections() ? " outgoing_connections" : "");
    out << (inst()->renderer_outputs() ? " renderer_outputs" : "");
    out << (inst()->writes_globals() ? " writes_globals" : "");
    out << (inst()->entry_layer() ? " entry_layer" : "");
    out << (inst()->last_layer() ? " last_layer" : "");
    out << "\n";
    out << "  symbols:\n";
    for (size_t i = 0, e = inst()->symbols().size();  i < e;  ++i)
        inst()->symbol(i)->print (out, 256);
#if 0
    out << "  int consts:\n    ";
    for (size_t i = 0;  i < inst()->m_iconsts.size();  ++i)
        out << inst()->m_iconsts[i] << ' ';
    out << "\n";
    out << "  float consts:\n    ";
    for (size_t i = 0;  i < inst()->m_fconsts.size();  ++i)
        out << inst()->m_fconsts[i] << ' ';
    out << "\n";
    out << "  string consts:\n    ";
    for (size_t i = 0;  i < inst()->m_sconsts.size();  ++i)
        out << "\"" << Strutil::escape_chars(inst()->m_sconsts[i]) << "\" ";
    out << "\n";
#endif
    out << "  code:\n";
    for (size_t i = 0, e = inst()->ops().size();  i < e;  ++i) {
        const Opcode &op (inst()->ops()[i]);
        if (i == (size_t)inst()->maincodebegin())
            out << "(main)\n";
        out << "    " << i << ": " << op.opname();
        bool allconst = true;
        for (int a = 0;  a < op.nargs();  ++a) {
            const Symbol *s (inst()->argsymbol(op.firstarg()+a));
            out << " " << s->name();
            if (s->symtype() == SymTypeConst) {
                out << " (";
                s->print_vals(out,16);
                out << ")";
            }
            if (op.argread(a))
                allconst &= s->is_constant();
        }
        for (size_t j = 0;  j < Opcode::max_jumps;  ++j)
            if (op.jump(j) >= 0)
                out << " " << op.jump(j);
        out << "\t# ";
//        out << "    rw " << Strutil::format("%x",op.argread_bits())
//            << ' ' << op.argwrite_bits();
        if (op.argtakesderivs_all())
            out << " %derivs(" << op.argtakesderivs_all() << ") ";
        if (allconst)
            out << "  CONST";
        if (i == 0 || bblockid(i) != bblockid(i-1))
            out << "  BBLOCK-START";
        std::string filename = op.sourcefile().string();
        size_t slash = filename.find_last_of ("/");
        if (slash != std::string::npos)
            filename.erase (0, slash+1);
        if (filename.length())
            out << "  (" << filename << ":" << op.sourceline() << ")";
        out << "\n";
    }
    if (inst()->nconnections()) {
        out << "  connections upstream:\n";
        for (int i = 0, e = inst()->nconnections(); i < e; ++i) {
            const Connection &c (inst()->connection(i));
            out << "    " << c.dst.type.c_str() << ' '
                << inst()->symbol(c.dst.param)->name();
            if (c.dst.arrayindex >= 0)
                out << '[' << c.dst.arrayindex << ']';
            out << " upconnected from layer " << c.srclayer << ' ';
            const ShaderInstance *up = group()[c.srclayer];
            out << "(" << up->layername() << ") ";
            out << "    " << c.src.type.c_str() << ' '
                << up->symbol(c.src.param)->name();
            if (c.src.arrayindex >= 0)
                out << '[' << c.src.arrayindex << ']';
            out << "\n";
        }
    }
    return out;
}



void
RuntimeOptimizer::run ()
{
    Timer rop_timer;
    int nlayers = (int) group().nlayers ();
    if (debug())
        shadingcontext()->info ("About to optimize shader group %s (%d layers):",
                           group().name(), nlayers);
    if (debug())
        std::cout << "About to optimize shader group " << group().name() << "\n";

    for (int layer = 0;  layer < nlayers;  ++layer) {
        set_inst (layer);
        // These need to happen before merge_instances
        inst()->copy_code_from_master (group());
        mark_outgoing_connections();
    }

    // Inventory the network and print pre-optimized debug info
    size_t old_nsyms = 0, old_nops = 0;
    for (int layer = 0;  layer < nlayers;  ++layer) {
        set_inst (layer);
        if (debug() /* && optimize() >= 1*/) {
            find_basic_blocks ();
            std::cout.flush ();
            std::cout << "Before optimizing layer " << layer << " \"" 
                      << inst()->layername() << "\" (ID " << inst()->id() << ") :\n";
            printinst (std::cout);
            std::cout << "\n--------------------------------\n" << std::endl;
        }
        old_nsyms += inst()->symbols().size();
        old_nops += inst()->ops().size();
    }

    if (shadingsys().m_opt_merge_instances == 1)
        shadingsys().merge_instances (group());

    m_params_holding_globals.resize (nlayers);

    // Optimize each layer, from first to last
    for (int layer = 0;  layer < nlayers;  ++layer) {
        set_inst (layer);
        if (inst()->unused())
            continue;
        // N.B. we need to resolve isconnected() calls before the instance
        // is otherwise optimized, or else isconnected() may not reflect
        // the original connectivity after substitutions are made.
        resolve_isconnected ();
        optimize_instance ();
    }

    // Optimize each layer again, from last to first (because some
    // optimizations are only apparent when the subsequent shaders have
    // been simplified).
    for (int layer = nlayers-2;  layer >= 0;  --layer) {
        set_inst (layer);
        if (! inst()->unused())
            optimize_instance ();
    }

    // Try merging instances again, now that we've optimized
    shadingsys().merge_instances (group(), true);

    for (int layer = nlayers-1;  layer >= 0;  --layer) {
        set_inst (layer);
        if (inst()->unused())
            continue;
        track_variable_dependencies ();

        // For our parameters that require derivatives, mark their
        // upstream connections as also needing derivatives.
        BOOST_FOREACH (Connection &c, inst()->m_connections) {
            if (inst()->symbol(c.dst.param)->has_derivs()) {
                Symbol *source = group()[c.srclayer]->symbol(c.src.param);
                if (! source->typespec().is_closure_based() &&
                    source->typespec().elementtype().is_floatbased()) {
                    source->has_derivs (true);
                }
            }
        }
    }

    // Post-opt cleanup: add useparam, coalesce temporaries, etc.
    for (int layer = 0;  layer < nlayers;  ++layer) {
        set_inst (layer);
        post_optimize_instance ();
    }

    // Last chance to eliminate duplicate instances
    shadingsys().merge_instances (group(), true);

    // Get rid of nop instructions and unused symbols.
    size_t new_nsyms = 0, new_nops = 0;
    for (int layer = 0;  layer < nlayers;  ++layer) {
        set_inst (layer);
        if (inst()->unused())
            continue;  // no need to print or gather stats for unused layers
        if (optimize() >= 1) {
            collapse_syms ();
            collapse_ops ();
        }
        if (debug() && !inst()->unused()) {
            track_variable_lifetimes ();
            find_basic_blocks ();
            std::cout << "After optimizing layer " << layer << " \"" 
                      << inst()->layername() << "\" (ID " << inst()->id() << ") :\n";
            printinst (std::cout);
            std::cout << "\n--------------------------------\n" << std::endl;
        }
        new_nsyms += inst()->symbols().size();
        new_nops += inst()->ops().size();
    }

    m_unknown_textures_needed = false;
    m_unknown_closures_needed = false;
    m_unknown_attributes_needed = false;
    m_textures_needed.clear();
    m_closures_needed.clear();
    m_globals_needed.clear();
    m_userdata_needed.clear();
    m_attributes_needed.clear();
    bool does_nothing = true;
    for (int layer = 0;  layer < nlayers;  ++layer) {
        set_inst (layer);
        if (inst()->unused())
            continue;  // no need to print or gather stats for unused layers
        FOREACH_SYM (Symbol &s, inst()) {
            // set the layer numbers
            s.layer (layer);
            // Find interpolated parameters
            if ((s.symtype() == SymTypeParam || s.symtype() == SymTypeOutputParam)
                && ! s.lockgeom()) {
                UserDataNeeded udn (s.name(), s.typespec().simpletype(), s.has_derivs());
                std::set<UserDataNeeded>::iterator found;
                found = m_userdata_needed.find (udn);
                if (found == m_userdata_needed.end())
                    m_userdata_needed.insert (udn);
                else if (udn.derivs && ! found->derivs) {
                    m_userdata_needed.erase (found);
                    m_userdata_needed.insert (udn);
                }
            }
            // Track which globals the group needs
            if (s.symtype() == SymTypeGlobal) {
                m_globals_needed.insert (s.name());
            }
        }
        BOOST_FOREACH (const Opcode &op, inst()->ops()) {
            const OpDescriptor *opd = shadingsys().op_descriptor (op.opname());
            if (! opd)
                continue;
            if (op.opname() != Strings::end && op.opname() != Strings::useparam)
                does_nothing = false;  // a non-unused layer with a nontrivial op
            if (opd->flags & OpDescriptor::Tex) {
                // for all the texture ops, arg 1 is the texture name
                Symbol *sym = opargsym (op, 1);
                ASSERT (sym && sym->typespec().is_string());
                if (sym->is_constant()) {
                    ustring texname = *(ustring *)sym->data();
                    m_textures_needed.insert (texname);
                } else {
                    m_unknown_textures_needed = true;
                }
            }
            if (op.opname() == u_closure) {
                // It's either 'closure result weight name' or 'closure result name'
                Symbol *sym = opargsym (op, 1); // arg 1 is the closure name
                if (sym && !sym->typespec().is_string())
                    sym = opargsym (op, 2);
                ASSERT (sym && sym->typespec().is_string());
                if (sym->is_constant()) {
                    ustring closurename = *(ustring *)sym->data();
                    m_closures_needed.insert (closurename);
                } else {
                    m_unknown_closures_needed = true;
                }
            } else if (op.opname() == u_getattribute) {
                Symbol *sym1 = opargsym (op, 1);
                ASSERT (sym1 && sym1->typespec().is_string());
                if (sym1->is_constant()) {
                    if (op.nargs() == 3) {
                        // getattribute( attributename, result )
                        m_attributes_needed.insert( AttributeNeeded( *(ustring *)sym1->data() ) );
                    } else {
                        ASSERT (op.nargs() == 4 || op.nargs() == 5);
                        Symbol *sym2 = opargsym (op, 2);
                        if (sym2->typespec().is_string()) {
                            // getattribute( scopename, attributename, result ) or
                            // getattribute( scopename, attributename, arrayindex, result )
                            if (sym2->is_constant()) {
                                m_attributes_needed.insert( AttributeNeeded(
                                    *(ustring *)sym2->data(), *(ustring *)sym1->data()
                                ) );
                            } else {
                                m_unknown_attributes_needed = true;
                            }
                        } else {
                            // getattribute( attributename, arrayindex, result )
                            m_attributes_needed.insert( AttributeNeeded( *(ustring *)sym1->data() ) );
                        }
                    }
                } else { // sym1 not constant
                    m_unknown_attributes_needed = true;
                }
            }
        }
    }
    group().does_nothing (does_nothing);

    m_stat_specialization_time = rop_timer();
    {
        // adjust memory stats
        ShadingSystemImpl &ss (shadingsys());
        spin_lock lock (ss.m_stat_mutex);
        ss.m_stat_preopt_syms += old_nsyms;
        ss.m_stat_preopt_ops += old_nops;
        ss.m_stat_postopt_syms += new_nsyms;
        ss.m_stat_postopt_ops += new_nops;
        if (does_nothing)
            ss.m_stat_empty_groups += 1;
    }
    if (shadingsys().m_compile_report) {
        shadingcontext()->info ("Optimized shader group %s:", group().name());
        shadingcontext()->info (" spec %1.2fs, New syms %llu/%llu (%5.1f%%), ops %llu/%llu (%5.1f%%)",
              m_stat_specialization_time, new_nsyms, old_nsyms,
              100.0*double((long long)new_nsyms-(long long)old_nsyms)/double(old_nsyms),
              new_nops, old_nops,
              100.0*double((long long)new_nops-(long long)old_nops)/double(old_nops));
        if (does_nothing)
            shadingcontext()->info ("Group does nothing");
        if (m_textures_needed.size()) {
            shadingcontext()->info ("Group needs textures:");
            BOOST_FOREACH (ustring f, m_textures_needed)
                shadingcontext()->info ("    %s", f);
            if (m_unknown_textures_needed)
                shadingcontext()->info ("    Also may construct texture names on the fly.");
        }
        if (m_userdata_needed.size()) {
            shadingcontext()->info ("Group potentially needs userdata:");
            BOOST_FOREACH (UserDataNeeded f, m_userdata_needed)
                shadingcontext()->info ("    %s %s %s", f.name, f.type,
                                        f.derivs ? "(derivs)" : "");
        }
        if (m_attributes_needed.size()) {
            shadingcontext()->info ("Group needs attributes:");
            BOOST_FOREACH (const AttributeNeeded &f, m_attributes_needed)
                shadingcontext()->info ("    %s %s", f.name, f.scope);
            if (m_unknown_attributes_needed)
                shadingcontext()->info ("    Also may construct attribute names on the fly.");
        }
    }
}


}; // namespace pvt
OSL_NAMESPACE_EXIT
