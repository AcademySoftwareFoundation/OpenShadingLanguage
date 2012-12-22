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
               u_break ("break"),
               u_continue ("continue"),
               u_return ("return"),
               u_useparam ("useparam"),
               u_pointcloud_write ("pointcloud_write"),
               u_isconnected ("isconnected"),
               u_setmessage ("setmessage"),
               u_getmessage ("getmessage");


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



RuntimeOptimizer::RuntimeOptimizer (ShadingSystemImpl &shadingsys,
                                    ShaderGroup &group, ShadingContext *ctx)
    : m_shadingsys(shadingsys),
      m_thread(shadingsys.get_perthread_info()),
      m_group(group),
      m_inst(NULL), m_context(ctx),
      m_pass(0),
      m_debug(shadingsys.debug()),
      m_optimize(shadingsys.optimize()),
      m_opt_constant_param(shadingsys.m_opt_constant_param),
      m_opt_constant_fold(shadingsys.m_opt_constant_fold),
      m_opt_stale_assign(shadingsys.m_opt_stale_assign),
      m_opt_elide_useless_ops(shadingsys.m_opt_elide_useless_ops),
      m_opt_elide_unconnected_outputs(shadingsys.m_opt_elide_unconnected_outputs),
      m_opt_peephole(shadingsys.m_opt_peephole),
      m_opt_coalesce_temps(shadingsys.m_opt_coalesce_temps),
      m_opt_assign(shadingsys.m_opt_assign),
      m_opt_mix(shadingsys.m_opt_mix),
      m_next_newconst(0), m_next_newtemp(0),
      m_stat_opt_locking_time(0), m_stat_specialization_time(0),
      m_stat_total_llvm_time(0), m_stat_llvm_setup_time(0),
      m_stat_llvm_irgen_time(0), m_stat_llvm_opt_time(0),
      m_stat_llvm_jit_time(0),
      m_llvm_context(NULL), m_llvm_module(NULL),
      m_llvm_exec(NULL), m_builder(NULL),
      m_llvm_passes(NULL), m_llvm_func_passes(NULL)
{
    set_debug ();
    memset (&m_shaderglobals, 0, sizeof(ShaderGlobals));
    m_shaderglobals.context = m_context;
}



RuntimeOptimizer::~RuntimeOptimizer ()
{
    delete m_builder;
    delete m_llvm_passes;
    delete m_llvm_func_passes;
}



void
RuntimeOptimizer::set_inst (int newlayer)
{
    m_layer = newlayer;
    m_inst = m_group[m_layer];
    ASSERT (m_inst != NULL);
    set_debug ();
    m_all_consts.clear ();
    m_symbol_aliases.clear ();
    m_block_aliases.clear ();
    m_param_aliases.clear ();
}



void
RuntimeOptimizer::set_debug ()
{
    // start with the shading system's idea of debugging level
    m_debug = shadingsys().debug();

    if (shadingsys().m_debug_groupname &&
        shadingsys().m_debug_groupname != m_group.name()) {
        m_debug = 0;
        if (shadingsys().m_optimize_nondebug) {
            // Debugging trick: if user said to only debug one group, turn
            // on full optimization for all others!  This prevents
            // everything from running 10x slower just because you want to
            // debug one shader.
            m_optimize = 3;
            m_opt_constant_param = true;
            m_opt_constant_fold = true;
            m_opt_stale_assign = true;
            m_opt_elide_useless_ops = true;
            m_opt_elide_unconnected_outputs = true;
            m_opt_peephole = true;
            m_opt_coalesce_temps = true;
            m_opt_assign = true;
            m_opt_mix = true;
        }
    }
    // if user said to only debug one layer, turn off debug if not it
    if (inst() && shadingsys().m_debug_layername &&
        shadingsys().m_debug_layername != inst()->layername()) {
        m_debug = 0;
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
        Symbol newconst (ustring::format ("$newconst%d", m_next_newconst++),
                         type, SymTypeConst);
        void *newdata;
        TypeDesc t (type.simpletype());
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
        ASSERT (inst()->symbols().capacity() > inst()->symbols().size() &&
                "we shouldn't have to realloc here");
        ind = (int) inst()->symbols().size ();
        inst()->symbols().push_back (newconst);
        m_all_consts.push_back (ind);
    }
    return ind;
}



int
RuntimeOptimizer::add_temp (const TypeSpec &type)
{
    Symbol newtemp (ustring::format ("$opttemp%d", m_next_newtemp++),
                    type, SymTypeTemp);
    ASSERT (inst()->symbols().capacity() > inst()->symbols().size() &&
            "we shouldn't have to realloc here");
    inst()->symbols().push_back (newtemp);
    return (int) inst()->symbols().size()-1;
}



void
RuntimeOptimizer::turn_into_new_op (Opcode &op, ustring newop, int newarg1,
                                    int newarg2, const char *why)
{
    int opnum = &op - &(inst()->ops()[0]);
    DASSERT (opnum >= 0 && opnum < (int)inst()->ops().size());
    if (debug() > 1)
        std::cout << "turned op " << opnum
                  << " from " << op.opname() << " to "
                  << newop << ' ' << inst()->symbol(newarg1)->name() << ' '
                  << (newarg2<0 ? "" : inst()->symbol(newarg2)->name().c_str())
                  << (why ? " : " : "") << (why ? why : "") << "\n";
    op.reset (newop, newarg2<0 ? 2 : 3);
    op.argwriteonly (0);
    inst()->args()[op.firstarg()+1] = newarg1;
    op.argread (1, true);
    op.argwrite (1, false);
    opargsym(op, 1)->mark_rw (opnum, true, false);
    if (newarg2 >= 0) {
        inst()->args()[op.firstarg()+2] = newarg2;
        op.argread (2, true);
        op.argwrite (2, false);
        opargsym(op, 2)->mark_rw (opnum, true, false);
    }
}



void
RuntimeOptimizer::turn_into_assign (Opcode &op, int newarg, const char *why)
{
    int opnum = &op - &(inst()->ops()[0]);
    if (debug() > 1)
        std::cout << "turned op " << opnum
                  << " from " << op.opname() << " to "
                  << opargsym(op,0)->name() << " = " 
                  << inst()->symbol(newarg)->name()
                  << (why ? " : " : "") << (why ? why : "") << "\n";
    op.reset (u_assign, 2);
    inst()->args()[op.firstarg()+1] = newarg;
    op.argwriteonly (0);
    op.argread (1, true);
    op.argwrite (1, false);
    // Need to make sure the symbol we're assigning is marked as read
    // for this op.  Unfortunately, mark_rw takes the op number, we just
    // have the pointer, so we subtract to get it.
    DASSERT (opnum >= 0 && opnum < (int)inst()->ops().size());
    Symbol *arg = opargsym (op, 1);
    arg->mark_rw (opnum, true, false);
}



// Turn the current op into a simple assignment to zero (of the first arg).
void
RuntimeOptimizer::turn_into_assign_zero (Opcode &op, const char *why)
{
    static float zero[16] = { 0, 0, 0, 0,  0, 0, 0, 0,
                              0, 0, 0, 0,  0, 0, 0, 0 };
    Symbol &R (*(inst()->argsymbol(op.firstarg()+0)));
    int cind = add_constant (R.typespec(), &zero);
    turn_into_assign (op, cind, why);
}



// Turn the current op into a simple assignment to one (of the first arg).
void
RuntimeOptimizer::turn_into_assign_one (Opcode &op, const char *why)
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
RuntimeOptimizer::turn_into_nop (Opcode &op, const char *why)
{
    if (op.opname() != u_nop) {
        if (debug() > 1)
            std::cout << "turned op " << (&op - &(inst()->ops()[0]))
                      << " from " << op.opname() << " to nop"
                      << (why ? " : " : "") << (why ? why : "") << "\n";
        op.reset (u_nop, 0);
        return 1;
    }
    return 0;
}



int
RuntimeOptimizer::turn_into_nop (int begin, int end, const char *why)
{
    int changed = 0;
    for (int i = begin;  i != end;  ++i) {
        Opcode &op (inst()->ops()[i]);
        if (op.opname() != u_nop) {
            op.reset (u_nop, 0);
            ++changed;
        }
    }
    if (debug() > 1 && changed)
        std::cout << "turned ops " << begin << "-" << (end-1) << " into nop"
                  << (why ? " : " : "") << (why ? why : "") << "\n";
    return changed;
}



void
RuntimeOptimizer::insert_code (int opnum, ustring opname,
                               const int *argsbegin, const int *argsend,
                               bool recompute_rw_ranges, int relation)
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
                               bool recompute_rw_ranges, int relation)
{
    const int *argsbegin = (args_to_add.size())? &args_to_add[0]: NULL;
    const int *argsend = argsbegin + args_to_add.size();

    insert_code (opnum, opname, argsbegin, argsend,
                 recompute_rw_ranges, relation);
}



void
RuntimeOptimizer::insert_code (int opnum, ustring opname, int relation,
                               int arg0, int arg1, int arg2, int arg3)
{
    int args[4];
    int nargs = 0;
    if (arg0 >= 0) args[nargs++] = arg0;
    if (arg1 >= 0) args[nargs++] = arg1;
    if (arg2 >= 0) args[nargs++] = arg2;
    if (arg3 >= 0) args[nargs++] = arg3;
    insert_code (opnum, opname, args, args+nargs, true, relation);
}



/// Insert a 'useparam' instruction in front of instruction 'opnum', to
/// reference the symbols in 'params'.
void
RuntimeOptimizer::insert_useparam (size_t opnum,
                                   const std::vector<int> &params_to_use)
{
    ASSERT (params_to_use.size() > 0);
    OpcodeVec &code (inst()->ops());
    insert_code (opnum, u_useparam, params_to_use, 1);

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
                    if (! m_in_conditional[opnum] &&
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



/// For all the instance's parameters, if they can be found to be
/// effectively constants, make constants for them an alias them to the
/// constant.
void
RuntimeOptimizer::find_constant_params (ShaderGroup &group)
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

        if (s->valuesource() == Symbol::InstanceVal ||
            (s->valuesource() == Symbol::DefaultVal && !s->has_init_ops())) {
            // Instance value or a plain default value (no init ops) --
            // turn it into a constant
            make_symbol_room (1);
            s = inst()->symbol(i);  // In case make_symbol_room changed ptrs
            int cind = add_constant (s->typespec(), s->data());
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
            // the upstream shader is effectively constant, then so is
            // this variable.
            BOOST_FOREACH (Connection &c, inst()->connections()) {
                if (c.dst.param == i) {
                    Symbol *srcsym = group[c.srclayer]->symbol(c.src.param);
                    if (!srcsym->everused() &&
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
                        break;
                    }
                }
            }
        }
    }
}



/// Set up m_in_conditional[] to be true for all ops that are inside of
/// conditionals, false for all unconditionally-executed ops.
void
RuntimeOptimizer::find_conditionals ()
{
    OpcodeVec &code (inst()->ops());

    m_in_conditional.clear ();
    m_in_conditional.resize (code.size(), false);
    m_in_loop.clear ();
    m_in_loop.resize (code.size(), false);
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
    }
}



/// Identify basic blocks by assigning a basic block ID for each
/// instruction.  Within any basic bock, there are no jumps in or out.
/// Also note which instructions are inside conditional states.
/// If do_llvm is true, also construct the m_bb_map that maps opcodes
/// beginning BB's to llvm::BasicBlock records.
void
RuntimeOptimizer::find_basic_blocks (bool do_llvm)
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
    block_begin[inst()->m_maincodebegin] = true;

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
    std::map<int,int>::iterator i = m_stale_syms.find(sym);
    if (i != m_stale_syms.end())
        m_stale_syms.erase (i);
}



bool
RuntimeOptimizer::is_simple_assign (Opcode &op)
{
    // Simple only if arg0 is the only write, and is write only.
    if (op.argwrite_bits() != 1 || op.argread(0))
        return false;
    const OpDescriptor *opd = m_shadingsys.op_descriptor (op.opname());
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
        std::map<int,int>::iterator i = m_stale_syms.find(sym);
        if (i != m_stale_syms.end()) {
            Opcode &uselessop (inst()->ops()[i->second]);
            turn_into_nop (uselessop,
                           debug() > 1 ? Strutil::format("remove stale value assignment to %s, reassigned on op %d", opargsym(uselessop,0)->name().c_str(), opnum).c_str() : "");
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
    if ((A->symtype() == SymTypeOutputParam || A->symtype() == SymTypeParam) &&
        (A->connected_down() || ! m_opt_elide_unconnected_outputs))
        return false;

    // For all else, check if it's either never read at all in this
    // layer or it's only read earlier and we're not part of a loop
    return !A->everread() || (A->lastread() < opnum && !m_in_loop[opnum]);
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
RuntimeOptimizer::make_param_use_instanceval (Symbol *R, const char *why)
{
    if (debug() > 1)
        std::cout << "Turning " << R->valuesourcename() << ' ' 
                  << R->name() << " into an instance value "
                  << (why ? why : "") << "\n";

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

    /// Check for assignment of output params that are written only once
    /// in the whole shader -- on this statement -- and assigned a
    /// constant, and the assignment is unconditional.  In that case,
    /// just alias it to the constant from here on out.
    ///
    /// Furthermore, if nobody READS the output param prior to this
    /// assignment, let's just change its initial value to the constant
    /// and get rid of the assignment altogether!
    if (A->is_constant() && R->typespec() == A->typespec() &&
            R->firstwrite() == opnum && R->lastwrite() == opnum &&
            !m_in_conditional[opnum]) {
        // It's assigned only once, and unconditionally assigned a
        // constant -- alias it
        int cind = inst()->args()[op.firstarg()+1];
        global_alias (inst()->args()[op.firstarg()], cind);

        // If it's also never read before this assignment, just replace its
        // default value entirely and get rid of the assignment.
        if (R->firstread() > opnum) {
            make_param_use_instanceval (R, "- written once, with a constant, before any reads");
            replace_param_value (R, A->data(), A->typespec());
            turn_into_nop (op, debug() > 1 ? Strutil::format("oparam %s never subsequently read or connected", R->name().c_str()).c_str() : "");
            return true;
        }
    }

    // If the output param will neither be read later in the shader nor
    // connected to a downstream layer, then we don't really need this
    // assignment at all.
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
        std::map<int,int>::const_iterator found;
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



/// Make sure there's room for at least one more symbol, so that we can
/// add a const if we need to, without worrying about the addresses of
/// symbols changing if we add a new one soon.  We need an extra
/// entry for block_aliases, too.
void
RuntimeOptimizer::make_symbol_room (int howmany)
{
    inst()->make_symbol_room (howmany);
    m_block_aliases.resize (inst()->symbols().size()+howmany, -1);
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
        if (sym.symtype() == SymTypeOutputParam)
            return m_rop.opt_elide_unconnected_outputs() &&
                   (sym.lastuse() < sym.initend()) && !sym.connected_down();
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
RuntimeOptimizer::peephole2 (int opnum)
{
    Opcode &op (inst()->ops()[opnum]);
    if (op.opname() == u_nop)
        return 0;   // Wasn't a real instruction to start with

    // Find the next instruction
    int op2num = next_block_instruction (opnum);
    if (! op2num)
        return 0;    // Not a next instruction within the same block

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

    // No changes
    return 0;
}



/// Mark our params that feed to later layers, and whether we have any
/// outgoing connections.
void
RuntimeOptimizer::mark_outgoing_connections ()
{
    inst()->outgoing_connections (false);
    FOREACH_PARAM (Symbol &s, inst())
        s.connected_down (false);
    for (int lay = m_layer+1;  lay < m_group.nlayers();  ++lay) {
        BOOST_FOREACH (Connection &c, m_group[lay]->m_connections)
            if (c.srclayer == m_layer) {
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
            turn_into_nop (s.initbegin(), s.initend(),
                           "remove init ops of unused param");
            s.set_initrange (0, 0);
            s.clear_rw();   // mark as totally unused
            ++alterations;
            if (debug() > 1)
                std::cout << "Realized that param " << s.name() << " is not needed\n";
        }
    }

    // Get rid of the Connections themselves
    erase_if (inst()->connections(), param_never_used);

    return alterations;
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

    // Turn all geom-locked parameters into constants.
    if (optimize() >= 2 && m_opt_constant_param) {
        find_constant_params (group());
    }

#ifdef DEBUG
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
    // passes, but we have a hard cutoff at 10 just to be sure we don't
    // ever get into an infinite loop from an unforseen cycle where we
    // end up inadvertently transforming A => B => A => etc.
    int totalchanged = 0;
    int reallydone = 0;   // Force a few passes after we think we're done
    for (m_pass = 0;  m_pass < 10;  ++m_pass) {

        // Once we've made one pass (and therefore called
        // mark_outgoing_connections), we may notice that the layer is
        // unused, and therefore can stop doing work to optimize it.
        if (m_pass != 0 && inst()->unused())
            break;

        // Track basic blocks and conditional states
        find_conditionals ();
        find_basic_blocks ();

        // Constant aliases valid for just this basic block
        clear_block_aliases ();

        // Clear local messages for this instance
        m_local_unknown_message_sent = false;
        m_local_messages_sent.clear ();

        int changed = 0;
        int lastblock = -1;
        size_t num_ops = inst()->ops().size();
        int skipops = 0;   // extra inserted ops to skip over
        for (int opnum = 0;  opnum < (int)num_ops;  opnum += 1) {
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
                clear_block_aliases ();
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
                const OpDescriptor *opd = m_shadingsys.op_descriptor (op.opname());
                if (opd && opd->folder) {
                    size_t old_num_ops = inst()->ops().size();
                    changed += (*opd->folder) (*this, opnum);
                    // Re-check num_ops in case the folder inserted something
                    num_ops = inst()->ops().size();
                    skipops = num_ops - old_num_ops;
                }
            }

            // Clear local block aliases for any args that were written
            // by this op
            block_unalias_written_args (op);

            // Get rid of an 'if' if it contains no statements to execute
            if (optimize() >= 2 && op.opname() == u_if &&
                    m_opt_constant_fold) {
                int jump = op.farthest_jump ();
                bool only_nops = true;
                for (int i = opnum+1;  i < jump && only_nops;  ++i)
                    only_nops &= (inst()->ops()[i].opname() == u_nop);
                if (only_nops) {
                    turn_into_nop (op, "'if' with no body");
                    changed = 1;
                    continue;
                }
            }

            // Now we handle assignments.
            if (optimize() >= 2 && op.opname() == u_assign &&
                    m_opt_assign) {
                Symbol *R (inst()->argsymbol(op.firstarg()+0));
                Symbol *A (inst()->argsymbol(op.firstarg()+1));
                bool R_local_or_tmp = (R->symtype() == SymTypeLocal ||
                                       R->symtype() == SymTypeTemp);

                if (block_alias(inst()->arg(op.firstarg())) == inst()->arg(op.firstarg()+1) ||
                    block_alias(inst()->arg(op.firstarg()+1)) == inst()->arg(op.firstarg())) {
                    // We're re-assigning something already aliased, skip it
                    turn_into_nop (op, "reassignment of current value (2)");
                    ++changed;
                    continue;
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

                if ((A->is_constant() || A->lastwrite() < opnum) &&
                    equivalent(R->typespec(), A->typespec())) {
                    // Safe to alias R to A for this block, if A is a
                    // constant or if it's never written to again.
                    block_alias (inst()->arg(op.firstarg()),
                                     inst()->arg(op.firstarg()+1));
//                  std::cerr << opnum << " aliasing " << R->mangled() << " to "
//                        << inst()->argsymbol(op.firstarg()+1)->mangled() << "\n";
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
                    ++changed;
                    continue;
                }
                if (R_local_or_tmp && ! R->everread()) {
                    // This local is written but NEVER READ.  nop it.
                    turn_into_nop (op, "local/tmp never read");
                    ++changed;
                    continue;
                }
                if (outparam_assign_elision (opnum, op)) {
                    ++changed;
                    continue;
                }
                if (R == A) {
                    // Just an assignment to itself -- turn into NOP!
                    turn_into_nop (op, "self-assignment");
                    ++changed;
                } else if (R_local_or_tmp && R->lastread() < opnum
                           && ! m_in_loop[opnum]) {
                    // Don't bother assigning if we never read it again
                    turn_into_nop (op, "symbol never read again");
                    ++changed;
                }
            }

            if (optimize() >= 2 && m_opt_elide_useless_ops)
                changed += useless_op_elision (op, opnum);

            // Peephole optimization involving pair of instructions
            if (optimize() >= 2 && m_opt_peephole)
                changed += peephole2 (opnum);
        }

        // Now that we've rewritten the code, we need to re-track the
        // variable lifetimes.
        track_variable_lifetimes ();

        // Recompute which of our params have downstream connections.
        mark_outgoing_connections ();

        // Elide unconnected parameters that are never read.
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
            if (s->connected())
                turn_into_assign_one (op, "resolve isconnected() [1]");
            else
                turn_into_assign_zero (op, "resolve isconnected() [0]");
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



// Add to the dependency map that "symbol A depends on symbol B".
void
RuntimeOptimizer::add_dependency (SymDependency &dmap, int A, int B)
{
    ASSERT (A < (int)inst()->symbols().size());
    ASSERT (B < (int)inst()->symbols().size());
    dmap[A].insert (B);
    // Unification -- make all of B's dependencies be dependencies of A.
    BOOST_FOREACH (int r, dmap[B])
        dmap[A].insert (r);
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
    BOOST_FOREACH (int d, symdeps[DerivSym]) {
        Symbol *s = inst()->symbol(d);
        if (! s->typespec().is_closure_based() && 
                s->typespec().elementtype().is_floatbased())
            s->has_derivs (true);
    }

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

#if 0
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
    FOREACH_PARAM (Symbol &s, inst())
        s.connected_down (false);
    for (int lay = m_layer+1;  lay < m_group.nlayers();  ++lay) {
        BOOST_FOREACH (Connection &c, m_group[lay]->m_connections)
            if (c.srclayer == m_layer)
                inst()->symbol(c.src.param)->connected_down (true);
    }

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
    for (int lay = m_layer+1;  lay < m_group.nlayers();  ++lay) {
        BOOST_FOREACH (Connection &c, m_group[lay]->m_connections)
            if (c.srclayer == m_layer)
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
#ifdef DEBUG
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



void
RuntimeOptimizer::optimize_group ()
{
    Timer rop_timer;
    if (debug())
        m_shadingsys.info ("About to optimize shader group %s:",
                           m_group.name().c_str());
    int nlayers = (int) m_group.nlayers ();

    // Clear info about which messages have been set
    m_unknown_message_sent = false;
    m_messages_sent.clear ();

    // If no closures were provided, register the builtin ones
    if (m_shadingsys.m_closure_registry.empty())
        m_shadingsys.register_builtin_closures();

    // Optimize each layer, from first to last
    size_t old_nsyms = 0, old_nops = 0;
    for (int layer = 0;  layer < nlayers;  ++layer) {
        set_inst (layer);
        if (inst()->unused())
            continue;
        m_inst->copy_code_from_master ();
        if (debug() && optimize() >= 1) {
            std::cout.flush ();
            std::cout << "Before optimizing layer " << layer << " " 
                      << inst()->layername() 
                      << ", I get:\n" << inst()->print()
                      << "\n--------------------------------\n\n";
            std::cout.flush ();
        }

        old_nsyms += inst()->symbols().size();
        old_nops += inst()->ops().size();

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
                Symbol *source = m_group[c.srclayer]->symbol(c.src.param);
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
        if (! inst()->unused())
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
            if (debug()) {
                track_variable_lifetimes ();
                std::cout << "After optimizing layer " << layer << " " 
                          << inst()->layername() << " (" << inst()->id()
                          << "): \n" << inst()->print() 
                          << "\n--------------------------------\n\n";
                std::cout.flush ();
            }
        }
        new_nsyms += inst()->symbols().size();
        new_nops += inst()->ops().size();
    }

    m_stat_specialization_time = rop_timer();

    Timer timer;
    build_llvm_group ();
    m_stat_total_llvm_time = timer();

    // Once we're generated the IR, we really don't need the ops and args,
    // and we only need the syms that include the params.
    off_t symmem = 0;
    size_t connectionmem = 0;
    for (int layer = 0;  layer < nlayers;  ++layer) {
        set_inst (layer);
        // We no longer needs ops and args -- create empty vectors and
        // swap with the ones in the instance.
        OpcodeVec noops;
        std::swap (inst()->ops(), noops);
        std::vector<int> noargs;
        std::swap (inst()->args(), noargs);

        if (inst()->unused()) {
            // If we'll never use the layer, we don't need the syms at all
            SymbolVec nosyms;
            std::swap (inst()->symbols(), nosyms);
            symmem += vectorbytes(nosyms);
            // also don't need the connection info any more
            connectionmem += (off_t) inst()->clear_connections ();
        }
    }
    {
        // adjust memory stats
        ShadingSystemImpl &ss (shadingsys());
        spin_lock lock (ss.m_stat_mutex);
        ss.m_stat_mem_inst_syms -= symmem;
        ss.m_stat_mem_inst_connections -= connectionmem;
        ss.m_stat_mem_inst -= symmem + connectionmem;
        ss.m_stat_memory -= symmem + connectionmem;
        ss.m_stat_preopt_syms += old_nsyms;
        ss.m_stat_preopt_ops += old_nops;
        ss.m_stat_postopt_syms += new_nsyms;
        ss.m_stat_postopt_ops += new_nops;
        ss.m_stat_max_llvm_local_mem = std::max (ss.m_stat_max_llvm_local_mem,
                                                  m_llvm_local_mem);
    }

    if (m_shadingsys.m_compile_report) {
        if (m_group.name()) {
            m_shadingsys.info ("Optimized shader group %s:", m_group.name().c_str());
            m_shadingsys.info ("    New syms %llu/%llu (%5.1f%%), ops %llu/%llu (%5.1f%%)",
              new_nsyms, old_nsyms,
              100.0*double((long long)new_nsyms-(long long)old_nsyms)/double(old_nsyms),
              new_nops, old_nops,
              100.0*double((long long)new_nops-(long long)old_nops)/double(old_nops));
        } else {
            m_shadingsys.info ("Optimized shader group: New syms %llu/%llu (%5.1f%%), ops %llu/%llu (%5.1f%%)",
              new_nsyms, old_nsyms,
              100.0*double((long long)new_nsyms-(long long)old_nsyms)/double(old_nsyms),
              new_nops, old_nops,
              100.0*double((long long)new_nops-(long long)old_nops)/double(old_nops));
        }
        m_shadingsys.info ("    (%1.2fs = %1.2f spc, %1.2f lllock, %1.2f llset, %1.2f ir, %1.2f opt, %1.2f jit; local mem %dKB)",
                           m_stat_total_llvm_time+m_stat_specialization_time,
                           m_stat_specialization_time, 
                           m_stat_opt_locking_time, m_stat_llvm_setup_time,
                           m_stat_llvm_irgen_time, m_stat_llvm_opt_time,
                           m_stat_llvm_jit_time,
                           m_llvm_local_mem/1024);
    }
}



void
ShadingSystemImpl::optimize_group (ShadingAttribState &attribstate, 
                                   ShaderGroup &group)
{
    Timer timer;
    lock_guard lock (group.m_mutex);
    if (group.optimized()) {
        // The group was somehow optimized by another thread between the
        // time we checked group.optimized() and now that we have the lock.
        // Nothing to do but record how long we waited for the lock.
        spin_lock stat_lock (m_stat_mutex);
        double t = timer();
        m_stat_optimization_time += t;
        m_stat_opt_locking_time += t;
        return;
    }

    if (m_only_groupname && m_only_groupname != group.name()) {
        // For debugging purposes, we are requested to compile only one
        // shader group, and this is not it.  Mark it as does_nothing,
        // and also as optimized so nobody locks on it again, and record
        // how long we waited for the lock.
        group.does_nothing (true);
        group.m_optimized = true;
        spin_lock stat_lock (m_stat_mutex);
        double t = timer();
        m_stat_optimization_time += t;
        m_stat_opt_locking_time += t;
        return;
    }

    double locking_time = timer();

    ShadingContext *ctx = get_context ();
    RuntimeOptimizer rop (*this, group, ctx);
    rop.optimize_group ();
    release_context (ctx);

    attribstate.changed_shaders ();
    group.m_optimized = true;
    spin_lock stat_lock (m_stat_mutex);
    m_stat_optimization_time += timer();
    m_stat_opt_locking_time += locking_time + rop.m_stat_opt_locking_time;
    m_stat_specialization_time += rop.m_stat_specialization_time;
    m_stat_total_llvm_time += rop.m_stat_total_llvm_time;
    m_stat_llvm_setup_time += rop.m_stat_llvm_setup_time;
    m_stat_llvm_irgen_time += rop.m_stat_llvm_irgen_time;
    m_stat_llvm_opt_time += rop.m_stat_llvm_opt_time;
    m_stat_llvm_jit_time += rop.m_stat_llvm_jit_time;
    m_stat_groups_compiled += 1;
    m_stat_instances_compiled += group.nlayers();
}



static void optimize_all_groups_wrapper (ShadingSystemImpl *ss)
{
    ss->optimize_all_groups (1);
}



void
ShadingSystemImpl::optimize_all_groups (int nthreads)
{
    if (! m_greedyjit) {
        // No greedy JIT, just free any groups we've recorded
        spin_lock lock (m_groups_to_compile_mutex);
        m_groups_to_compile.clear ();
        m_groups_to_compile_count = 0;
        return;
    }

    // Spawn a bunch of threads to do this in parallel -- just call this
    // routine again (with threads=1) for each thread.
    if (nthreads < 1)  // threads <= 0 means use all hardware available
        nthreads = std::min ((int)boost::thread::hardware_concurrency(),
                             (int)m_groups_to_compile_count);
    if (nthreads > 1) {
        if (m_threads_currently_compiling)
            return;   // never mind, somebody else spawned the JIT threads
        boost::thread_group threads;
        m_threads_currently_compiling += nthreads;
        for (int t = 0;  t < nthreads;  ++t)
            threads.add_thread (new boost::thread (optimize_all_groups_wrapper, this));
        threads.join_all ();
        m_threads_currently_compiling -= nthreads;
        return;
    }

    // And here's the single thread case
    while (m_groups_to_compile_count) {
        ShadingAttribStateRef sas;
        {
            spin_lock lock (m_groups_to_compile_mutex);
            if (m_groups_to_compile.size() == 0)
                return;  // Nothing left to compile
            sas = m_groups_to_compile.back ();
            m_groups_to_compile.pop_back ();
        }
        --m_groups_to_compile_count;
        if (! sas.unique()) {   // don't compile if nobody recorded it but us
            ShaderGroup &sgroup (sas->shadergroup (ShadUseSurface));
                optimize_group (*sas, sgroup);
        }
    }
}



int
ShadingSystemImpl::merge_instances (ShaderGroup &group, bool post_opt)
{
    // Look through the shader group for pairs of nodes/layers that
    // actually do exactly the same thing, and eliminate one of the
    // rundantant shaders, carefully rewiring all its outgoing
    // connections to later layers to refer to the one we keep.
    //
    // It turns out that in practice, it's not uncommon to have
    // duplicate nodes.  For example, some materials are "layered" --
    // like a character skin shader that has separate sub-networks for
    // skin, oil, wetness, and so on -- and those different sub-nets
    // often reference the same texture maps or noise functions by
    // repetition.  Yes, ideally, the redundancies would be eliminated
    // before they were fed to the renderer, but in practice that's hard
    // and for many scenes we get substantial savings of time (mostly
    // because of reduced texture calls) and instance memory by finding
    // these redundancies automatically.  The amount of savings is quite
    // scene dependent, as well as probably very dependent on the
    // general shading and lookdev approach of the studio.  But it was
    // very helpful for us in many cases.
    //
    // The basic loop below looks very inefficient, O(n^2) in number of
    // instances in the group. But it's really not -- a few seconds (sum
    // of all threads) for even our very complex scenes. This is because
    // most potential pairs have a very fast rejection case if they are
    // not using the same master.  Since there's no appreciable cost to
    // the brute force approach, it seems silly to have a complex scheme
    // to try to reduce the number of pairings.

    if (! m_opt_merge_instances)
        return 0;

    Timer timer;                // Time we spend looking for and doing merges
    int merges = 0;             // number of merges we do
    size_t connectionmem = 0;   // Connection memory we free
    int nlayers = group.nlayers();

    // Loop over all layers...
    for (int a = 0;  a < nlayers;  ++a) {
        if (group[a]->unused())    // Don't merge a layer that's not used
            continue;
        // Check all later layers...
        for (int b = a+1;  b < nlayers;  ++b) {
            if (group[b]->unused())    // Don't merge a layer that's not used
                continue;

            // Now we have two used layers, a and b, to examine.
            // See if they are mergeable (identical).  All the heavy
            // lifting is done by ShaderInstance::mergeable().
            if (! group[a]->mergeable (*group[b], group))
                continue;

            // The two nodes a and b are mergeable, so merge them.
            ShaderInstance *A = group[a];
            ShaderInstance *B = group[b];
            ++merges;

            // We'll keep A, get rid of B.  For all layers later than B,
            // check its incoming connections and replace all references
            // to B with references to A.
            for (int j = b+1;  j < nlayers;  ++j) {
                ShaderInstance *inst = group[j];
                if (inst->unused())  // don't bother if it's unused
                    continue;
                for (int c = 0, ce = inst->nconnections();  c < ce;  ++c) {
                    Connection &con = inst->connection(c);
                    if (con.srclayer == b) {
                        con.srclayer = a;
                        if (B->symbols().size()) {
                            ASSERT (A->symbol(con.src.param)->name() ==
                                    B->symbol(con.src.param)->name());
                        }
                    }
                }
            }

            // Mark parameters of B as no longer connected
            for (int p = B->firstparam();  p < B->lastparam();  ++p) {
                if (B->symbols().size())
                    B->symbol(p)->connected_down(false);
                if (B->m_instoverrides.size())
                    B->instoverride(p)->connected_down(false);
            }
            // B won't be used, so mark it as having no outgoing
            // connections and clear its incoming connections (which are
            // no longer used).
            B->outgoing_connections (false);
            B->run_lazily (true);
            connectionmem += B->clear_connections ();
            ASSERT (B->unused());
        }
    }

    {
        // Adjust stats
        spin_lock lock (m_stat_mutex);
        m_stat_mem_inst_connections -= connectionmem;
        m_stat_mem_inst -= connectionmem;
        m_stat_memory -= connectionmem;
        if (post_opt)
            m_stat_merged_inst_opt += merges;
        else
            m_stat_merged_inst += merges;
        m_stat_inst_merge_time += timer();
    }

    return merges;
}


}; // namespace pvt
OSL_NAMESPACE_EXIT
