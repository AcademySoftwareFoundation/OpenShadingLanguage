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
#include <cmath>

#include <boost/foreach.hpp>
#include <boost/regex.hpp>

#include <OpenImageIO/hash.h>
#include <OpenImageIO/timer.h>
#include <OpenImageIO/thread.h>

#include "oslexec_pvt.h"
#include "runtimeoptimize.h"
#include "../liboslcomp/oslcomp_pvt.h"
#include "dual.h"
using namespace OSL;
using namespace OSL::pvt;


// names of ops we'll be using frequently
static ustring u_nop    ("nop"),
               u_assign ("assign"),
               u_add    ("add"),
               u_sub    ("sub"),
               u_if     ("if"),
               u_break ("break"),
               u_continue ("continue"),
               u_return ("return"),
               u_useparam ("useparam"),
               u_setmessage ("setmessage"),
               u_getmessage ("getmessage");


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
namespace pvt {   // OSL::pvt

#ifdef OIIO_NAMESPACE
using OIIO::spin_lock;
using OIIO::Timer;
#endif

/// Wrapper that erases elements of c for which predicate p is true.
/// (Unlike std::remove_if, it resizes the container so that it contains
/// ONLY elements for which the predicate is true.)
template<class Container, class Predicate>
void erase_if (Container &c, const Predicate &p)
{
    c.erase (std::remove_if (c.begin(), c.end(), p), c.end());
}



RuntimeOptimizer::RuntimeOptimizer (ShadingSystemImpl &shadingsys,
                                    ShaderGroup &group)
    : m_shadingsys(shadingsys),
      m_thread(shadingsys.get_perthread_info()),
      m_group(group),
      m_inst(NULL),
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
      m_next_newconst(0),
      m_stat_opt_locking_time(0), m_stat_specialization_time(0),
      m_stat_total_llvm_time(0), m_stat_llvm_setup_time(0),
      m_stat_llvm_irgen_time(0), m_stat_llvm_opt_time(0),
      m_stat_llvm_jit_time(0),
      m_llvm_context(NULL), m_llvm_module(NULL),
      m_llvm_exec(NULL), m_builder(NULL),
      m_llvm_passes(NULL), m_llvm_func_passes(NULL)
{
    set_debug ();
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
RuntimeOptimizer::add_constant (const TypeSpec &type, const void *data)
{
    int ind = find_constant (type, data);
    if (ind < 0) {
        Symbol newconst (ustring::format ("$newconst%d", m_next_newconst++),
                         type, SymTypeConst);
        void *newdata;
        TypeDesc t (type.simpletype());
        size_t n = t.aggregate * t.numelements();
        if (t.basetype == TypeDesc::INT)
            newdata = inst()->shadingsys().alloc_int_constants (n);
        else if (t.basetype == TypeDesc::FLOAT)
            newdata = inst()->shadingsys().alloc_float_constants (n);
        else if (t.basetype == TypeDesc::STRING)
            newdata = inst()->shadingsys().alloc_string_constants (n);
        else { ASSERT (0 && "unsupported type for add_constant"); }
        memcpy (newdata, data, t.size());
        newconst.data (newdata);
        ASSERT (inst()->symbols().capacity() > inst()->symbols().size() &&
                "we shouldn't have to realloc here");
        ind = (int) inst()->symbols().size ();
        inst()->symbols().push_back (newconst);
        m_all_consts.push_back (ind);
    }
    return ind;
}



void
RuntimeOptimizer::turn_into_assign (Opcode &op, int newarg, const char *why)
{
    int opnum = &op - &(inst()->ops()[0]);
    if (debug() > 1)
        std::cout << "turned op " << opnum
                  << " from " << op.opname() << " to "
                  << opargsym(op,0)->name() << " = " << opargsym(op,1)->name()
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



/// Insert instruction 'opname' with arguments 'args_to_add' into the 
/// code at instruction 'opnum'.  The existing code and concatenated 
/// argument lists can be found in code and opargs, respectively, and
/// allsyms contains pointers to all symbols.  mainstart is a reference
/// to the address where the 'main' shader begins, and may be modified
/// if the new instruction is inserted before that point.
/// If recompute_rw_ranges is true, also adjust all symbols' read/write
/// ranges to take the new instruction into consideration.
void
RuntimeOptimizer::insert_code (int opnum, ustring opname,
                               const std::vector<int> &args_to_add,
                               bool recompute_rw_ranges)
{
    OpcodeVec &code (inst()->ops());
    std::vector<int> &opargs (inst()->args());
    ustring method = (opnum < (int)code.size()) ? code[opnum].method() : OSLCompilerImpl::main_method_name();
    Opcode op (opname, method, opargs.size(), args_to_add.size());
    code.insert (code.begin()+opnum, op);
    opargs.insert (opargs.end(), args_to_add.begin(), args_to_add.end());
    if (opnum < inst()->m_maincodebegin)
        ++inst()->m_maincodebegin;
    ++inst()->m_maincodeend;

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
                if (first > opnum)
                    ++first;
                if (last >= opnum)
                    ++last;
                s.set_read (first, last);
            }
            if (s.everwritten()) {
                int first = s.firstwrite(), last = s.lastwrite();
                if (first > opnum)
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

    if (opname != u_useparam) {
        // Mark the args as being used for this op (assume that the
        // first is written, the others are read).  Enforce that with an
        // DASSERT to be sure we only use insert_code for the couple of
        // instructions that we think it is used for.
        DASSERT (opname == u_assign);
        for (size_t a = 0;  a < args_to_add.size();  ++a)
            inst()->symbol(args_to_add[a])->mark_rw (opnum, a>0, a==0);
    }
}



/// Insert a 'useparam' instruction in front of instruction 'opnum', to
/// reference the symbols in 'params'.
void
RuntimeOptimizer::insert_useparam (size_t opnum,
                                   std::vector<int> &params_to_use)
{
    OpcodeVec &code (inst()->ops());
    insert_code (opnum, u_useparam, params_to_use);

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



inline bool
equal_consts (const Symbol &A, const Symbol &B)
{
    return (&A == &B ||
            (equivalent (A.typespec(), B.typespec()) &&
             !memcmp (A.data(), B.data(), A.typespec().simpletype().size())));
}



inline bool
unequal_consts (const Symbol &A, const Symbol &B)
{
    return (equivalent (A.typespec(), B.typespec()) &&
            memcmp (A.data(), B.data(), A.typespec().simpletype().size()));
}



inline bool
is_zero (const Symbol &A)
{
    const TypeSpec &Atype (A.typespec());
    static Vec3 Vzero (0, 0, 0);
    return (Atype.is_float() && *(const float *)A.data() == 0) ||
        (Atype.is_int() && *(const int *)A.data() == 0) ||
        (Atype.is_triple() && *(const Vec3 *)A.data() == Vzero);
}



inline bool
is_one (const Symbol &A)
{
    const TypeSpec &Atype (A.typespec());
    static Vec3 Vone (1, 1, 1);
    static Matrix44 Mone (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    return (Atype.is_float() && *(const float *)A.data() == 1) ||
        (Atype.is_int() && *(const int *)A.data() == 1) ||
        (Atype.is_triple() && *(const Vec3 *)A.data() == Vone) ||
        (Atype.is_matrix() && *(const Matrix44 *)A.data() == Mone);
}



DECLFOLDER(constfold_none)
{
    return 0;
}



DECLFOLDER(constfold_add)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &B (*rop.inst()->argsymbol(op.firstarg()+2));
    if (A.is_constant()) {
        if (is_zero(A)) {
            // R = 0 + B  =>   R = B
            rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+2),
                                  "const fold");
            return 1;
        }
    }
    if (B.is_constant()) {
        if (is_zero(B)) {
            // R = A + 0   =>   R = A
            rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+1),
                                  "const fold");
            return 1;
        }
    }
    if (A.is_constant() && B.is_constant()) {
        if (A.typespec().is_int() && B.typespec().is_int()) {
            int result = *(int *)A.data() + *(int *)B.data();
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        } else if (A.typespec().is_float() && B.typespec().is_float()) {
            float result = *(float *)A.data() + *(float *)B.data();
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        } else if (A.typespec().is_triple() && B.typespec().is_triple()) {
            Vec3 result = *(Vec3 *)A.data() + *(Vec3 *)B.data();
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        }
    }
    return 0;
}



DECLFOLDER(constfold_sub)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &B (*rop.inst()->argsymbol(op.firstarg()+2));
    if (B.is_constant()) {
        if (is_zero(B)) {
            // R = A - 0   =>   R = A
            rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+1),
                                  "subtract zero");
            return 1;
        }
    }
    // R = A - B, if both are constants, =>  R = C
    if (A.is_constant() && B.is_constant()) {
        if (A.typespec().is_int() && B.typespec().is_int()) {
            int result = *(int *)A.data() - *(int *)B.data();
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        } else if (A.typespec().is_float() && B.typespec().is_float()) {
            float result = *(float *)A.data() - *(float *)B.data();
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        } else if (A.typespec().is_triple() && B.typespec().is_triple()) {
            Vec3 result = *(Vec3 *)A.data() - *(Vec3 *)B.data();
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        }
    }
    // R = A - A  =>  R = 0    even if not constant!
    if (&A == &B) {
        rop.turn_into_assign_zero (op, "sub from itself");
    }
    return 0;
}



DECLFOLDER(constfold_mul)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &B (*rop.inst()->argsymbol(op.firstarg()+2));
    if (A.is_constant()) {
        if (is_one(A)) {
            // R = 1 * B  =>   R = B
            rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+2),
                                  "mul by 1");
            return 1;
        }
        if (is_zero(A)) {
            // R = 0 * B  =>   R = 0
            rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+1),
                                  "mul by 0");
            return 1;
        }
    }
    if (B.is_constant()) {
        if (is_one(B)) {
            // R = A * 1   =>   R = A
            rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+1),
                                  "mul by 1");
            return 1;
        }
        if (is_zero(B)) {
            // R = A * 0   =>   R = 0
            rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+2),
                                  "mul by 0");
            return 1;
        }
    }
    if (A.is_constant() && B.is_constant()) {
        if (A.typespec().is_int() && B.typespec().is_int()) {
            int result = *(int *)A.data() * *(int *)B.data();
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        } else if (A.typespec().is_float() && B.typespec().is_float()) {
            float result = (*(float *)A.data()) * (*(float *)B.data());
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        } else if (A.typespec().is_triple() && B.typespec().is_triple()) {
            Vec3 result = (*(Vec3 *)A.data()) * (*(Vec3 *)B.data());
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        } else if (A.typespec().is_triple() && B.typespec().is_float()) {
            Vec3 result = (*(Vec3 *)A.data()) * (*(float *)B.data());
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        } else if (A.typespec().is_float() && B.typespec().is_triple()) {
            Vec3 result = (*(float *)A.data()) * (*(Vec3 *)B.data());
            int cind = rop.add_constant (B.typespec(), &result);
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        }
    }
    return 0;
}



DECLFOLDER(constfold_div)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &R (*rop.inst()->argsymbol(op.firstarg()+0));
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &B (*rop.inst()->argsymbol(op.firstarg()+2));
    if (B.is_constant()) {
        if (is_one(B)) {
            // R = A / 1   =>   R = A
            rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+1),
                                  "div by 1");
            return 1;
        }
        if (is_zero(B) && (B.typespec().is_float() ||
                           B.typespec().is_triple() || B.typespec().is_int())) {
            // R = A / 0   =>   R = 0      because of OSL div by zero rule
            rop.turn_into_assign_zero (op, "div by 0");
            return 1;
        }
    }
    if (A.is_constant() && B.is_constant()) {
        int cind = -1;
        if (A.typespec().is_int() && B.typespec().is_int()) {
            int result = *(int *)A.data() / *(int *)B.data();
            cind = rop.add_constant (R.typespec(), &result);
        } else if (A.typespec().is_float() && B.typespec().is_int()) {
            float result = *(float *)A.data() / *(int *)B.data();
            cind = rop.add_constant (R.typespec(), &result);
        } else if (A.typespec().is_float() && B.typespec().is_float()) {
            float result = *(float *)A.data() / *(float *)B.data();
            cind = rop.add_constant (R.typespec(), &result);
        } else if (A.typespec().is_int() && B.typespec().is_float()) {
            float result = *(int *)A.data() / *(float *)B.data();
            cind = rop.add_constant (R.typespec(), &result);
        } else if (A.typespec().is_triple() && B.typespec().is_triple()) {
            Vec3 result = *(Vec3 *)A.data() / *(Vec3 *)B.data();
            cind = rop.add_constant (R.typespec(), &result);
        } else if (A.typespec().is_triple() && B.typespec().is_float()) {
            Vec3 result = *(Vec3 *)A.data() / *(float *)B.data();
            cind = rop.add_constant (R.typespec(), &result);
        } else if (A.typespec().is_float() && B.typespec().is_triple()) {
            float a = *(float *)A.data();
            Vec3 result = Vec3(a,a,a) / *(Vec3 *)B.data();
            cind = rop.add_constant (R.typespec(), &result);
        }
        if (cind >= 0) {
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        }
    }
    return 0;
}



DECLFOLDER(constfold_dot)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &B (*rop.inst()->argsymbol(op.firstarg()+2));

    // Dot with (0,0,0) -> 0
    if ((A.is_constant() && is_zero(A)) || (B.is_constant() && is_zero(B))) {
        rop.turn_into_assign_zero (op, "dot with 0");
        return 1;
    }

    // dot(const,const) -> const
    if (A.is_constant() && B.is_constant()) {
        DASSERT (A.typespec().is_triple() && B.typespec().is_triple());
        float result = (*(Vec3 *)A.data()).dot (*(Vec3 *)B.data());
        int cind = rop.add_constant (TypeDesc::TypeFloat, &result);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }

    return 0;
}



DECLFOLDER(constfold_neg)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1));
    if (A.is_constant()) {
        if (A.typespec().is_int()) {
            int result =  - *(int *)A.data();
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        } else if (A.typespec().is_float()) {
            float result =  - *(float *)A.data();
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        } else if (A.typespec().is_triple()) {
            Vec3 result = - *(Vec3 *)A.data();
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        }
    }
    return 0;
}



DECLFOLDER(constfold_abs)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1));
    if (A.is_constant()) {
        if (A.typespec().is_int()) {
            int result = std::abs(*(int *)A.data());
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        } else if (A.typespec().is_float()) {
            float result =  std::abs(*(float *)A.data());
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        } else if (A.typespec().is_triple()) {
            Vec3 result = *(Vec3 *)A.data();
            result.x = std::abs(result.x);
            result.y = std::abs(result.y);
            result.z = std::abs(result.z);
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        }
    }
    return 0;
}



DECLFOLDER(constfold_eq)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &B (*rop.inst()->argsymbol(op.firstarg()+2));
    if (A.is_constant() && B.is_constant()) {
        bool val = false;
        if (equivalent (A.typespec(), B.typespec())) {
            val = equal_consts (A, B);
        } else if (A.typespec().is_float() && B.typespec().is_int()) {
            val = (*(float *)A.data() == *(int *)B.data());
        } else if (A.typespec().is_int() && B.typespec().is_float()) {
            val = (*(int *)A.data() == *(float *)B.data());
        } else {
            return 0;  // unhandled cases
        }
        // Turn the 'eq R A B' into 'assign R X' where X is 0 or 1.
        static const int int_zero = 0, int_one = 1;
        int cind = rop.add_constant (TypeDesc::TypeInt,
                                     val ? &int_one : &int_zero);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_neq)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &B (*rop.inst()->argsymbol(op.firstarg()+2));
    if (A.is_constant() && B.is_constant()) {
        bool val = false;
        if (equivalent (A.typespec(), B.typespec())) {
            val = ! equal_consts (A, B);
        } else if (A.typespec().is_float() && B.typespec().is_int()) {
            val = (*(float *)A.data() != *(int *)B.data());
        } else if (A.typespec().is_int() && B.typespec().is_float()) {
            val = (*(int *)A.data() != *(float *)B.data());
        } else {
            return 0;  // unhandled case
        }
        // Turn the 'neq R A B' into 'assign R X' where X is 0 or 1.
        static const int int_zero = 0, int_one = 1;
        int cind = rop.add_constant (TypeDesc::TypeInt,
                                     val ? &int_one : &int_zero);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_lt)
{
    static const int int_zero = 0, int_one = 1;
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &B (*rop.inst()->argsymbol(op.firstarg()+2));
    const TypeSpec &ta (A.typespec()); 
    const TypeSpec &tb (B.typespec()); 
    if (A.is_constant() && B.is_constant()) {
        // Turn the 'leq R A B' into 'assign R X' where X is 0 or 1.
        bool val = false;
        if (ta.is_float() && tb.is_float()) {
            val = (*(float *)A.data() < *(float *)B.data());
        } else if (ta.is_float() && tb.is_int()) {
            val = (*(float *)A.data() < *(int *)B.data());
        } else if (ta.is_int() && tb.is_float()) {
            val = (*(int *)A.data() < *(float *)B.data());
        } else if (ta.is_int() && tb.is_int()) {
            val = (*(int *)A.data() < *(int *)B.data());
        } else {
            return 0;  // unhandled case
        }
        int cind = rop.add_constant (TypeDesc::TypeInt,
                                     val ? &int_one : &int_zero);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_le)
{
    static const int int_zero = 0, int_one = 1;
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &B (*rop.inst()->argsymbol(op.firstarg()+2));
    const TypeSpec &ta (A.typespec()); 
    const TypeSpec &tb (B.typespec()); 
    if (A.is_constant() && B.is_constant()) {
        // Turn the 'leq R A B' into 'assign R X' where X is 0 or 1.
        bool val = false;
        if (ta.is_float() && tb.is_float()) {
            val = (*(float *)A.data() <= *(float *)B.data());
        } else if (ta.is_float() && tb.is_int()) {
            val = (*(float *)A.data() <= *(int *)B.data());
        } else if (ta.is_int() && tb.is_float()) {
            val = (*(int *)A.data() <= *(float *)B.data());
        } else if (ta.is_int() && tb.is_int()) {
            val = (*(int *)A.data() <= *(int *)B.data());
        } else {
            return 0;  // unhandled case
        }
        int cind = rop.add_constant (TypeDesc::TypeInt,
                                     val ? &int_one : &int_zero);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_gt)
{
    static const int int_zero = 0, int_one = 1;
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &B (*rop.inst()->argsymbol(op.firstarg()+2));
    const TypeSpec &ta (A.typespec()); 
    const TypeSpec &tb (B.typespec()); 
    if (A.is_constant() && B.is_constant()) {
        // Turn the 'gt R A B' into 'assign R X' where X is 0 or 1.
        bool val = false;
        if (ta.is_float() && tb.is_float()) {
            val = (*(float *)A.data() > *(float *)B.data());
        } else if (ta.is_float() && tb.is_int()) {
            val = (*(float *)A.data() > *(int *)B.data());
        } else if (ta.is_int() && tb.is_float()) {
            val = (*(int *)A.data() > *(float *)B.data());
        } else if (ta.is_int() && tb.is_int()) {
            val = (*(int *)A.data() > *(int *)B.data());
        } else {
            return 0;  // unhandled case
        }
        int cind = rop.add_constant (TypeDesc::TypeInt,
                                     val ? &int_one : &int_zero);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_ge)
{
    static const int int_zero = 0, int_one = 1;
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &B (*rop.inst()->argsymbol(op.firstarg()+2));
    const TypeSpec &ta (A.typespec()); 
    const TypeSpec &tb (B.typespec()); 
    if (A.is_constant() && B.is_constant()) {
        // Turn the 'leq R A B' into 'assign R X' where X is 0 or 1.
        bool val = false;
        if (ta.is_float() && tb.is_float()) {
            val = (*(float *)A.data() >= *(float *)B.data());
        } else if (ta.is_float() && tb.is_int()) {
            val = (*(float *)A.data() >= *(int *)B.data());
        } else if (ta.is_int() && tb.is_float()) {
            val = (*(int *)A.data() >= *(float *)B.data());
        } else if (ta.is_int() && tb.is_int()) {
            val = (*(int *)A.data() >= *(int *)B.data());
        } else {
            return 0;  // unhandled case
        }
        int cind = rop.add_constant (TypeDesc::TypeInt,
                                     val ? &int_one : &int_zero);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_or)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &B (*rop.inst()->argsymbol(op.firstarg()+2));
    if (A.is_constant() && B.is_constant()) {
        DASSERT (A.typespec().is_int() && B.typespec().is_int());
        bool val = *(int *)A.data() || *(int *)B.data();
        // Turn the 'or R A B' into 'assign R X' where X is 0 or 1.
        static const int int_zero = 0, int_one = 1;
        int cind = rop.add_constant (TypeDesc::TypeInt,
                                     val ? &int_one : &int_zero);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_and)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &B (*rop.inst()->argsymbol(op.firstarg()+2));
    if (A.is_constant() && B.is_constant()) {
        // Turn the 'and R A B' into 'assign R X' where X is 0 or 1.
        DASSERT (A.typespec().is_int() && B.typespec().is_int());
        bool val = *(int *)A.data() && *(int *)B.data();
        static const int int_zero = 0, int_one = 1;
        int cind = rop.add_constant (TypeDesc::TypeInt,
                                     val ? &int_one : &int_zero);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_if)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &C (*rop.inst()->argsymbol(op.firstarg()+0));
    if (C.is_constant()) {
        int result = -1;   // -1 == we don't know
        if (C.typespec().is_int())
            result = (((int *)C.data())[0] != 0);
        else if (C.typespec().is_float())
            result = (((float *)C.data())[0] != 0.0f);
        else if (C.typespec().is_triple())
            result = (((Vec3 *)C.data())[0] != Vec3(0,0,0));
        else if (C.typespec().is_string()) {
            ustring s = ((ustring *)C.data())[0];
            result = (s.length() != 0);
        }
        int changed = 0;
        if (result > 0) {
            changed += rop.turn_into_nop (op.jump(0), op.jump(1), "elide 'else'");
            changed += rop.turn_into_nop (op, "elide 'else'");
        } else if (result == 0) {
            changed += rop.turn_into_nop (opnum, op.jump(0), "elide 'if'");
        }
        return changed;
    }
    return 0;
}



// Is an array known to have all elements having the same value?
static bool
array_all_elements_equal (const Symbol &s)
{
    TypeDesc t = s.typespec().simpletype();
    size_t size = t.elementsize();
    size_t n = t.numelements();
    for (size_t i = 1;  i < n;  ++i)
        if (memcmp ((const char *)s.data(), (const char *)s.data()+i*size, size))
            return false;
    return true;
}



DECLFOLDER(constfold_aref)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &R (*rop.inst()->argsymbol(op.firstarg()+0));
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &Index (*rop.inst()->argsymbol(op.firstarg()+2));
    DASSERT (A.typespec().is_array() && Index.typespec().is_int());

    // Try to turn R=A[I] into R=C if A and I are const.
    if (A.is_constant() && Index.is_constant()) {
        TypeSpec elemtype = A.typespec().elementtype();
        ASSERT (equivalent(elemtype, R.typespec()));
        int index = *(int *)Index.data();
        if (index < 0 || index >= A.typespec().arraylength()) {
            // We are indexing a const array out of range.  But this
            // isn't necessarily a reportable error, because it may be a
            // code path that will never be taken.  Punt -- don't
            // optimize this op, leave it to the execute-time range
            // check to catch, if indeed it is a problem.
            return 0;
        }
        ASSERT (index < A.typespec().arraylength());
        int cind = rop.add_constant (elemtype,
                        (char *)A.data() + index*elemtype.simpletype().size());
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    // Even if the index isn't constant, we still know the answer if all
    // the array elements are equal!
    if (A.is_constant() && array_all_elements_equal(A)) {
        TypeSpec elemtype = A.typespec().elementtype();
        ASSERT (equivalent(elemtype, R.typespec()));
        int cind = rop.add_constant (elemtype, (char *)A.data());
        rop.turn_into_assign (op, cind, "aref of elements-equal array");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_arraylength)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &R (*rop.inst()->argsymbol(op.firstarg()+0));
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1));
    ASSERT (R.typespec().is_int() && A.typespec().is_array());

    // Try to turn R=arraylength(A) into R=C if the array length is known
    int len = A.typespec().arraylength();
    if (len > 0) {
        int cind = rop.add_constant (TypeSpec(TypeDesc::INT), &len);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_compassign)
{
    // Component assignment
    Opcode &op (rop.inst()->ops()[opnum]);
    // Symbol *A (rop.inst()->argsymbol(op.firstarg()+0));
    // We are obviously not assigning to a constant, but it could be
    // that at this point in our current block, the value of A is known,
    // and that will show up as a block alias.
    int Aalias = rop.block_alias (rop.inst()->arg(op.firstarg()+0));
    Symbol *AA = rop.inst()->symbol(Aalias);
    Symbol *I (rop.inst()->argsymbol(op.firstarg()+1));
    Symbol *C (rop.inst()->argsymbol(op.firstarg()+2));
    // N.B. symbol returns NULL if Aalias is < 0

    // Try to turn A[I]=C into nop if A[I] already is C
    // The optimization we are making here is that if the current (at
    // this point in this block) value of A is known (revealed by A's
    // block alias, AA, being a constant), and we are assigning the same
    // value it already has, then this is a nop.
    if (I->is_constant() && C->is_constant() && AA && AA->is_constant()) {
        ASSERT (AA->typespec().is_triple() &&
                (C->typespec().is_float() || C->typespec().is_int()));
        int index = *(int *)I->data();
        if (index < 0 || index >= 3) {
            // We are indexing a const triple out of range.  But this
            // isn't necessarily a reportable error, because it may be a
            // code path that will never be taken.  Punt -- don't
            // optimize this op, leave it to the execute-time range
            // check to catch, if indeed it is a problem.
            return 0;
        }
        float *aa = (float *)AA->data();
        float c = C->typespec().is_int() ? *(int *)C->data()
                                         : *(float *)C->data();
        if (aa[index] == c) {
            rop.turn_into_nop (op, "useless compassign");
            return 1;
        }
        // FIXME -- we can take this one step further, by giving A a new
        // alias that is the modified constant.
    }
    return 0;
}



DECLFOLDER(constfold_compref)
{
    // Component reference
    // Try to turn R=A[I] into R=C if A and I are const.
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &Index (*rop.inst()->argsymbol(op.firstarg()+2));
    if (A.is_constant() && Index.is_constant()) {
        ASSERT (A.typespec().is_triple() && Index.typespec().is_int());
        int index = *(int *)Index.data();
        if (index < 0 || index >= 3) {
            // We are indexing a const triple out of range.  But this
            // isn't necessarily a reportable error, because it may be a
            // code path that will never be taken.  Punt -- don't
            // optimize this op, leave it to the execute-time range
            // check to catch, if indeed it is a problem.
            return 0;
        }
        int cind = rop.add_constant (TypeDesc::TypeFloat, (float *)A.data() + index);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_strlen)
{
    // Try to turn R=strlen(s) into R=C
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &S (*rop.inst()->argsymbol(op.firstarg()+1));
    if (S.is_constant()) {
        ASSERT (S.typespec().is_string());
        int result = (int) (*(ustring *)S.data()).length();
        int cind = rop.add_constant (TypeDesc::TypeInt, &result);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_endswith)
{
    // Try to turn R=endswith(s,e) into R=C
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &S (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &E (*rop.inst()->argsymbol(op.firstarg()+2));
    if (S.is_constant() && E.is_constant()) {
        ASSERT (S.typespec().is_string() && E.typespec().is_string());
        ustring s = *(ustring *)S.data();
        ustring e = *(ustring *)E.data();
        size_t elen = e.length(), slen = s.length();
        int result = 0;
        if (elen <= slen)
            result = (strncmp (s.c_str()+slen-elen, e.c_str(), elen) == 0);
        int cind = rop.add_constant (TypeDesc::TypeInt, &result);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_concat)
{
    // Try to turn R=concat(s,...) into R=C
    Opcode &op (rop.inst()->ops()[opnum]);
    ustring result;
    for (int i = 1;  i < op.nargs();  ++i) {
        Symbol &S (*rop.inst()->argsymbol(op.firstarg()+i));
        if (! S.is_constant())
            return 0;  // something non-constant
        ustring old = result;
        ustring s = *(ustring *)S.data();
        result = ustring::format ("%s%s", old.c_str() ? old.c_str() : "",
                                  s.c_str() ? s.c_str() : "");
    }
    // If we made it this far, all args were constants, and the
    // concatenation is in result.
    int cind = rop.add_constant (TypeDesc::TypeString, &result);
    rop.turn_into_assign (op, cind, "const fold");
    return 1;
}



DECLFOLDER(constfold_format)
{
    // Try to turn R=format(fmt,...) into R=C
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &Format (*rop.opargsym(op, 1));
    ustring fmt = *(ustring *)Format.data();
    std::vector<void *> argptrs;
    for (int i = 2;  i < op.nargs();  ++i) {
        Symbol &S (*rop.opargsym(op, i));
        if (! S.is_constant())
            return 0;  // something non-constant
        argptrs.push_back (S.data());
    }
    // If we made it this far, all args were constants, and the
    // arg data pointers are in argptrs[].

    // It's actually a HUGE pain to make this work generally, because
    // the Strutil::vformat we use in the runtime implementation wants a
    // va_list, but we just have raw pointers at this point.  No matter,
    // let's just make it work for several simple common cases.
    if (op.nargs() == 3) {
        // Just result=format(fmt, one_argument)
        Symbol &Val (*rop.opargsym(op, 2));
        if (Val.typespec().is_string()) {
            // Single %s
            ustring result = ustring::format (fmt.c_str(),
                                              ((ustring *)Val.data())->c_str());
            int cind = rop.add_constant (TypeDesc::TypeString, &result);
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        }
    }

    return 0;
}



DECLFOLDER(constfold_regex_search)
{
    // Try to turn R=regex_search(subj,reg) into R=C
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &Subj (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &Reg (*rop.inst()->argsymbol(op.firstarg()+2));
    if (op.nargs() == 3 // only the 2-arg version without search results
          && Subj.is_constant() && Reg.is_constant()) {
        DASSERT (Subj.typespec().is_string() && Reg.typespec().is_string());
        const ustring &s (*(ustring *)Subj.data());
        const ustring &r (*(ustring *)Reg.data());
        boost::regex reg (r.string());
        int result = boost::regex_search (s.string(), reg);
        int cind = rop.add_constant (TypeDesc::TypeInt, &result);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



inline float clamp (float x, float minv, float maxv)
{
    if (x < minv) return minv;
    else if (x > maxv) return maxv;
    else return x;
}



DECLFOLDER(constfold_clamp)
{
    // Try to turn R=clamp(x,min,max) into R=C
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &X (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &Min (*rop.inst()->argsymbol(op.firstarg()+2));
    Symbol &Max (*rop.inst()->argsymbol(op.firstarg()+3));
    if (X.is_constant() && Min.is_constant() && Max.is_constant() &&
        equivalent(X.typespec(), Min.typespec()) &&
        equivalent(X.typespec(), Max.typespec()) &&
        (X.typespec().is_float() || X.typespec().is_triple())) {
        const float *x = (const float *) X.data();
        const float *min = (const float *) Min.data();
        const float *max = (const float *) Max.data();
        float result[3];
        result[0] = clamp (x[0], min[0], max[0]);
        if (X.typespec().is_triple()) {
            result[1] = clamp (x[1], min[1], max[1]);
            result[2] = clamp (x[2], min[2], max[2]);
        }
        int cind = rop.add_constant (X.typespec(), &result);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_min)
{
    // Try to turn R=min(x,y) into R=C
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &X (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &Y (*rop.inst()->argsymbol(op.firstarg()+2));
    if (X.is_constant() && Y.is_constant() &&
        equivalent(X.typespec(), Y.typespec()) &&
        (X.typespec().is_float() || X.typespec().is_triple())) {
        const float *x = (const float *) X.data();
        const float *y = (const float *) Y.data();
        float result[3];
        result[0] = std::min (x[0], y[0]);
        if (X.typespec().is_triple()) {
            result[1] = std::min (x[1], y[1]);
            result[2] = std::min (x[2], y[2]);
        }
        int cind = rop.add_constant (X.typespec(), &result);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_max)
{
    // Try to turn R=max(x,y) into R=C
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &X (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &Y (*rop.inst()->argsymbol(op.firstarg()+2));
    if (X.is_constant() && Y.is_constant() &&
        equivalent(X.typespec(), Y.typespec()) &&
        (X.typespec().is_float() || X.typespec().is_triple())) {
        const float *x = (const float *) X.data();
        const float *y = (const float *) Y.data();
        float result[3];
        result[0] = std::max (x[0], y[0]);
        if (X.typespec().is_triple()) {
            result[1] = std::max (x[1], y[1]);
            result[2] = std::max (x[2], y[2]);
        }
        int cind = rop.add_constant (X.typespec(), &result);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_sqrt)
{
    // Try to turn R=sqrt(x) into R=C
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &X (*rop.inst()->argsymbol(op.firstarg()+1));
    if (X.is_constant() &&
          (X.typespec().is_float() || X.typespec().is_triple())) {
        const float *x = (const float *) X.data();
        float result[3];
        result[0] = sqrtf (std::max (0.0f, x[0]));
        if (X.typespec().is_triple()) {
            result[1] = sqrtf (std::max (0.0f, x[1]));
            result[2] = sqrtf (std::max (0.0f, x[2]));
        }
        int cind = rop.add_constant (X.typespec(), &result);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_floor)
{
    // Try to turn R=floor(x) into R=C
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &X (*rop.inst()->argsymbol(op.firstarg()+1));
    if (X.is_constant() &&
          (X.typespec().is_float() || X.typespec().is_triple())) {
        const float *x = (const float *) X.data();
        float result[3];
        result[0] = floorf (x[0]);
        if (X.typespec().is_triple()) {
            result[1] = floorf (x[1]);
            result[2] = floorf (x[2]);
        }
        int cind = rop.add_constant (X.typespec(), &result);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_ceil)
{
    // Try to turn R=ceil(x) into R=C
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &X (*rop.inst()->argsymbol(op.firstarg()+1));
    if (X.is_constant() &&
          (X.typespec().is_float() || X.typespec().is_triple())) {
        const float *x = (const float *) X.data();
        float result[3];
        result[0] = ceilf (x[0]);
        if (X.typespec().is_triple()) {
            result[1] = ceilf (x[1]);
            result[2] = ceilf (x[2]);
        }
        int cind = rop.add_constant (X.typespec(), &result);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_pow)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &X (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &Y (*rop.inst()->argsymbol(op.firstarg()+2));

    if (Y.is_constant() && is_zero(Y)) {
        // x^0 == 1
        rop.turn_into_assign_one (op, "pow^0");
        return 1;
    }
    if (Y.is_constant() && is_one(Y)) {
        // x^1 == x
        rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+1), "pow^1");
        return 1;
    }
    if (X.is_constant() && is_zero(X)) {
        // 0^y == 0
        rop.turn_into_assign_zero (op, "pow 0^x");
        return 1;
    }
    if (X.is_constant() && Y.is_constant() && Y.typespec().is_float() &&
            (X.typespec().is_float() || X.typespec().is_triple())) {
        // if x and y are both constant, pre-compute x^y
        const float *x = (const float *) X.data();
        float y = *(const float *) Y.data();
        int ncomps = X.typespec().is_triple() ? 3 : 1;
        float result[3];
        for (int i = 0;  i < ncomps;  ++i)
            result[i] = safe_pow (x[i], y);
        int cind = rop.add_constant (X.typespec(), &result);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    if (Y.is_constant() && Y.typespec().is_float() &&
            *(const float *)Y.data() == 2.0f) {
        // Turn x^2 into x*x, even if x is not constant
        static ustring kmul("mul");
        op.reset (kmul, 2);
        rop.inst()->args()[op.firstarg()+2] = rop.inst()->args()[op.firstarg()+1];
        return 1;
    }

    return 0;
}



DECLFOLDER(constfold_triple)
{
    // Turn R=triple(a,b,c) into R=C if the components are all constants
    Opcode &op (rop.inst()->ops()[opnum]);
    DASSERT (op.nargs() == 4 || op.nargs() == 5); 
    bool using_space = (op.nargs() == 5);
    Symbol &R (*rop.inst()->argsymbol(op.firstarg()+0));
//    Symbol &Space (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1+using_space));
    Symbol &B (*rop.inst()->argsymbol(op.firstarg()+2+using_space));
    Symbol &C (*rop.inst()->argsymbol(op.firstarg()+3+using_space));
    if (A.is_constant() && A.typespec().is_float() &&
            B.is_constant() && C.is_constant() && !using_space) {
        DASSERT (A.typespec().is_float() && 
                 B.typespec().is_float() && C.typespec().is_float());
        float result[3];
        result[0] = *(const float *)A.data();
        result[1] = *(const float *)B.data();
        result[2] = *(const float *)C.data();
        int cind = rop.add_constant (R.typespec(), &result);
        rop.turn_into_assign (op, cind, "const fold");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_matrix)
{
    // Try to turn R=matrix(from,to) into R=const if it's an identity
    // transform or if the result is a non-time-varying matrix.
    Opcode &op (rop.inst()->ops()[opnum]);
    if (op.nargs() == 3) {
        Symbol &From (*rop.inst()->argsymbol(op.firstarg()+1));
        Symbol &To (*rop.inst()->argsymbol(op.firstarg()+2));
        if (! (From.is_constant() && From.typespec().is_string() &&
               To.is_constant() && To.typespec().is_string()))
            return 0;
        // OK, From and To are constant strings.
        ustring from = *(ustring *)From.data();
        ustring to = *(ustring *)To.data();
        ustring commonsyn = rop.inst()->shadingsys().commonspace_synonym();
        if (from == to || (from == Strings::common && to == commonsyn) ||
            (from == commonsyn && to == Strings::common)) {
            static Matrix44 ident (1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
            int cind = rop.add_constant (TypeDesc::TypeMatrix, &ident);
            rop.turn_into_assign (op, cind, "identity matrix");
            return 1;
        }
        // Shader and object spaces will vary from execution to execution,
        // so we can't optimize those away.
        if (from == Strings::shader || from == Strings::object ||
            to == Strings::shader || to == Strings::object)
            return 0;
        // But whatever spaces are left *may* be optimizable if they are
        // not time-varying.
        RendererServices *rs = rop.shadingsys().renderer();
        Matrix44 Mfrom, Mto;
        bool ok = true;
        if (from == Strings::common || from == commonsyn)
            Mfrom.makeIdentity ();
        else
            ok &= rs->get_matrix (Mfrom, from);
        if (to == Strings::common || to == commonsyn)
            Mto.makeIdentity ();
        else
            ok &= rs->get_inverse_matrix (Mto, to);
        if (ok) {
            // The from-to matrix is known and not time-varying, so just
            // turn it into a constant rather than calling getmatrix at
            // execution time.
            Matrix44 Mresult = Mfrom * Mto;
            int cind = rop.add_constant (TypeDesc::TypeMatrix, &Mresult);
            rop.turn_into_assign (op, cind, "const fold");
            return 1;
        }
    }
    return 0;
}



DECLFOLDER(constfold_getmatrix)
{
    // Try to turn R=getmatrix(from,to,M) into R=1,M=const if it's an
    // identity transform or if the result is a non-time-varying matrix.
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &From (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &To (*rop.inst()->argsymbol(op.firstarg()+2));
    if (! (From.is_constant() && To.is_constant()))
        return 0;
    // OK, From and To are constant strings.
    ustring from = *(ustring *)From.data();
    ustring to = *(ustring *)To.data();
    ustring commonsyn = rop.inst()->shadingsys().commonspace_synonym();
    if (from == to || (from == Strings::common && to == commonsyn) ||
        (from == commonsyn && to == Strings::common)) {
        static Matrix44 ident (1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
        int cind = rop.add_constant (TypeDesc::TypeMatrix, &ident);
        rop.turn_into_assign (op, cind, "identity matrix");
        return 1;
    }
    // Shader and object spaces will vary from execution to execution,
    // so we can't optimize those away.
    if (from == Strings::shader || from == Strings::object ||
        to == Strings::shader || to == Strings::object)
        return 0;
    // But whatever spaces are left *may* be optimizable if they are
    // not time-varying.
    RendererServices *rs = rop.shadingsys().renderer();
    Matrix44 Mfrom, Mto;
    bool ok = true;
    if (from == Strings::common || from == commonsyn)
        Mfrom.makeIdentity ();
    else
        ok &= rs->get_matrix (Mfrom, from);
    if (to == Strings::common || to == commonsyn)
        Mto.makeIdentity ();
    else
        ok &= rs->get_inverse_matrix (Mto, to);
    if (ok) {
        // The from-to matrix is known and not time-varying, so just
        // turn it into a constant rather than calling getmatrix at
        // execution time.
        int resultarg = rop.inst()->args()[op.firstarg()+0];
        int dataarg = rop.inst()->args()[op.firstarg()+3];
        // Make data the first argument
        rop.inst()->args()[op.firstarg()+0] = dataarg;
        // Now turn it into an assignment
        Matrix44 Mresult = Mfrom * Mto;
        int cind = rop.add_constant (TypeDesc::TypeMatrix, &Mresult);
        rop.turn_into_assign (op, cind, "known matrix");
        
        // Now insert a new instruction that assigns 1 to the
        // original return result of getmatrix.
        int one = 1;
        std::vector<int> args_to_add;
        args_to_add.push_back (resultarg);
        args_to_add.push_back (rop.add_constant (TypeDesc::TypeInt, &one));
        rop.insert_code (opnum, u_assign, args_to_add, true);
        Opcode &newop (rop.inst()->ops()[opnum]);
        newop.argwriteonly (0);
        newop.argread (1, true);
        newop.argwrite (1, false);
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_transform)
{
    // Try to turn identity transforms into assignments
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &M (*rop.inst()->argsymbol(op.firstarg()+1));
    if (op.nargs() == 3 && M.typespec().is_matrix() &&
          M.is_constant() && is_one(M)) {
        rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+2),
                              "transform by identity");
        return 1;
    }
    if (op.nargs() == 4) {
        Symbol &T (*rop.inst()->argsymbol(op.firstarg()+2));
        if (M.is_constant() && T.is_constant()) {
            DASSERT (M.typespec().is_string() && T.typespec().is_string());
            ustring from = *(ustring *)M.data();
            ustring to = *(ustring *)T.data();
            ustring syn = rop.shadingsys().commonspace_synonym();
            if (from == syn)
                from = Strings::common;
            if (to == syn)
                to = Strings::common;
            if (from == to) {
                rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+3),
                                      "transform by identity");
                return 1;
            }
        }
    }
    return 0;
}



DECLFOLDER(constfold_setmessage)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &Name (*rop.inst()->argsymbol(op.firstarg()+0));

    // Record that the inst set a message
    if (Name.is_constant()) {
        ASSERT (Name.typespec().is_string());
        rop.register_message (*(ustring *)Name.data());
    } else {
        rop.register_unknown_message ();
    }

    return 0;
}




DECLFOLDER(constfold_getmessage)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    int has_source = (op.nargs() == 4);
    if (has_source)
        return 0;    // Don't optimize away sourced getmessage
    Symbol &Name (*rop.inst()->argsymbol(op.firstarg()+1+(int)has_source));
    if (Name.is_constant()) {
        ASSERT (Name.typespec().is_string());
        if (! rop.message_possibly_set (*(ustring *)Name.data())) {
            // If the messages could not have been sent, get rid of the
            // getmessage op, leave the destination value alone, and
            // assign 0 to the returned status of getmessage.
            rop.turn_into_assign_zero (op, "impossible getmessage");
            return 1;
        }
    }
    return 0;
}




DECLFOLDER(constfold_gettextureinfo)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &Result (*rop.inst()->argsymbol(op.firstarg()+0));
    Symbol &Filename (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &Dataname (*rop.inst()->argsymbol(op.firstarg()+2));
    Symbol &Data (*rop.inst()->argsymbol(op.firstarg()+3));
    ASSERT (Result.typespec().is_int() && Filename.typespec().is_string() && 
            Dataname.typespec().is_string());

    if (Filename.is_constant() && Dataname.is_constant() &&
          ! Data.typespec().is_array()) {
        ustring filename = *(ustring *)Filename.data();
        ustring dataname = *(ustring *)Dataname.data();
        TypeDesc t = Data.typespec().simpletype();
        void *mydata = alloca (t.size ());
#if OPENIMAGEIO_VERSION >= 900  /* 0.9.0 */
        // FIXME(ptex) -- exclude folding of ptex, since these things
        // can vary per face.
        int result = rop.texturesys()->get_texture_info (filename, 0,
                                                         dataname, t, mydata);
#else
        int result = rop.texturesys()->get_texture_info (filename, dataname,
                                                         t, mydata);
#endif
        // Now we turn
        //       gettextureinfo result filename dataname data
        // into this for success:
        //       assign result 1
        //       assign data [retrieved values]
        // or, if it failed:
        //       assign result 0
        if (result) {
            int resultarg = rop.inst()->args()[op.firstarg()+0];
            int dataarg = rop.inst()->args()[op.firstarg()+3];
            // If not an array, turn the getattribute into an assignment
            // to data.  (Punt on arrays -- just let the gettextureinfo
            // happen as before.)
            if (! t.arraylen) {
                // Make data the first argument
                rop.inst()->args()[op.firstarg()+0] = dataarg;
                // Now turn it into an assignment
                int cind = rop.add_constant (Data.typespec(), mydata);
                rop.turn_into_assign (op, cind, "const fold");
            }

            // Now insert a new instruction that assigns 1 to the
            // original return result of gettextureinfo.
            int one = 1;
            std::vector<int> args_to_add;
            args_to_add.push_back (resultarg);
            args_to_add.push_back (rop.add_constant (TypeDesc::TypeInt, &one));
            rop.insert_code (opnum, u_assign, args_to_add, true);
            Opcode &newop (rop.inst()->ops()[opnum]);
            newop.argwriteonly (0);
            newop.argread (1, true);
            newop.argwrite (1, false);
            return 1;
        } else {
#if 1
            // Return without constant folding gettextureinfo -- because
            // we WANT the shader to fail and issue error messages at
            // the appropriate time.
            (void) rop.texturesys()->geterror (); // eat the error
            return 0;
#else
            rop.turn_into_assign_zero (op, "const fold");
            // If the get_texture_info failed, bubble error messages
            // from the texture system back up to the renderer.
            std::string err = rop.texturesys()->geterror();
            if (! err.empty())
                rop.shadingsys().error ("%s", err.c_str());
            return 1;
#endif
        }
    }
    return 0;
}



// texture -- we can eliminate a lot of superfluous setting of optional
// parameters to their default values.
DECLFOLDER(constfold_texture)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    // Symbol &Result = *rop.opargsym (op, 0);
    // Symbol &Filename = *rop.opargsym (op, 1);
    // Symbol &S = *rop.opargsym (op, 2);
    // Symbol &T = *rop.opargsym (op, 3);

    bool user_derivs = false;
    int first_optional_arg = 4;
    if (op.nargs() > 4 && rop.opargsym(op,4)->typespec().is_float()) {
        user_derivs = true;
        first_optional_arg = 8;
        DASSERT (rop.opargsym(op,5)->typespec().is_float());
        DASSERT (rop.opargsym(op,6)->typespec().is_float());
        DASSERT (rop.opargsym(op,7)->typespec().is_float());
    }

    TextureOpt opt;  // So we can check the defaults
    bool swidth_set = false, twidth_set = false, rwidth_set = false;
    bool sblur_set = false, tblur_set = false, rblur_set = false;
    bool swrap_set = false, twrap_set = false, rwrap_set = false;
    bool firstchannel_set = false, fill_set = false, interp_set = false;
    bool any_elided = false;
    for (int i = first_optional_arg;  i < op.nargs()-1;  i += 2) {
        Symbol &Name = *rop.opargsym (op, i);
        Symbol &Value = *rop.opargsym (op, i+1);
        DASSERT (Name.typespec().is_string());
        if (Name.is_constant()) {
            ustring name = *(ustring *)Name.data();
            bool elide = false;
            void *value = Value.is_constant() ? Value.data() : NULL;
            TypeDesc valuetype = Value.typespec().simpletype();

// Keep from repeating the same tedious code for {s,t,r, }{width,blur,wrap}
#define CHECK(field,ctype,osltype)                              \
            if (name == Strings::field && ! field##_set) {      \
                if (value && osltype == TypeDesc::FLOAT &&      \
                    valuetype == TypeDesc::INT &&               \
                    *(int *)value == opt.field)                 \
                    elide = true;                               \
                else if (value && valuetype == osltype &&       \
                      *(ctype *)value == opt.field)             \
                    elide = true;                               \
                else                                            \
                    field##_set = true;                         \
            }
#define CHECK_str(field,ctype,osltype)                                  \
            CHECK (s##field,ctype,osltype)                              \
            else CHECK (t##field,ctype,osltype)                         \
            else CHECK (r##field,ctype,osltype)                         \
            else if (name == Strings::field && !s##field##_set &&       \
                     ! t##field##_set && ! r##field##_set &&            \
                     valuetype == osltype) {                            \
                ctype *v = (ctype *)value;                              \
                if (v && *v == opt.s##field && *v == opt.t##field       \
                    && *v == opt.r##field)                              \
                    elide = true;                                       \
                else {                                                  \
                    s##field##_set = true;                              \
                    t##field##_set = true;                              \
                    r##field##_set = true;                              \
                }                                                       \
            }
            
            CHECK_str (width, float, TypeDesc::FLOAT)
            else CHECK_str (blur, float, TypeDesc::FLOAT)
            else CHECK_str (wrap, ustring, TypeDesc::STRING)
            else CHECK (firstchannel, int, TypeDesc::INT)
            else CHECK (fill, float, TypeDesc::FLOAT)
#undef CHECK_STR
#undef CHECK

            // Cases that don't fit the pattern
            else if (name == Strings::interp && !interp_set) {
                if (value && valuetype == TypeDesc::STRING &&
                    tex_interp_to_code(*(ustring *)value) == opt.interpmode)
                    elide = true;
                else
                    interp_set = true;
            }

            if (elide) {
                // Just turn the param name into empty string and it will
                // be skipped.
                ustring empty;
                int cind = rop.add_constant (TypeDesc::TypeString, &empty);
                rop.inst()->args()[op.firstarg()+i] = cind;
                rop.inst()->args()[op.firstarg()+i+1] = cind;
                any_elided = true;
            }
        }
    }
    return any_elided;
}




DECLFOLDER(constfold_functioncall)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    // Make a "functioncall" block disappear if the only non-nop statements
    // inside it is 'return'.
    bool has_return = false;
    bool has_anything_else = false;
    for (int i = opnum+1, e = op.jump(0);  i < e;  ++i) {
        Opcode &op (rop.inst()->ops()[i]);
        if (op.opname() == u_return)
            has_return = true;
        else if (op.opname() != u_nop)
            has_anything_else = true;
    }
    int changed = 0;
    if (! has_anything_else) {
        // Possibly due to optimizations, there's nothing in the
        // function body but the return.  So just eliminate the whole
        // block of ops.
        for (int i = opnum, e = op.jump(0);  i < e;  ++i) {
            if (rop.inst()->ops()[i].opname() != u_nop) {
                rop.turn_into_nop (rop.inst()->ops()[i], "empty function");
                ++changed;
            }
        }
    } else if (! has_return) {
        // The function is just a straight-up execution, no return
        // statement, so kill the "function" op.
        rop.turn_into_nop (op, "'function' not necessary");
        ++changed;
    }
    
    return changed;
}




DECLFOLDER(constfold_useparam)
{
    // Just eliminate useparam (from shaders compiled with old oslc)
    Opcode &op (rop.inst()->ops()[opnum]);
    rop.turn_into_nop (op);
    return 1;
}



DECLFOLDER(constfold_assign)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol *B (rop.inst()->argsymbol(op.firstarg()+1));
    int Aalias = rop.block_alias (rop.inst()->arg(op.firstarg()+0));
    Symbol *AA = rop.inst()->symbol(Aalias);
    // N.B. symbol() returns NULL if alias is < 0

    if (B->is_constant() && AA && AA->is_constant()) {
        // Try to turn A=C into nop if A already is C
        if (AA->typespec().is_int() && B->typespec().is_int()) {
            if (*(int *)AA->data() == *(int *)B->data()) {
                rop.turn_into_nop (op, "reassignment of current value");
                return 1;
            }
        } else if (AA->typespec().is_float() && B->typespec().is_float()) {
            if (*(float *)AA->data() == *(float *)B->data()) {
                rop.turn_into_nop (op, "reassignment of current value");
                return 1;
            }
        } else if (AA->typespec().is_float() && B->typespec().is_int()) {
            if (*(float *)AA->data() == *(int *)B->data()) {
                rop.turn_into_nop (op, "reassignment of current value");
                return 1;
            }
        } else if (AA->typespec().is_triple() && B->typespec().is_triple()) {
            if (*(Vec3 *)AA->data() == *(Vec3 *)B->data()) {
                rop.turn_into_nop (op, "reassignment of current value");
                return 1;
            }
        } else if (AA->typespec().is_triple() && B->typespec().is_float()) {
            float b = *(float *)B->data();
            if (*(Vec3 *)AA->data() == Vec3(b,b,b)) {
                rop.turn_into_nop (op, "reassignment of current value");
                return 1;
            }
        }
    }
    return 0;
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
                        int cind = add_constant (s->typespec(), srcsym->data());
                        // Alias this symbol to the new const
                        global_alias (i, cind);
                        make_param_use_instanceval (s, "- upstream layer sets it to a constant");
                        replace_param_value (s, srcsym->data());
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
                           Strutil::format("remove stale value assignment to %s, reassigned on op %d", opargsym(uselessop,0)->name().c_str(), opnum).c_str());
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
RuntimeOptimizer::replace_param_value (Symbol *R, const void *newdata)
{
    ASSERT (R->symtype() == SymTypeParam || R->symtype() == SymTypeOutputParam);
    TypeDesc Rtype = R->typespec().simpletype();
    void *Rdefault = NULL;
    DASSERT (R->dataoffset() >= 0);
#ifdef DEBUG
    int nvals = int(Rtype.aggregate * Rtype.numelements());
#endif
    if (Rtype.basetype == TypeDesc::FLOAT) {
        Rdefault = &inst()->m_fparams[R->dataoffset()];
        DASSERT ((R->dataoffset()+nvals) <= (int)inst()->m_fparams.size());
    }
    else if (Rtype.basetype == TypeDesc::INT) {
        Rdefault = &inst()->m_iparams[R->dataoffset()];
        DASSERT ((R->dataoffset()+nvals) <= (int)inst()->m_iparams.size());
    }
    else if (Rtype.basetype == TypeDesc::STRING) {
        Rdefault = &inst()->m_sparams[R->dataoffset()];
        DASSERT ((R->dataoffset()+nvals) <= (int)inst()->m_sparams.size());
    } else {
        ASSERT (0 && "replace_param_value: unexpected type");
    }
    DASSERT (Rdefault != NULL);
    memcpy (Rdefault, newdata, Rtype.size());
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

    // Point the symbol's data pointer to its param default and make it
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
    R->step (0);

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
            replace_param_value (R, A->data());
            turn_into_nop (op, Strutil::format("oparam %s never subsequently read or connected", R->name().c_str()).c_str());
            return true;
        }
    }

    // If the output param will neither be read later in the shader nor
    // connected to a downstream layer, then we don't really need this
    // assignment at all.
    if (unread_after(R,opnum)) {
        turn_into_nop (op, Strutil::format("oparam %s never subsequently read or connected", R->name().c_str()).c_str());
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




// Predicate to test if the connection's destination is never used
struct ConnectionDestNeverUsed
{
    ConnectionDestNeverUsed (const ShaderInstance *inst) : m_inst(inst) { }
    bool operator() (const Connection &c) {
        return ! m_inst->symbol(c.dst.param)->everused();
    }
private:
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



void
RuntimeOptimizer::optimize_instance ()
{
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
    // ever get into an infinite loop from an unforseen cycle.  where we
    // end up inadvertently transforming A => B => A => etc.
    int totalchanged = 0;
    int reallydone = 0;   // Force one pass after we think we're done
    for (int pass = 0;  pass < 10;  ++pass) {

        // Once we've made one pass (and therefore called
        // mark_outgoing_connections), we may notice that the layer is
        // unused, and therefore can stop doing work to optimize it.
        if (pass != 0 && inst()->unused())
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
        for (int opnum = 0;  opnum < (int)num_ops;  ++opnum) {
            // Before getting a reference to this op, be sure that a space
            // is reserved at the end in case a folding routine inserts an
            // op.  That ensures that the reference won't be invalid.
            inst()->ops().reserve (num_ops+1);
            Opcode &op (inst()->ops()[opnum]);

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

            // Make sure there's room for at least one more symbol, so that
            // we can add a const if we need to, without worrying about the
            // addresses of symbols changing when we add a new one below.
            make_symbol_room (1);

            // For various ops that we know how to effectively
            // constant-fold, dispatch to the appropriate routine.
            if (optimize() >= 2 && m_opt_constant_fold) {
                const OpDescriptor *opd = m_shadingsys.op_descriptor (op.opname());
                if (opd && opd->folder) {
                    changed += (*opd->folder) (*this, opnum);
                    // Re-check num_ops in case the folder inserted something
                    num_ops = inst()->ops().size();
                }
            }

            // Clear local block aliases for any args that were written
            // by this op
            for (int i = 0, e = op.nargs();  i < e;  ++i)
                if (op.argwrite(i))
                    block_unalias (inst()->arg(op.firstarg()+i));

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

        totalchanged += changed;
        // info ("Pass %d, changed %d\n", pass, changed);

        // Now that we've rewritten the code, we need to re-track the
        // variable lifetimes.
        track_variable_lifetimes ();

        // Recompute which of our params have downstream connections.
        mark_outgoing_connections ();

        // Elide unconnected parameters that are never read.
        FOREACH_PARAM (Symbol &s, inst()) {
            if (!s.connected_down() && ! s.everread()) {
                changed += turn_into_nop (s.initbegin(), s.initend(),
                                          "remove init ops of unread param");
                s.set_initrange ();
                s.clear_rw ();
            }
        }

        // FIXME -- we should re-evaluate whether writes_globals() is still
        // true for this layer.

        // If nothing changed, we're done optimizing.  But wait, it may be
        // that after re-tracking variable lifetimes, we can notice new
        // optimizations!  So force another pass, then we're really done.
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
                       Strutil::format("eliminate layer %s with no outward connections", inst()->layername().c_str()).c_str());
        BOOST_FOREACH (Symbol &s, inst()->symbols())
            s.clear_rw ();
    }

    // Erase this layer's incoming connections and init ops for params
    // it no longer uses
    erase_if (inst()->connections(), ConnectionDestNeverUsed(inst()));

    // Clear init ops of params that aren't used.
    FOREACH_PARAM (Symbol &s, inst()) {
        if (s.symtype() == SymTypeParam && ! s.everused() &&
                s.initbegin() < s.initend()) {
            turn_into_nop (s.initbegin(), s.initend(),
                           "remove init ops of unused param");
            s.set_initrange (0, 0);
        }
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

    // First, just count how many we need and set up the mapping
    BOOST_FOREACH (const Symbol &s, inst()->symbols()) {
        symbol_remap.push_back (total_syms);
        if (s.everused() ||
            (s.symtype() == SymTypeParam && s.connected_down()) ||
              s.symtype() == SymTypeOutputParam)
            ++total_syms;
    }

    // Now make a new table of the right (new) size, and copy the used syms
    new_symbols.reserve (total_syms);
    BOOST_FOREACH (const Symbol &s, inst()->symbols()) {
        if (s.everused() ||
            (s.symtype() == SymTypeParam && s.connected_down()) ||
              s.symtype() == SymTypeOutputParam)
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

    for (int layer = nlayers-1;  layer >= 0;  --layer) {
        set_inst (layer);
        track_variable_dependencies ();

        // For our parameters that require derivatives, mark their
        // upstream connections as also needing derivatives.
        bool any = false;
        BOOST_FOREACH (Connection &c, inst()->m_connections) {
            if (inst()->symbol(c.dst.param)->has_derivs()) {
                Symbol *source = m_group[c.srclayer]->symbol(c.src.param);
                if (! source->typespec().is_closure_based() &&
                    source->typespec().elementtype().is_floatbased()) {
                    source->has_derivs (true);
                    any = true;
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
        }

    }
    {
        // adjust memory stats
        ShadingSystemImpl &ss (shadingsys());
        spin_lock lock (ss.m_stat_mutex);
        ss.m_stat_mem_inst_syms -= symmem;
        ss.m_stat_mem_inst -= symmem;
        ss.m_stat_memory -= symmem;
        ss.m_stat_preopt_syms += old_nsyms;
        ss.m_stat_preopt_ops += old_nops;
        ss.m_stat_postopt_syms += new_nsyms;
        ss.m_stat_postopt_ops += new_nops;
    }

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
    m_shadingsys.info ("    (%1.2fs = %1.2f spc, %1.2f lllock, %1.2f llset, %1.2f ir, %1.2f opt, %1.2f jit)",
                       m_stat_total_llvm_time+m_stat_specialization_time,
                       m_stat_specialization_time, 
                       m_stat_opt_locking_time, m_stat_llvm_setup_time,
                       m_stat_llvm_irgen_time, m_stat_llvm_opt_time,
                       m_stat_llvm_jit_time);
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

    RuntimeOptimizer rop (*this, group);
    rop.optimize_group ();

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


}; // namespace pvt
}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
