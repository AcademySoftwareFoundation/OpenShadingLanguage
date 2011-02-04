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

#include <OpenImageIO/hash.h>
#include <OpenImageIO/timer.h>

#include "oslexec_pvt.h"
#include "oslops.h"
#include "runtimeoptimize.h"
#include "../liboslcomp/oslcomp_pvt.h"

#include "llvm_headers.h"
using namespace OSL;
using namespace OSL::pvt;


// names of ops we'll be using frequently
static ustring u_nop    ("nop"),
               u_assign ("assign"),
               u_add    ("add"),
               u_sub    ("sub"),
               u_if     ("if"),
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



void
RuntimeOptimizer::set_inst (int newlayer)
{
    m_layer = newlayer;
    m_inst = m_group[m_layer];
    ASSERT (m_inst != NULL);
    m_all_consts.clear ();
    m_symbol_aliases.clear ();
    m_block_aliases.clear ();
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
RuntimeOptimizer::turn_into_assign (Opcode &op, int newarg)
{
    op.reset (u_assign, 2);
    inst()->args()[op.firstarg()+1] = newarg;
    op.argwriteonly (0);
    op.argread (1, true);
    op.argwrite (1, false);
}



// Turn the current op into a simple assignment to zero (of the first arg).
void
RuntimeOptimizer::turn_into_assign_zero (Opcode &op)
{
    static float zero[16] = { 0, 0, 0, 0,  0, 0, 0, 0,
                              0, 0, 0, 0,  0, 0, 0, 0 };
    Symbol &R (*(inst()->argsymbol(op.firstarg()+0)));
    int cind = add_constant (R.typespec(), &zero);
    turn_into_assign (op, cind);
}



// Turn the current op into a simple assignment to one (of the first arg).
void
RuntimeOptimizer::turn_into_assign_one (Opcode &op)
{
    Symbol &R (*(inst()->argsymbol(op.firstarg()+0)));
    if (R.typespec().is_int()) {
        int one = 1;
        int cind = add_constant (R.typespec(), &one);
        turn_into_assign (op, cind);
    } else {
        ASSERT (R.typespec().is_triple() || R.typespec().is_float());
        static float one[3] = { 1, 1, 1 };
        int cind = add_constant (R.typespec(), &one);
        turn_into_assign (op, cind);
    }
}



// Turn the op into a no-op
void
RuntimeOptimizer::turn_into_nop (Opcode &op)
{
    op.reset (u_nop, 0);
}



/// Return true if the op is guaranteed to completely overwrite all of its
/// writable arguments and doesn't need their prior vaues at all.  If the
/// op uses the prior values of any writeable arguments, or in any way 
/// preserves their original values, return false.
/// Example: ADD -> true, since 'add r a b' completely replaces the old
/// value of r and doesn't care what it was before (other than
/// coincidentally, if a or b is also r).
/// Example: COMPASSIGN -> false, since 'compassign r i x' only changes
/// one component of r.
#if 0
static bool
fully_writing_op (const Opcode &op)
{
    // Just do a table comparison, since there are so few ops that don't
    // have this property.
    static OpImpl exceptions[] = {
        // array and component routines only partially override their result
        OP_aassign, OP_compassign, 
        // the "get" routines don't touch their data arg if the named
        // item isn't found
        OP_getattribute, OP_getmessage, OP_gettextureinfo,
        // Anything else?
        NULL
    };
    for (int i = 0; exceptions[i];  ++i)
        if (op.implementation() == exceptions[i])
            return false;
    return true;
}
#endif



/// Insert instruction 'opname' with arguments 'args_to_add' into the 
/// code at instruction 'opnum'.  The existing code and concatenated 
/// argument lists can be found in code and opargs, respectively, and
/// allsyms contains pointers to all symbols.  mainstart is a reference
/// to the address where the 'main' shader begins, and may be modified
/// if the new instruction is inserted before that point.
void
RuntimeOptimizer::insert_code (int opnum, ustring opname, const std::vector<int> &args_to_add)
{
    OpcodeVec &code (inst()->ops());
    std::vector<int> &opargs (inst()->args());
    ustring method = (opnum < (int)code.size()) ? code[opnum].method() : OSLCompilerImpl::main_method_name();
    Opcode op (opname, method, opargs.size(), args_to_add.size());
    off_t oldcodesize = vectorbytes(code);
    off_t oldargsize = vectorbytes(opargs);
    code.insert (code.begin()+opnum, op);
    opargs.insert (opargs.end(), args_to_add.begin(), args_to_add.end());
    {
        // Remember that they're already swapped
        off_t opmem = vectorbytes(code) - oldcodesize;
        off_t argmem = vectorbytes(opargs) - oldargsize;
        // adjust memory stats
        ShadingSystemImpl &ss (shadingsys());
        spin_lock lock (ss.m_stat_mutex);
        ss.m_stat_mem_inst_ops += opmem;
        ss.m_stat_mem_inst_args += argmem;
        ss.m_stat_mem_inst += (opmem+argmem);
        ss.m_stat_memory += (opmem+argmem);
    }
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
}



/// Insert a 'useparam' instruction in front of instruction 'opnum', to
/// reference the symbols in 'params'.
void
RuntimeOptimizer::insert_useparam (size_t opnum,
                                   std::vector<int> &params_to_use)
{
    OpcodeVec &code (inst()->ops());
    static ustring useparam("useparam");
    insert_code (opnum, useparam, params_to_use);

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
        if (op.opname() == "useparam")
            continue;  // skip useparam ops themselves, if we hit one
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
            m_in_conditional.insert (m_in_conditional.begin()+opnum,
                                     m_in_conditional[opnum]);
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



DECLFOLDER(constfold_add)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &A (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &B (*rop.inst()->argsymbol(op.firstarg()+2));
    if (A.is_constant()) {
        if (is_zero(A)) {
            // R = 0 + B  =>   R = B
            rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+2));
            return 1;
        }
    }
    if (B.is_constant()) {
        if (is_zero(B)) {
            // R = A + 0   =>   R = A
            rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+1));
            return 1;
        }
    }
    if (A.is_constant() && B.is_constant()) {
        if (A.typespec().is_int() && B.typespec().is_int()) {
            int result = *(int *)A.data() + *(int *)B.data();
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind);
            return 1;
        } else if (A.typespec().is_float() && B.typespec().is_float()) {
            float result = *(float *)A.data() + *(float *)B.data();
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind);
            return 1;
        } else if (A.typespec().is_triple() && B.typespec().is_triple()) {
            Vec3 result = *(Vec3 *)A.data() + *(Vec3 *)B.data();
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind);
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
            rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+1));
            return 1;
        }
    }
    // R = A - B, if both are constants, =>  R = C
    if (A.is_constant() && B.is_constant()) {
        if (A.typespec().is_int() && B.typespec().is_int()) {
            int result = *(int *)A.data() - *(int *)B.data();
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind);
            return 1;
        } else if (A.typespec().is_float() && B.typespec().is_float()) {
            float result = *(float *)A.data() - *(float *)B.data();
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind);
            return 1;
        } else if (A.typespec().is_triple() && B.typespec().is_triple()) {
            Vec3 result = *(Vec3 *)A.data() - *(Vec3 *)B.data();
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind);
            return 1;
        }
    }
    // R = A - A  =>  R = 0    even if not constant!
    if (&A == &B) {
        rop.turn_into_assign_zero (op);
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
            rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+2));
            return 1;
        }
        if (is_zero(A)) {
            // R = 0 * B  =>   R = 0
            rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+1));
            return 1;
        }
    }
    if (B.is_constant()) {
        if (is_one(B)) {
            // R = A * 1   =>   R = A
            rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+1));
            return 1;
        }
        if (is_zero(B)) {
            // R = A * 0   =>   R = 0
            rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+2));
            return 1;
        }
    }
    if (A.is_constant() && B.is_constant()) {
        if (A.typespec().is_int() && B.typespec().is_int()) {
            int result = *(int *)A.data() * *(int *)B.data();
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind);
            return 1;
        } else if (A.typespec().is_float() && B.typespec().is_float()) {
            float result = (*(float *)A.data()) * (*(float *)B.data());
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind);
            return 1;
        } else if (A.typespec().is_triple() && B.typespec().is_triple()) {
            Vec3 result = (*(Vec3 *)A.data()) * (*(Vec3 *)B.data());
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind);
            return 1;
        } else if (A.typespec().is_triple() && B.typespec().is_float()) {
            Vec3 result = (*(Vec3 *)A.data()) * (*(float *)B.data());
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind);
            return 1;
        } else if (A.typespec().is_float() && B.typespec().is_triple()) {
            Vec3 result = (*(float *)A.data()) * (*(Vec3 *)B.data());
            int cind = rop.add_constant (B.typespec(), &result);
            rop.turn_into_assign (op, cind);
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
            rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+1));
            return 1;
        }
        if (is_zero(B) && (B.typespec().is_float() ||
                           B.typespec().is_triple() || B.typespec().is_int())) {
            // R = A / 0   =>   R = 0      because of OSL div by zero rule
            rop.turn_into_assign_zero (op);
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
            rop.turn_into_assign (op, cind);
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
        rop.turn_into_assign_zero (op);
        return 1;
    }

    // dot(const,const) -> const
    if (A.is_constant() && B.is_constant()) {
        DASSERT (A.typespec().is_triple() && B.typespec().is_triple());
        float result = (*(Vec3 *)A.data()).dot (*(Vec3 *)B.data());
        int cind = rop.add_constant (TypeDesc::TypeFloat, &result);
        rop.turn_into_assign (op, cind);
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
            rop.turn_into_assign (op, cind);
            return 1;
        } else if (A.typespec().is_float()) {
            float result =  - *(float *)A.data();
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind);
            return 1;
        } else if (A.typespec().is_triple()) {
            Vec3 result = - *(Vec3 *)A.data();
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind);
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
            rop.turn_into_assign (op, cind);
            return 1;
        } else if (A.typespec().is_float()) {
            float result =  std::abs(*(float *)A.data());
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind);
            return 1;
        } else if (A.typespec().is_triple()) {
            Vec3 result = *(Vec3 *)A.data();
            result.x = std::abs(result.x);
            result.y = std::abs(result.y);
            result.z = std::abs(result.z);
            int cind = rop.add_constant (A.typespec(), &result);
            rop.turn_into_assign (op, cind);
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
        rop.turn_into_assign (op, cind);
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
        rop.turn_into_assign (op, cind);
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
        rop.turn_into_assign (op, cind);
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
        rop.turn_into_assign (op, cind);
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
        // Turn the 'leq R A B' into 'assign R X' where X is 0 or 1.
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
        rop.turn_into_assign (op, cind);
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
        rop.turn_into_assign (op, cind);
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
        rop.turn_into_assign (op, cind);
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
        rop.turn_into_assign (op, cind);
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
            for (int i = op.jump(0);  i < op.jump(1);  ++i, ++changed)
                rop.turn_into_nop (rop.inst()->ops()[i]);
            rop.turn_into_nop (op);
            return changed+1;
        } else if (result == 0) {
            for (int i = opnum+1;  i < op.jump(0);  ++i, ++changed)
                rop.turn_into_nop (rop.inst()->ops()[i]);
            rop.turn_into_nop (op);
            return changed+1;
        }
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
        DASSERT (index < A.typespec().arraylength());
        int cind = rop.add_constant (elemtype,
                        (char *)A.data() + index*elemtype.simpletype().size());
        rop.turn_into_assign (op, cind);
        return 1;
    }
    // Even if the index isn't constant, we still know the answer if all
    // the array elements are equal!
    if (A.is_constant() && array_all_elements_equal(A)) {
        TypeSpec elemtype = A.typespec().elementtype();
        ASSERT (equivalent(elemtype, R.typespec()));
        int cind = rop.add_constant (elemtype, (char *)A.data());
        rop.turn_into_assign (op, cind);
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_compassign)
{
    // Component assignment
    Opcode &op (rop.inst()->ops()[opnum]);
    // Symbol *A (rop.inst()->argsymbol(op.firstarg()+0));
    Symbol *I (rop.inst()->argsymbol(op.firstarg()+1));
    Symbol *C (rop.inst()->argsymbol(op.firstarg()+2));
    int Aalias = rop.block_alias (rop.inst()->arg(op.firstarg()+0));
    Symbol *AA = rop.inst()->symbol(Aalias);
    // N.B. symbol returns NULL if Aalias is < 0

    if (I->is_constant() && C->is_constant() && AA && AA->is_constant()) {
        // Try to turn A[I]=C into nop if A[I] already is C
        if (AA->typespec().is_int() && C->typespec().is_int()) {
            int *aa = (int *)AA->data();
            int i = *(int *)I->data();
            int c = *(int *)C->data();
            if (aa[i] == c) {
                rop.turn_into_nop (op);
                return 1;
            }
        } else if (AA->typespec().is_float() && C->typespec().is_float()) {
            float *aa = (float *)AA->data();
            int i = *(int *)I->data();
            float c = *(float *)C->data();
            if (aa[i] == c) {
                rop.turn_into_nop (op);
                return 1;
            }
        } else if (AA->typespec().is_triple() && C->typespec().is_triple()) {
            Vec3 *aa = (Vec3 *)AA->data();
            int i = *(int *)I->data();
            Vec3 c = *(Vec3 *)C->data();
            if (aa[i] == c) {
                rop.turn_into_nop (op);
                return 1;
            }
        }
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
        int cind = rop.add_constant (TypeDesc::TypeFloat, (float *)A.data() + index);
        rop.turn_into_assign (op, cind);
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
        rop.turn_into_assign (op, cind);
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
        rop.turn_into_assign (op, cind);
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
    rop.turn_into_assign (op, cind);
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
            rop.turn_into_assign (op, cind);
            return 1;
        }
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
        rop.turn_into_assign (op, cind);
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
        rop.turn_into_assign (op, cind);
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
        rop.turn_into_assign (op, cind);
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
        rop.turn_into_assign (op, cind);
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
        rop.turn_into_assign (op, cind);
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
        rop.turn_into_assign (op, cind);
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
        rop.turn_into_assign_one (op);
        return 1;
    }
    if (Y.is_constant() && is_one(Y)) {
        // x^1 == x
        rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+1));
        return 1;
    }
    if (X.is_constant() && is_zero(X)) {
        // 0^y == 0
        rop.turn_into_assign_zero (op);
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
        rop.turn_into_assign (op, cind);
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
        rop.turn_into_assign (op, cind);
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_matrix)
{
    // Try to turn R=matrix(from,to) into R=1 if it's an identity transform
    Opcode &op (rop.inst()->ops()[opnum]);
    if (op.nargs() == 3) {
        Symbol &From (*rop.inst()->argsymbol(op.firstarg()+1));
        Symbol &To (*rop.inst()->argsymbol(op.firstarg()+2));
        if (From.is_constant() && From.typespec().is_string() &&
            To.is_constant() && To.typespec().is_string()) {
            ustring from = *(ustring *)From.data();
            ustring to = *(ustring *)To.data();
            ustring commonsyn = rop.inst()->shadingsys().commonspace_synonym();
            if (from == to || (from == Strings::common && to == commonsyn) ||
                              (from == commonsyn && to == Strings::common)) {
                static Matrix44 ident (1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
                int cind = rop.add_constant (TypeDesc::TypeMatrix, &ident);
                rop.turn_into_assign (op, cind);
                return 1;
            }
        }
    }
    return 0;
}



DECLFOLDER(constfold_transform)
{
    // Try to turn R=transform(M,P) into R=P if it's an identity transform
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &M (*rop.inst()->argsymbol(op.firstarg()+1));
    Symbol &P (*rop.inst()->argsymbol(op.firstarg()+2));
    if (op.nargs() == 3 && M.typespec().is_matrix() &&
          M.is_constant() && is_one(M)) {
        ASSERT (P.typespec().is_triple());
        rop.turn_into_assign (op, rop.inst()->arg(op.firstarg()+2));
        return 1;
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
    Symbol &Name (*rop.inst()->argsymbol(op.firstarg()+1));
    if (Name.is_constant()) {
        ASSERT (Name.typespec().is_string());
        if (! rop.message_possibly_set (*(ustring *)Name.data())) {
            // If the messages could not have been sent, get rid of the
            // getmessage op, leave the destination value alone, and
            // assign 0 to the returned status of getmessage.
            rop.turn_into_assign_zero (op);
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

    if (Filename.is_constant()) {
        ustring filename = *(ustring *)Filename.data();

        if (Dataname.is_constant()) {
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
                    rop.turn_into_assign (op, cind);
                }

                // Now insert a new instruction that assigns 1 to the
                // original return result of gettextureinfo.
                int one = 1;
                std::vector<int> args_to_add;
                args_to_add.push_back (resultarg);
                args_to_add.push_back (rop.add_constant (TypeDesc::TypeInt, &one));
                rop.insert_code (opnum, u_assign, args_to_add);
                Opcode &newop (rop.inst()->ops()[opnum]);
                newop.argwriteonly (0);
                newop.argread (1, true);
                newop.argwrite (1, false);
                return 1;
            } else {
                rop.turn_into_assign_zero (op);
                return 1;
            }
        }
    }
    return 0;
}




DECLFOLDER(constfold_useparam)
{
    // Just eliminate useparam (from shaders compiled with old oslc)
    Opcode &op (rop.inst()->ops()[opnum]);
    rop.turn_into_nop (op);
    return 1;
}


#ifdef OIIO_HAVE_BOOST_UNORDERED_MAP
typedef boost::unordered_map<ustring, OpFolder, ustringHash> FolderTable;
#else
typedef hash_map<ustring, OpFolder, ustringHash> FolderTable;
#endif

static FolderTable folder_table;

void
initialize_folder_table ()
{
    static spin_mutex folder_table_mutex;
    static bool folder_table_initialized = false;
    spin_lock lock (folder_table_mutex);
    if (folder_table_initialized)
        return;   // already initialized
#define INIT2(name,folder) folder_table[ustring(#name)] = folder
#define INIT(name) folder_table[ustring(#name)] = constfold_##name;

    INIT (add);    INIT (sub);
    INIT (mul);    INIT (div);
    INIT (dot);
    INIT (neg);    INIT (abs);
    INIT (eq);     INIT (neq);
    INIT (le);     INIT (ge);
    INIT (lt);     INIT (gt);
    INIT (or);     INIT (and);
    INIT (if);
    INIT (compassign);
    INIT (compref);
    INIT (aref);
    INIT (strlen);
    INIT (endswith);
    INIT (concat);
    INIT (format);
    INIT (clamp);
    INIT (min);
    INIT (max);
    INIT (sqrt);
    INIT (pow);
    INIT (floor);
    INIT (ceil);
    INIT2 (color, constfold_triple);
    INIT2 (point, constfold_triple);
    INIT2 (normal, constfold_triple);
    INIT2 (vector, constfold_triple);
    INIT (matrix);
    INIT2 (transform, constfold_transform);
    INIT2 (transformv, constfold_transform);
    INIT2 (transformn, constfold_transform);
    INIT (setmessage);
    INIT (getmessage);
    INIT (gettextureinfo);
    INIT (useparam);
//    INIT (assign);  N.B. do not include here -- we want this run AFTER
//                    all other constant folding is done, since many of
//                    them turn other statements into assignments.
#undef INIT
#undef INIT2

    folder_table_initialized = true;
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
                rop.turn_into_nop (op);
                return 1;
            }
        } else if (AA->typespec().is_float() && B->typespec().is_float()) {
            if (*(float *)AA->data() == *(float *)B->data()) {
                rop.turn_into_nop (op);
                return 1;
            }
        } else if (AA->typespec().is_float() && B->typespec().is_int()) {
            if (*(float *)AA->data() == *(int *)B->data()) {
                rop.turn_into_nop (op);
                return 1;
            }
        } else if (AA->typespec().is_triple() && B->typespec().is_triple()) {
            if (*(Vec3 *)AA->data() == *(Vec3 *)B->data()) {
                rop.turn_into_nop (op);
                return 1;
            }
        } else if (AA->typespec().is_triple() && B->typespec().is_float()) {
            float b = *(float *)B->data();
            if (*(Vec3 *)AA->data() == Vec3(b,b,b)) {
                rop.turn_into_nop (op);
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
        if (// it's a paramter that can't change with the geom
            s->symtype() == SymTypeParam && s->lockgeom() &&
            // and it's NOT a default val that needs to run init ops
            !(s->valuesource() == Symbol::DefaultVal && s->has_init_ops()) &&
            // and it not a structure or closure variable...
            !s->typespec().is_structure() && !s->typespec().is_closure())
        {
            // We can turn it into a constant if there's no connection,
            // OR if the connection is itself a constant

            if (s->valuesource() == Symbol::ConnectedVal) {
                // It's connected to an earlier layer.  But see if the
                // output var of the upstream shader is effectively constant.
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
                                make_param_use_instanceval (s);
                                replace_param_value (s, srcsym->data());
                                break;
                        }
                    }
                }
            } else {
                // Not a connected value -- make a new const using the
                // param's instance values
                make_symbol_room (1);
                s = inst()->symbol(i);  // In case make_symbol_room changed ptrs
                int cind = add_constant (s->typespec(), s->data());
                // Alias this symbol to the new const
                global_alias (i, cind);
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
    for (int i = 0;  i < (int)code.size();  ++i) {
        if (code[i].jump(0) >= 0)
            std::fill (m_in_conditional.begin()+i,
                       m_in_conditional.begin()+code[i].farthest_jump(), true);
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
        // Anyplace that's the target of a jump instruction starts a basic block
        for (int j = 0;  j < (int)Opcode::max_jumps;  ++j) {
            if (code[opnum].jump(j) >= 0)
                block_begin[code[opnum].jump(j)] = true;
            else
                break;
        }
        // The first instruction in a conditional or loop (which is not
        // itself a jump target) also begins a basic block.  If the op has
        // any jump targets at all, it must be a conditional or loop.
        if (code[opnum].jump(0) >= 0)
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

    if (! A->is_constant() || R->typespec().is_closure())
        return false;   // we don't handle those cases

    // turn 'R_float = A_int_const' into a float const assignment
    if (A->typespec().is_int() && R->typespec().is_float()) {
        float result = *(int *)A->data();
        int cind = add_constant (R->typespec(), &result);
        turn_into_assign (op, cind);
        return true;
    }

    // turn 'R_int = A_float_const' into an int const assignment
    if (A->typespec().is_float() && R->typespec().is_int()) {
        int result = (int) *(float *)A->data();
        int cind = add_constant (R->typespec(), &result);
        turn_into_assign (op, cind);
        return true;
    }

    // turn 'R_triple = A_int_const' into a float const assignment
    if (A->typespec().is_int() && R->typespec().is_triple()) {
        float f = *(int *)A->data();
        Vec3 result (f, f, f);
        int cind = add_constant (R->typespec(), &result);
        turn_into_assign (op, cind);
        return true;
    }

    // turn 'R_triple = A_float_const' into a triple const assignment
    if (A->typespec().is_float() && R->typespec().is_triple()) {
        float f = *(float *)A->data();
        Vec3 result (f, f, f);
        int cind = add_constant (R->typespec(), &result);
        turn_into_assign (op, cind);
        return true;
    }

    // Turn 'R_triple = A_other_triple_constant' into a triple const assign
    if (A->typespec().is_triple() && R->typespec().is_triple() &&
        A->typespec() != R->typespec()) {
        Vec3 *f = (Vec3 *)A->data();
        int cind = add_constant (R->typespec(), f);
        turn_into_assign (op, cind);
        return true;
    }

    return false;
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
RuntimeOptimizer::make_param_use_instanceval (Symbol *R)
{
    // Mark its source as the default value, and not connected
    R->valuesource (Symbol::InstanceVal);
    R->connected (false);
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
        for (int i = R->initbegin();  i < R->initend();  ++i)
            turn_into_nop (inst()->ops()[i]);
        R->initbegin (0);
        R->initend (0);
    }
    // Erase R's incoming connections
    erase_if (inst()->connections(), ConnectionDestIs(*inst(),R));
}



/// Check for assignment of output params that are written only once in
/// the whole shader -- on this statement -- and assigned a constant, and
/// the assignment is unconditional.  In that case, just alias it to the
/// constant from here on out.
///
/// Furthermore, if nobody READS the output param prior to this
/// assignment, let's just change its initial value to the constant and
/// get rid of the assignment altogether!
///
/// Return true if the assignment is removed entirely.
bool
RuntimeOptimizer::outparam_assign_elision (int opnum, Opcode &op)
{
    ASSERT (op.opname() == u_assign);
    Symbol *R (inst()->argsymbol(op.firstarg()+0));
    Symbol *A (inst()->argsymbol(op.firstarg()+1));

    if (A->is_constant() && R->typespec() == A->typespec() &&
            R->symtype() == SymTypeOutputParam &&
            R->firstwrite() == opnum && R->lastwrite() == opnum &&
            !m_in_conditional[opnum]) {
        // It's assigned only once, and unconditionally assigned a
        // constant -- alias it
        int cind = inst()->args()[op.firstarg()+1];
        global_alias (inst()->args()[op.firstarg()], cind);

        // If it's also never read before this assignment, just replace its
        // default value entirely and get rid of the assignment.
        if (R->firstread() > opnum) {
            make_param_use_instanceval (R);
            replace_param_value (R, A->data());
            turn_into_nop (op);
            return true;
        }
    }
    return false;
}




/// If every potentially-written argument to this op is NEVER read, turn
/// it into a nop and return true.  We don't do this to ops that have no
/// written args at all, since they tend to have side effects (e.g.,
/// printf, setmessage).
bool
RuntimeOptimizer::useless_op_elision (Opcode &op)
{
    if (op.nargs()) {
        bool noeffect = true;
        bool writes_something = false;
        for (int a = 0;  a < op.nargs();  ++a) {
            if (op.argwrite(a)) {
                writes_something = true;
                Symbol *A (inst()->argsymbol(op.firstarg()+a));
                bool local_or_tmp = (A->symtype() == SymTypeLocal ||
                                     A->symtype() == SymTypeTemp);
                if (A->everread() || ! local_or_tmp)
                    noeffect = false;
            }
        }
        if (writes_something && noeffect) {
            turn_into_nop (op);
            return true;
        }
    }
    return false;
}



int
RuntimeOptimizer::dealias_symbol (int symindex)
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

    // Two assignments in a row to the same variable -- get rid of the first
    if (op.opname() == u_assign &&
          next.opname() == u_assign &&
          opargsym(op,0) == opargsym(next,0)) {
        // std::cerr << "double-assign " << opnum << " & " << op2num << ": " 
        //           << opargsym(op,0)->mangled() << "\n";
        turn_into_nop (op);
        return 1;
    }

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
                turn_into_nop (next);
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
                turn_into_assign (next, inst()->arg(op.firstarg()+1));
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
        turn_into_nop (op);
        turn_into_nop (next);
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
    for (int i = 0;  i < (int)inst()->symbols().size();  ++i)
        if (inst()->symbol(i)->symtype() == SymTypeConst)
            m_all_consts.push_back (i);

    // Turn all geom-locked parameters into constants.
    if (m_shadingsys.optimize() >= 2) {
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
        for (int opnum = 0;  opnum < (int)inst()->ops().size();  ++opnum) {
            Opcode &op (inst()->ops()[opnum]);
            // Find the farthest this instruction jumps to (-1 for ops
            // that don't jump) so we can mark conditional regions.
            int jumpend = op.farthest_jump();
            for (int i = (int)opnum+1;  i < jumpend;  ++i)
                m_in_conditional[i] = true;

            // If we've just moved to a new basic block, clear the aliases
            if (lastblock != m_bblockids[opnum]) {
                clear_block_aliases ();
                lastblock = m_bblockids[opnum];
            }

            // De-alias the readable args to the op and figure out if
            // there are any constants involved.
            for (int i = 0;  i < op.nargs();  ++i) {
                if (op.argwrite(i))
                    continue;    // Don't de-alias args that are written
                int argindex = op.firstarg() + i;
                int argsymindex = dealias_symbol (inst()->arg(argindex));
                inst()->args()[argindex] = argsymindex;
            }

            // Make sure there's room for at least one more symbol, so that
            // we can add a const if we need to, without worrying about the
            // addresses of symbols changing when we add a new one below.
            make_symbol_room (1);

            // For various ops that we know how to effectively
            // constant-fold, dispatch to the appropriate routine.
            if (m_shadingsys.optimize() >= 2) {
                FolderTable::const_iterator found = folder_table.find (op.opname());
                if (found != folder_table.end())
                    changed += (*found->second) (*this, opnum);
            }

            // Clear local block aliases for any args that were written
            // by this op
            for (int i = 0;  i < op.nargs();  ++i)
                if (op.argwrite(i))
                    block_unalias (inst()->arg(op.firstarg()+i));

            // Get rid of an 'if' if it contains no statements to execute
            if (m_shadingsys.optimize() >= 2 && op.opname() == u_if) {
                int jump = op.farthest_jump ();
                bool only_nops = true;
                for (int i = opnum+1;  i < jump && only_nops;  ++i)
                    only_nops &= (inst()->ops()[i].opname() == u_nop);
                if (only_nops) {
                    turn_into_nop (op);
                    changed = 1;
                    continue;
                }
            }

            // Now we handle assignments.
            //
            // N.B. This is a regular "if", not an "else if", because we
            // definitely want to catch any 'assign' statements that
            // were put in by the constant folding routines above.
            if (m_shadingsys.optimize() >= 2 && op.opname() == u_assign/* &&
                                                                                   inst()->argsymbol(op.firstarg()+1)->is_constant()*/) {
                Symbol *R (inst()->argsymbol(op.firstarg()+0));
                Symbol *A (inst()->argsymbol(op.firstarg()+1));
                bool R_local_or_tmp = (R->symtype() == SymTypeLocal ||
                                       R->symtype() == SymTypeTemp);

                if (block_alias(inst()->arg(op.firstarg())) == inst()->arg(op.firstarg()+1) ||
                    block_alias(inst()->arg(op.firstarg()+1)) == inst()->arg(op.firstarg())) {
                    // We're re-assigning something already aliased, skip it
                    turn_into_nop (op);
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

                if (A->is_constant() &&
                        equivalent(R->typespec(), A->typespec())) {
                    block_alias (inst()->arg(op.firstarg()),
                                     inst()->arg(op.firstarg()+1));
//                  std::cerr << opnum << " aliasing " << R->mangled() " to "
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
                    turn_into_nop (op);
                    ++changed;
                    continue;
                }
                if (R_local_or_tmp && ! R->everread()) {
                    // This local is written but NEVER READ.  nop it.
                    turn_into_nop (op);
                    ++changed;
                    continue;
                }
                if (outparam_assign_elision (opnum, op)) {
                    ++changed;
                    continue;
                }
                if (R == A) {
                    // Just an assignment to itself -- turn into NOP!
                    turn_into_nop (op);
                    ++changed;
                } else if (R_local_or_tmp && R->lastread() < opnum) {
                    // Don't bother assigning if we never read it again
                    turn_into_nop (op);
                    ++changed;
                }
            }

            if (m_shadingsys.optimize() >= 2)
                changed += useless_op_elision (op);

            // Peephole optimization involving pair of instructions
            if (m_shadingsys.optimize() >= 2)
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
                for (int i = s.initbegin();  i < s.initend();  ++i)
                    turn_into_nop (inst()->ops()[i]);
                s.set_initrange ();
                s.clear_rw ();
                ++changed;
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
        for (int i = 0;  i < (int)inst()->ops().size()-1;  ++i)
            turn_into_nop (inst()->ops()[i]);
        BOOST_FOREACH (Symbol &s, inst()->symbols())
            s.clear_rw ();
    }

    // Erase this layer's incoming connections and init ops for params
    // it no longer uses
    erase_if (inst()->connections(), ConnectionDestNeverUsed(inst()));

    // Clear init ops of params that aren't used.
    // FIXME -- is this ineffective?  Should it be never READ?
    FOREACH_PARAM (Symbol &s, inst()) {
        if (s.symtype() == SymTypeParam && ! s.everused() &&
                s.initbegin() < s.initend()) {
            for (int i = s.initbegin();  i < s.initend();  ++i)
                turn_into_nop (inst()->ops()[i]);
            s.set_initrange (0, 0);
        }
    }

    // Now that we've optimized this layer, walk through the ops and
    // note which messages may have been sent, so subsequent layers will
    // know.
    for (int opnum = 0;  opnum < (int)inst()->ops().size();  ++opnum) {
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
                        // Careful -- not all globals can take derivs
                        Symbol &s (*opargsym (op, a));
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
              !s.typespec().is_closure() && s.mangled() != Strings::N)
            s.has_derivs(true);
        if (s.has_derivs())
            add_dependency (symdeps, DerivSym, snum);
        ++snum;
    }

    // Mark all symbols needing derivatives as such
    BOOST_FOREACH (int d, symdeps[DerivSym]) {
        Symbol *s = inst()->symbol(d);
        if (! s->typespec().is_closure() && 
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

    add_useparam (allsymptrs);

    if (m_shadingsys.optimize() >= 1)
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
    {
        // adjust memory stats
        // Remember that they're already swapped
        off_t mem = vectorbytes(new_ops);
        ShadingSystemImpl &ss (shadingsys());
        spin_lock lock (ss.m_stat_mutex);
        ss.m_stat_mem_inst_ops -= mem;
        ss.m_stat_mem_inst -= mem;
        ss.m_stat_memory -= mem;
        mem = vectorbytes(inst()->m_instops);
        ss.m_stat_mem_inst_ops += mem;
        ss.m_stat_mem_inst += mem;
        ss.m_stat_memory += mem;
    }
}



void
RuntimeOptimizer::optimize_group ()
{
    Timer rop_timer;
    initialize_folder_table ();

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
        if (m_shadingsys.debug() && m_shadingsys.optimize() >= 1 && layer==0) {
            std::cout << "Before optimizing layer " << layer << " " 
                      << inst()->layername() 
                      << ", I get:\n" << inst()->print()
                      << "\n--------------------------------\n\n";
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
                if (! source->typespec().is_closure() &&
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
        if (inst()->unused()) {
            // Clear the syms and ops, we'll never use this layer
            SymbolVec nosyms;
            std::swap (inst()->symbols(), nosyms);
            OpcodeVec noops;
            std::swap (inst()->ops(), noops);
            {
                // adjust memory stats
                // Remember that they're already swapped
                off_t symmem = vectorbytes(nosyms);
                off_t opmem = vectorbytes(noops);
                ShadingSystemImpl &ss (shadingsys());
                spin_lock lock (ss.m_stat_mutex);
                ss.m_stat_mem_inst_syms -= symmem;
                ss.m_stat_mem_inst_ops -= opmem;
                ss.m_stat_mem_inst -= (symmem+opmem);
                ss.m_stat_memory -= (symmem+opmem);
            }
            continue;
        }
        if (m_shadingsys.optimize() >= 1) {
            collapse_syms ();
            collapse_ops ();
            if (m_shadingsys.debug()) {
                track_variable_lifetimes ();
                std::cout << "After optimizing layer " << layer << " " 
                          << inst()->layername() << " (" << inst()->id()
                          << "): \n" << inst()->print() 
                          << "\n--------------------------------\n\n";
            }
        }
        new_nsyms += inst()->symbols().size();
        new_nops += inst()->ops().size();
    }

    m_stat_specialization_time = rop_timer();

    Timer timer;
    // Let's punt on multithreading LLVM for the time being,
    // just make a big lock.
    static mutex llvm_mutex;
    lock_guard llvm_lock (llvm_mutex);
    m_stat_opt_locking_time = timer();

    m_shadingsys.SetupLLVM ();
    m_stat_llvm_setup_time = timer() - m_stat_opt_locking_time;
    build_llvm_group ();

    m_stat_total_llvm_time = timer();


    m_shadingsys.info ("Optimized shader group: New syms %llu/%llu (%5.1f%%), ops %llu/%llu (%5.1f%%)",
          new_nsyms, old_nsyms,
          100.0*double((long long)new_nsyms-(long long)old_nsyms)/double(old_nsyms),
          new_nops, old_nops,
          100.0*double((long long)new_nops-(long long)old_nops)/double(old_nops));
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
}



}; // namespace pvt
}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
