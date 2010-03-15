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

#include <boost/foreach.hpp>

#include "oslexec_pvt.h"
#include "oslops.h"
#include "../liboslcomp/oslcomp_pvt.h"
using namespace OSL;
using namespace OSL::pvt;



#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
namespace pvt {   // OSL::pvt


// Search for a constant whose type and value match type and data[...].
// Return -1 if no matching const is found.
int
find_constant (const SymbolVec &syms, const std::vector<int> &all_consts,
               const TypeSpec &type, const void *data)
{
    for (int i = 0;  i < (int)all_consts.size();  ++i) {
        const Symbol &s (syms[all_consts[i]]);
        ASSERT (s.symtype() == SymTypeConst);
        if (equivalent (s.typespec(), type) &&
              !memcmp (s.data(), data, s.typespec().simpletype().size())) {
            return all_consts[i];
        }
    }
    return -1;
}



static int
add_constant (SymbolVec &syms, std::vector<int> &all_consts,
              int &next_newconst, const TypeSpec &type, const void *data)
{
    int ind = find_constant (syms, all_consts, type, data);
    if (ind < 0) {
        Symbol newconst (ustring::format ("$newconst%d", next_newconst++),
                         type, SymTypeConst);
        newconst.data ((void *)data);
        ASSERT (syms.capacity() > syms.size());  // ensure we don't realloc
        ind = (int) syms.size ();
        syms.push_back (newconst);
        all_consts.push_back (ind);
    }
    return ind;
}



// Insert instruction 'opname' with arguments 'args_to_add' into the 
// code at instruction 'opnum'.  The existing code and concatenated 
// argument lists can be found in code and opargs, respectively, and
// allsyms contains pointers to all symbols.  mainstart is a reference
// to the address where the 'main' shader begins, and may be modified
// if the new instruction is inserted before that point.
void
insert_code (ustring opname, OpImpl impl,
             std::vector<int> &args_to_add, int opnum, 
             OpcodeVec &code, std::vector<int> &opargs,
             SymbolPtrVec &allsyms, int &mainstart)
{
    ustring method = (opnum < (int)code.size()) ? code[opnum].method() : OSLCompilerImpl::main_method_name();
    Opcode op (opname, method, opargs.size(), args_to_add.size());
    op.implementation (impl);
    code.insert (code.begin()+opnum, op);
    opargs.insert (opargs.end(), args_to_add.begin(), args_to_add.end());
    if (opnum < mainstart)
        ++mainstart;

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
        BOOST_FOREACH (Symbol *s, allsyms) {
            if (s->symtype() == SymTypeParam ||
                  s->symtype() == SymTypeOutputParam) {
                if (s->initbegin() > opnum)
                    s->initbegin (s->initbegin()+1);
                if (s->initend() > opnum)
                    s->initend (s->initend()+1);
            }
        }
    }
}



// Insert a 'useparam' instruction in front of instruction 'opnum', to
// reference the symbols in 'params'.
void
insert_useparam (OpcodeVec &code, size_t opnum, std::vector<int> &opargs,
                 SymbolPtrVec &allsyms, std::vector<int> &params_to_use,
                 int &mainstart)
{
    static ustring useparam("useparam");
    insert_code (useparam, OP_useparam, params_to_use, opnum,
                 code, opargs, allsyms, mainstart);

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



// Add a 'useparam' before any op that reads parameters.  This is what
// tells the runtime that it needs to run the layer it came from, if
// not already done.
void
add_useparam (OpcodeVec &code, std::vector<int> &opargs,
              SymbolPtrVec &allsyms, int &mainstart)
{
    // Mark all symbols as un-initialized
    BOOST_FOREACH (Symbol *s, allsyms)
        s->initialized (false);

    if (mainstart < 0)
        mainstart = (int)code.size();

    // Take care of the output params right off the bat -- as soon as the
    // shader starts running 'main'.
    std::vector<int> outputparams;
    for (int i = 0;  i < (int)allsyms.size();  ++i) {
        Symbol *s = allsyms[i];
        if (s->symtype() == SymTypeOutputParam) {
            outputparams.push_back (i);
            s->initialized (true);
        }
    }
    if (outputparams.size())
        insert_useparam (code, mainstart, opargs, allsyms, outputparams,
                         mainstart);

    // Figure out which statements are inside conditional states
    std::vector<bool> in_conditional (code.size(), false);
    for (size_t opnum = 0;  opnum < code.size();  ++opnum) {
        // Find the farthest this instruction jumps to (-1 for instructions
        // that don't jump)
        int jumpend = code[opnum].farthest_jump();
        // Mark all instructions from here to there as inside conditionals
        for (int i = (int)opnum+1;  i < jumpend;  ++i)
            in_conditional[i] = true;
    }

    // Loop over all ops...
    for (int opnum = 0;  opnum < (int)code.size();  ++opnum) {
        Opcode &op (code[opnum]);  // handy ref to the op
        if (op.opname() == "useparam")
            continue;  // skip useparam ops themselves, if we hit one
        std::vector<int> params;   // list of params referenced by this op
        // For each argument...
        for (int a = 0;  a < op.nargs();  ++a) {
            int argind = op.firstarg() + a;
            SymbolPtr s = allsyms[opargs[argind]];
            DASSERT (s->dealias() == s);
            // If this arg is a param and is read, remember it
            if (s->symtype() != SymTypeParam && s->symtype() != SymTypeOutputParam)
                continue;  // skip non-params
            // skip if we've already 'usedparam'ed it unconditionally
            if (s->initialized() && opnum >= mainstart)
                continue;
            bool inside_init = (opnum >= s->initbegin() && opnum < s->initend());
            if (op.argread(a) || (op.argwrite(a) && !inside_init)) {
                // Don't add it more than once
                if (std::find (params.begin(), params.end(), opargs[argind]) == params.end()) {
                    params.push_back (opargs[argind]);
                    // mark as already initialized unconditionally, if we do
                    if (! in_conditional[opnum] && op.method() == OSLCompilerImpl::main_method_name())
                        s->initialized (true);
                }
            }
        }

        // If the arg we are examining read any params, insert a "useparam"
        // op whose arguments are the list of params we are about to use.
        if (params.size()) {
            insert_useparam (code, opnum, opargs, allsyms, params, mainstart);
            in_conditional.insert (in_conditional.begin()+opnum, false);
            // Skip the op we just added
            ++opnum;
        }
    }
    // Mark all symbols as un-initialized
    BOOST_FOREACH (Symbol *s, allsyms)
        s->initialized (false);
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
    return (Atype.is_float() && *(const float *)A.data() == 0) ||
        (Atype.is_int() && *(const int *)A.data() == 0) ||
        (Atype.is_triple() && ((const float *)A.data())[0] == 0 &&
         ((const float *)A.data())[1] == 0 && ((const float *)A.data())[2] == 0);
}



inline bool
is_one (const Symbol &A)
{
    const TypeSpec &Atype (A.typespec());
    return (Atype.is_float() && *(const float *)A.data() == 1) ||
        (Atype.is_int() && *(const int *)A.data() == 1) ||
        (Atype.is_triple() && ((const float *)A.data())[0] == 1 &&
         ((const float *)A.data())[1] == 1 && ((const float *)A.data())[2] == 1);
}



int
constfold_add (OpcodeVec &ops, int opnum,
               std::vector<int> &args, SymbolVec &symbols,
               std::vector<int> &all_consts, int &next_newconst)
{
    Opcode &op (ops[opnum]);
    Symbol &A (symbols[args[op.firstarg()+1]]);
    Symbol &B (symbols[args[op.firstarg()+2]]);
    if (A.is_const()) {
        if (is_zero(A)) {
            // R = 0 + B  =>   R = 0
            op.reset (ustring("assign"), OP_assign, 2);   // change to assign
            return 1;
        }
    } else if (B.is_const()) {
        if (is_zero(B)) {
            // R = A + 0   =>   R = 0
            op.reset (ustring("assign"), OP_assign, 2);
            args[op.firstarg()+1] = args[op.firstarg()+2]; // arg 1 gets B
            return 1;
        }
    }
    return 0;
}



int
constfold_sub (OpcodeVec &ops, int opnum,
               std::vector<int> &args, SymbolVec &symbols,
               std::vector<int> &all_consts, int &next_newconst)
{
    Opcode &op (ops[opnum]);
    Symbol &B (symbols[args[op.firstarg()+2]]);
    if (B.is_const()) {
        if (is_zero(B)) {
            // R = A - 0   =>   R = A
            op.reset (ustring("assign"), OP_assign, 2);
            return 1;
        }
    }
    return 0;
}



int
constfold_mul (OpcodeVec &ops, int opnum,
               std::vector<int> &args, SymbolVec &symbols,
               std::vector<int> &all_consts, int &next_newconst)
{
    Opcode &op (ops[opnum]);
    Symbol &A (symbols[args[op.firstarg()+1]]);
    Symbol &B (symbols[args[op.firstarg()+2]]);
    if (A.is_const()) {
        if (is_one(A)) {
            // R = 1 * B  =>   R = B
            op.reset (ustring("assign"), OP_assign, 2);   // change to assign
            args[op.firstarg()+1] = args[op.firstarg()+2]; // arg 1 gets B
            return 1;
        }
    } else if (B.is_const()) {
        if (is_one(B)) {
            // R = A * 1   =>   R = A
            op.reset (ustring("assign"), OP_assign, 2);
            return 1;
        }
    }
    return 0;
}



int
constfold_div (OpcodeVec &ops, int opnum,
               std::vector<int> &args, SymbolVec &symbols,
               std::vector<int> &all_consts, int &next_newconst)
{
    Opcode &op (ops[opnum]);
    Symbol &A (symbols[args[op.firstarg()+1]]);
    Symbol &B (symbols[args[op.firstarg()+2]]);
    if (B.is_const()) {
        if (is_one(B)) {
            // R = A / 1   =>   R = A
            op.reset (ustring("assign"), OP_assign, 2);
            return 1;
        }
        else if (A.is_const()) {
            // const/const
            // FIXME!  need to make new consts for this
        }
    }
    return 0;
}



int
constfold_eq (OpcodeVec &ops, int opnum,
              std::vector<int> &args, SymbolVec &symbols,
              std::vector<int> &all_consts, int &next_newconst)
{
    Opcode &op (ops[opnum]);
    Symbol &A (symbols[args[op.firstarg()+1]]);
    Symbol &B (symbols[args[op.firstarg()+2]]);
    if (A.is_const() && B.is_const()) {
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
        int cind = add_constant (symbols, all_consts,
                                 next_newconst, TypeDesc::TypeInt,
                                 val ? &int_one : &int_zero);
        op.reset (ustring("assign"), OP_assign, 2);
        args[op.firstarg()+1] = cind;
        return 1;
    }
    return 0;
}



int
constfold_neq (OpcodeVec &ops, int opnum,
               std::vector<int> &args, SymbolVec &symbols,
               std::vector<int> &all_consts, int &next_newconst)
{
    Opcode &op (ops[opnum]);
    Symbol &A (symbols[args[op.firstarg()+1]]);
    Symbol &B (symbols[args[op.firstarg()+2]]);
    if (A.is_const() && B.is_const()) {
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
        int cind = add_constant (symbols, all_consts,
                                 next_newconst, TypeDesc::TypeInt,
                                 val ? &int_one : &int_zero);
        op.reset (ustring("assign"), OP_assign, 2);
        args[op.firstarg()+1] = cind;
        return 1;
    }
    return 0;
}



int
constfold_lt (OpcodeVec &ops, int opnum,
              std::vector<int> &args, SymbolVec &symbols,
              std::vector<int> &all_consts, int &next_newconst)
{
    static const int int_zero = 0, int_one = 1;
    Opcode &op (ops[opnum]);
    Symbol &A (symbols[args[op.firstarg()+1]]);
    Symbol &B (symbols[args[op.firstarg()+2]]);
    const TypeSpec &ta (A.typespec()); 
    const TypeSpec &tb (B.typespec()); 
    if (A.is_const() && B.is_const()) {
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
        int cind = add_constant (symbols, all_consts,
                                 next_newconst, TypeDesc::TypeInt,
                                 val ? &int_one : &int_zero);
        op.reset (ustring("assign"), OP_assign, 2);
        args[op.firstarg()+1] = cind;
        return 1;
    }
    return 0;
}



int
constfold_le (OpcodeVec &ops, int opnum,
              std::vector<int> &args, SymbolVec &symbols,
              std::vector<int> &all_consts, int &next_newconst)
{
    static const int int_zero = 0, int_one = 1;
    Opcode &op (ops[opnum]);
    Symbol &A (symbols[args[op.firstarg()+1]]);
    Symbol &B (symbols[args[op.firstarg()+2]]);
    const TypeSpec &ta (A.typespec()); 
    const TypeSpec &tb (B.typespec()); 
    if (A.is_const() && B.is_const()) {
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
        int cind = add_constant (symbols, all_consts,
                                 next_newconst, TypeDesc::TypeInt,
                                 val ? &int_one : &int_zero);
        op.reset (ustring("assign"), OP_assign, 2);
        args[op.firstarg()+1] = cind;
        return 1;
    }
    return 0;
}



int
constfold_gt (OpcodeVec &ops, int opnum,
              std::vector<int> &args, SymbolVec &symbols,
              std::vector<int> &all_consts, int &next_newconst)
{
    static const int int_zero = 0, int_one = 1;
    Opcode &op (ops[opnum]);
    Symbol &A (symbols[args[op.firstarg()+1]]);
    Symbol &B (symbols[args[op.firstarg()+2]]);
    const TypeSpec &ta (A.typespec()); 
    const TypeSpec &tb (B.typespec()); 
    if (A.is_const() && B.is_const()) {
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
        int cind = add_constant (symbols, all_consts,
                                 next_newconst, TypeDesc::TypeInt,
                                 val ? &int_one : &int_zero);
        op.reset (ustring("assign"), OP_assign, 2);
        args[op.firstarg()+1] = cind;
        return 1;
    }
    return 0;
}



int
constfold_ge (OpcodeVec &ops, int opnum,
              std::vector<int> &args, SymbolVec &symbols,
              std::vector<int> &all_consts, int &next_newconst)
{
    static const int int_zero = 0, int_one = 1;
    Opcode &op (ops[opnum]);
    Symbol &A (symbols[args[op.firstarg()+1]]);
    Symbol &B (symbols[args[op.firstarg()+2]]);
    const TypeSpec &ta (A.typespec()); 
    const TypeSpec &tb (B.typespec()); 
    if (A.is_const() && B.is_const()) {
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
        int cind = add_constant (symbols, all_consts,
                                 next_newconst, TypeDesc::TypeInt,
                                 val ? &int_one : &int_zero);
        op.reset (ustring("assign"), OP_assign, 2);
        args[op.firstarg()+1] = cind;
        return 1;
    }
    return 0;
}



int
constfold_or (OpcodeVec &ops, int opnum,
              std::vector<int> &args, SymbolVec &symbols,
              std::vector<int> &all_consts, int &next_newconst)
{
    Opcode &op (ops[opnum]);
    Symbol &A (symbols[args[op.firstarg()+1]]);
    Symbol &B (symbols[args[op.firstarg()+2]]);
    if (A.is_const() && B.is_const()) {
        DASSERT (A.typespec().is_int() && B.typespec().is_int());
        bool val = *(int *)A.data() || *(int *)B.data();
        // Turn the 'or R A B' into 'assign R X' where X is 0 or 1.
        static const int int_zero = 0, int_one = 1;
        int cind = add_constant (symbols, all_consts,
                                 next_newconst, TypeDesc::TypeInt,
                                 val ? &int_one : &int_zero);
        op.reset (ustring("assign"), OP_assign, 2);
        args[op.firstarg()+1] = cind;
        return 1;
    }
    return 0;
}



int
constfold_and (OpcodeVec &ops, int opnum,
               std::vector<int> &args, SymbolVec &symbols,
               std::vector<int> &all_consts, int &next_newconst)
{
    Opcode &op (ops[opnum]);
    Symbol &A (symbols[args[op.firstarg()+1]]);
    Symbol &B (symbols[args[op.firstarg()+2]]);
    if (A.is_const() && B.is_const()) {
        DASSERT (A.typespec().is_int() && B.typespec().is_int());
        bool val = *(int *)A.data() && *(int *)B.data();
        // Turn the 'or R A B' into 'assign R X' where X is 0 or 1.
        static const int int_zero = 0, int_one = 1;
        int cind = add_constant (symbols, all_consts,
                                 next_newconst, TypeDesc::TypeInt,
                                 val ? &int_one : &int_zero);
        op.reset (ustring("assign"), OP_assign, 2);
        args[op.firstarg()+1] = cind;
        return 1;
    }
    return 0;
}



int
constfold_if (OpcodeVec &ops, int opnum,
              std::vector<int> &args, SymbolVec &symbols,
              std::vector<int> &all_consts, int &next_newconst)
{
    Opcode &op (ops[opnum]);
    Symbol &C (symbols[args[op.firstarg()+0]]);
    if (C.is_const()) {
        int result = -1;   // -1 == we don't know
        if (C.typespec().is_int())
            result = (((int *)C.data())[0] != 0);
        else if (C.typespec().is_float())
            result = (((float *)C.data())[0] != 0.0f);
        else if (C.typespec().is_triple())
            result = (((Vec3 *)C.data())[0] != Vec3(0,0,0));
        else if (C.typespec().is_string()) {
            ustring s = ((ustring *)C.data())[0];
            result = (s.c_str() == NULL || s.length() == 0);
        }
        int changed = 0;
        if (result > 0)
            for (int i = op.jump(0);  i < op.jump(1);  ++i, ++changed)
                ops[i].reset (ustring("nop"), OP_nop, 0);
        else if (result == 0)
            for (int i = opnum+1;  i < op.jump(0);  ++i, ++changed)
                ops[i].reset (ustring("nop"), OP_nop, 0);
        op.reset (ustring("nop"), OP_nop, 0);
        return changed+1;
    }
    return 0;
}



int
constfold_aref (OpcodeVec &ops, int opnum,
                std::vector<int> &args, SymbolVec &symbols,
                std::vector<int> &all_consts, int &next_newconst)
{
    // Array reference -- crops up more than you think in production shaders!
    // Try to turn R=A[I] into R=C if A and I are const.
    Opcode &op (ops[opnum]);
    Symbol &R (symbols[args[op.firstarg()+0]]);
    Symbol &A (symbols[args[op.firstarg()+1]]);
    Symbol &Index (symbols[args[op.firstarg()+2]]);
    if (A.is_const() && Index.is_const()) {
        TypeSpec elemtype = A.typespec().elementtype();
        ASSERT (elemtype.is_float() || elemtype.is_triple() || elemtype.is_int());
        ASSERT (equivalent(elemtype, R.typespec()));
        int index = *(int *)Index.data();
        int cind = add_constant (symbols, all_consts, next_newconst, elemtype,
                                 (char *)A.data() + index*elemtype.simpletype().size());
        std::cerr << "    array index " << index << ": ";
        if (elemtype.is_float())
            std::cerr << (*(float *)((char *)A.data() + index*elemtype.simpletype().size()));
        std::cerr << "\n";
        op.reset (ustring("assign"), OP_assign, 2);
        args[op.firstarg()+1] = cind;
        return 1;
    }
    return 0;
}



int
constfold_useparam (OpcodeVec &ops, int opnum,
                    std::vector<int> &args, SymbolVec &symbols,
                    std::vector<int> &all_consts, int &next_newconst)
{
    // Just eliminate useparam (from shaders compiled with old oslc)
    Opcode &op (ops[opnum]);
    op.reset (ustring("nop"), OP_nop, 0);
    return 1;
}



void
ShadingSystemImpl::optimize_instance (ShaderGroup &group, int layer,
                                      ShaderInstance &inst)
{
    if (debug()) {
        std::cerr << "Optimzing level " << optimize() << ' ' << Strutil::format("%p",&inst) << "\n";
        std::cerr << "Optimizing layer " << layer << " " << inst.layername()
                  << ' ' << inst.shadername() << "\n";
        std::cerr << "Before optimizing instance, I get: \n" << inst.print() 
                  << "\n--------------------------------\n\n";
    }

    // Start by making a list of the indices of all constants.
    std::vector<int> all_consts;
    for (int i = 0;  i < (int)inst.symbols().size();  ++i)
        if (inst.symbol(i)->symtype() == SymTypeConst)
            all_consts.push_back (i);

    typedef std::map<int,int> IntMap;
    IntMap symbol_aliases;   // Track symbol aliases
    int next_newconst = 0;   // Index of next new constant

    // Turn all geom-locked parameters into constants.  While we're at it,
    // build up the list of all constants.
    if (optimize() >= 2) {
        for (int i = inst.firstparam();  i <= inst.lastparam();  ++i) {
            Symbol *s (inst.symbol(i));
            if (s->symtype() == SymTypeParam && s->lockgeom() &&
                s->valuesource() != Symbol::ConnectedVal &&
                !(s->valuesource() == Symbol::DefaultVal && s->initbegin() < s->initend()) &&
                !s->typespec().is_structure() && !s->typespec().is_closure()) {
                // Make a new const using the param's instance values
                inst.make_symbol_room (1);
                s = inst.symbol(i);  // In case make_symbol_room changed ptrs
                int cind = add_constant (inst.symbols(), all_consts,
                                         next_newconst, s->typespec(), s->data());
                // Alias this symbol to the new const
                symbol_aliases[i] = cind;
            }
        }
    }

    // Try to fold constants.  We take several passes, until we get to
    // the point that not much is improving.  It rarely goes beyond 3-4
    // passes, but we have a hard cutoff at 10 just to be sure we don't
    // ever get into an infinite loop from an unforseen cycle.  where we
    // end up inadvertently transforming A => B => A => etc.
    SymbolPtrVec allsymptrs;
    int totalchanged = 0;
    for (int pass = 0;  pass < 10;  ++pass) {

        // Track which statements are inside conditional states
        std::vector<bool> in_conditional (inst.ops().size(), false);

        int opnum = 0;
        int changed = 0;
        BOOST_FOREACH (Opcode &op, inst.ops()) {
            // Find the farthest this instruction jumps to (-1 for ops
            // that don't jump) so we can mark conditional regions.
            int jumpend = op.farthest_jump();
            for (int i = (int)opnum+1;  i < jumpend;  ++i)
                in_conditional[i] = true;

            // De-alias the args to the op and figure out if there are
            // any constants involved.
            bool any_const_args = false;
            for (int i = 0;  i < op.nargs();  ++i) {
                int argindex = op.firstarg() + i;
                if (op.argwrite(i))  // Don't alias args that are written
                    continue;
                do {
                    IntMap::const_iterator found;
                    found = symbol_aliases.find (inst.arg (argindex));
                    if (found != symbol_aliases.end()) {  // an alias for the sym
                        inst.args()[argindex] = found->second;
                        continue;
                    }
                } while (0);
                any_const_args |= inst.argsymbol(argindex)->is_const();
            }

            // We aren't currently doing anything to optimize ops that have
            // no const args, so don't bother looking further in that case.
            if (! any_const_args) {
                ++opnum;
                continue;
            }

            // Make sure there's room for at least one more symbol, so that
            // we can add a const if we need to, without worrying about the
            // addresses of symbols changing when we add a new one below.
            inst.make_symbol_room (1);

            // For various ops that we know how to effectively
            // constant-fold, dispatch to the appropriate routine.
            //
            // FIXME -- eventually replace the following cascacing 'if'
            // with a hash lookup and single function call.
            if (optimize() < 2) {
                // no constant folding
            } else if (op.implementation() == OP_mul) {
                changed += constfold_mul (inst.ops(), opnum, inst.args(),
                           inst.symbols(), all_consts, next_newconst);
            } else if (op.implementation() == OP_div) {
                changed += constfold_div (inst.ops(), opnum, inst.args(),
                           inst.symbols(), all_consts, next_newconst);
            } else if (op.implementation() == OP_add) {
                changed += constfold_add (inst.ops(), opnum, inst.args(),
                           inst.symbols(), all_consts, next_newconst);
            } else if (op.implementation() == OP_sub) {
                changed += constfold_sub (inst.ops(), opnum, inst.args(),
                           inst.symbols(), all_consts, next_newconst);
            } else if (op.implementation() == OP_eq) {
                changed += constfold_eq (inst.ops(), opnum, inst.args(),
                          inst.symbols(), all_consts, next_newconst);
            } else if (op.implementation() == OP_neq) {
                changed += constfold_neq (inst.ops(), opnum, inst.args(),
                           inst.symbols(), all_consts, next_newconst);
            } else if (op.implementation() == OP_le) {
                changed += constfold_le (inst.ops(), opnum, inst.args(),
                          inst.symbols(), all_consts, next_newconst);
            } else if (op.implementation() == OP_ge) {
                changed += constfold_ge (inst.ops(), opnum, inst.args(),
                          inst.symbols(), all_consts, next_newconst);
            } else if (op.implementation() == OP_lt) {
                changed += constfold_lt (inst.ops(), opnum, inst.args(),
                          inst.symbols(), all_consts, next_newconst);
            } else if (op.implementation() == OP_gt) {
                changed += constfold_gt (inst.ops(), opnum, inst.args(),
                          inst.symbols(), all_consts, next_newconst);
            } else if (op.implementation() == OP_or) {
                changed += constfold_or (inst.ops(), opnum, inst.args(),
                          inst.symbols(), all_consts, next_newconst);
            } else if (op.implementation() == OP_and) {
                changed += constfold_and (inst.ops(), opnum, inst.args(),
                          inst.symbols(), all_consts, next_newconst);
            } else if (op.implementation() == OP_if) {
                changed += constfold_if (inst.ops(), opnum, inst.args(),
                          inst.symbols(), all_consts, next_newconst);
#if 0
// Comment out for now -- it's broken, but I'm not sure why
            } else if (op.implementation() == OP_aref) {
                std::cerr << "\n\nAref opt: " << inst.shadername() << ' ' << opnum << "\n";
                changed += constfold_aref (inst.ops(), opnum, inst.args(),
                          inst.symbols(), all_consts, next_newconst);
#endif
            } else if (op.implementation() == OP_useparam) {
                changed += constfold_useparam (inst.ops(), opnum, inst.args(),
                                inst.symbols(), all_consts, next_newconst);
            }

            // Now we handle assignments.
            //
            // N.B. This is a regular "if", not an "else if", because we
            // definitely want to catch any 'assign' statements that
            // were put in by the constant folding routines above.
            if (optimize() >= 2 && op.implementation() == OP_assign) {
                Symbol *R (inst.argsymbol(op.firstarg()+0));
                Symbol *A (inst.argsymbol(op.firstarg()+1));
                bool R_local_or_tmp = (R->symtype() == SymTypeLocal ||
                                       R->symtype() == SymTypeTemp);
                if (A->is_const() && R->typespec() == A->typespec() &&
                    R_local_or_tmp &&
                    R->firstwrite() == opnum && R->lastwrite() == opnum) {
                    // This local or temp is written only once in the
                    // whole shader -- on this statement -- and it's
                    // assigned a constant.  So just alias it to the
                    // constant.
                    int cind = inst.args()[op.firstarg()+1];
                    symbol_aliases[inst.args()[op.firstarg()]] = cind;
                    inst.args()[op.firstarg()] = cind;
                    R = A;
                    ++changed;
                }
                else if (A->is_const() && R->typespec() == A->typespec() &&
                         R->symtype() == SymTypeOutputParam &&
                         R->firstwrite() == opnum && R->lastwrite() == opnum &&
                         !in_conditional[opnum]) {
                    // This output param is written only once in the
                    // whole shader -- on this statement -- and it's
                    // assigned a constant, and the assignment is
                    // unconditional.  So just alias it to the constant from
                    // here on out.
                    int cind = inst.args()[op.firstarg()+1];
                    symbol_aliases[inst.args()[op.firstarg()]] = cind;
                }
                if (R == A) {
                    // Just an assignment to itself -- turn into NOP!
                    op.reset (ustring("nop"), OP_nop, 0);
                    ++changed;
                } else if (R_local_or_tmp && R->lastread() < opnum) {
                    // Don't bother assigning if we never read it again
                    op.reset (ustring("nop"), OP_nop, 0);
                    ++changed;
                }
            }

            ++opnum;
        }

        totalchanged += changed;
        // info ("Pass %d, changed %d\n", pass, changed);

        // Now that we've rewritten the code, we need to re-track the
        // variable lifetimes.
        allsymptrs.clear ();
        allsymptrs.reserve (inst.symbols().size());
        BOOST_FOREACH (Symbol &s, inst.symbols())
            allsymptrs.push_back (&s);
        {
            SymbolPtrVec oparg_ptrs;
            BOOST_FOREACH (int a, inst.args())
                oparg_ptrs.push_back (inst.symbol (a));
            OSLCompilerImpl::track_variable_lifetimes (inst.ops(), oparg_ptrs, allsymptrs);
        }

        // If only a couple things changed, we know that we almost never get
        // more benefit from another pass, so avoid the expense.
        if (changed < 3)
            break;
    }

    add_useparam (inst.ops(), inst.args(), allsymptrs, inst.m_maincodebegin);
    {
        SymbolPtrVec oparg_ptrs;
        BOOST_FOREACH (int a, inst.args())
            oparg_ptrs.push_back (inst.symbol (a));
        OSLCompilerImpl::track_variable_lifetimes (inst.ops(), oparg_ptrs, allsymptrs);
    }
    if (optimize() >= 1) {
        OSLCompilerImpl::coalesce_temporaries (allsymptrs);
        // coalesce_temporaries may have aliased temps, now we need to
        // make sure all symbol refs are dealiased.
        BOOST_FOREACH (int &arg, inst.m_instargs) {
            Symbol *s = &(inst.symbols()[arg]);
            s = s->dealias ();
            arg = s - &(inst.symbols()[0]);
        }
    }

    // Get rid of nop instructions and unused symbols.
    if (optimize() >= 1) {
        collapse (group, layer, inst);
    } else {
        inst.m_maincodeend = (int)inst.ops().size();
        info ("Processed %s",
              inst.shadername().c_str());
    }

    if (debug())
        std::cerr << "After optimizing: \n" << inst.print() 
                  << "\n--------------------------------\n\n";
}



void
ShadingSystemImpl::collapse (ShaderGroup &group, int layer,
                             ShaderInstance &inst)
{
    //
    // Make a new symbol table that removes all the unused symbols.
    //

    size_t old_nsyms = inst.symbols().size();
    SymbolVec new_symbols;          // buffer for new symbol table
    std::vector<int> symbol_remap;  // mapping of old sym index to new
    int total_syms = 0;             // number of new symbols we'll need

    // First, just count how many we need and set up the mapping
    BOOST_FOREACH (const Symbol &s, inst.symbols()) {
        symbol_remap.push_back (total_syms);
        if (s.everused() ||
              s.symtype() == SymTypeParam || s.symtype() == SymTypeOutputParam)
            ++total_syms;
    }

    // Now make a new table of the right (new) size, and copy the used syms
    new_symbols.reserve (total_syms);
    BOOST_FOREACH (const Symbol &s, inst.symbols()) {
        if (s.everused() ||
              s.symtype() == SymTypeParam || s.symtype() == SymTypeOutputParam)
            new_symbols.push_back (s);
    }

    // Remap all the function arguments to the new indices
    BOOST_FOREACH (int &arg, inst.m_instargs)
        arg = symbol_remap[arg];

    // Fix our connections from upstream shaders
    BOOST_FOREACH (Connection &c, inst.m_connections)
        c.dst.param = symbol_remap[c.dst.param];

    // Fix downstream connections that reference us
    for (int lay = layer+1;  lay < group.nlayers();  ++lay) {
        BOOST_FOREACH (Connection &c, group[lay]->m_connections)
            if (c.srclayer == layer)
                c.src.param = symbol_remap[c.src.param];
    }

    // Miscellaneous cleanup of other things that used symbol indices
    if (inst.m_Psym >= 0)
        inst.m_Psym = symbol_remap[inst.m_Psym];
    if (inst.m_Nsym >= 0)
        inst.m_Nsym = symbol_remap[inst.m_Nsym];
    if (inst.m_lastparam >= 0) {
        inst.m_firstparam = symbol_remap[inst.m_firstparam];
        inst.m_lastparam = symbol_remap[inst.m_lastparam];
    }

    // Swap the new symbol list for the old.
    std::swap (inst.m_instsymbols, new_symbols);


    //
    // Make new code that removes all the nops
    //
    size_t old_nops = inst.ops().size();
    OpcodeVec new_ops;              // buffer for new code
    std::vector<int> op_remap;      // mapping of old opcode indices to new
    int total_ops = 0;              // number of new ops we'll need

    // First, just count how many we need and set up the mapping
    BOOST_FOREACH (const Opcode &op, inst.ops()) {
        op_remap.push_back (total_ops);
        if (op.implementation() != OP_nop)
            ++total_ops;
    }

    // Now make a new table of the right (new) size, copy the used ops, and
    // reset the jump addresses.
    new_ops.reserve (total_ops);
    BOOST_FOREACH (const Opcode &op, inst.ops()) {
        if (op.implementation() != OP_nop) {
            new_ops.push_back (op);
            Opcode &newop (new_ops.back());
            for (int i = 0;  i < (int)Opcode::max_jumps;  ++i)
                if (newop.jump(i) >= 0)
                    newop.jump(i) = op_remap[newop.jump(i)];
        }
    }

    // Miscellaneous cleanup of other things that used instruction addresses
    inst.m_maincodebegin = op_remap[inst.m_maincodebegin];
    inst.m_maincodeend = (int)new_ops.size();

    // Swap the new code for the old.
    std::swap (inst.m_instops, new_ops);

    info ("Optimized %s: New syms %llu/%llu, ops %llu/%llu",
          inst.shadername().c_str(),
          inst.symbols().size(), old_nsyms,
          inst.ops().size(), old_nops);
}



void
ShadingSystemImpl::optimize_group (ShadingAttribState &attribstate, 
                                   ShaderGroup &group)
{
    lock_guard lock (group.m_mutex);
    if (group.optimized())
        return;
    int nlayers = (int) group.nlayers ();
    for (int layer = 0;  layer < nlayers;  ++layer) {
        ShaderInstance *inst = group[layer];
        if (inst) {
            optimize_instance (group, layer, *inst);
            ASSERT (!inst->m_heap_size_calculated);
            inst->m_heap_size_calculated = false;
        }
    }
    attribstate.changed_shaders ();
    group.m_optimized = true;
}



}; // namespace pvt
}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
