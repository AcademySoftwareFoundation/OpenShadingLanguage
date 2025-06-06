// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <cmath>
#include <cstdlib>
#include <vector>

#include <OpenImageIO/fmath.h>
#include <OpenImageIO/sysutil.h>

#include "oslexec_pvt.h"
#include <OSL/dual.h>
#include <OSL/oslnoise.h>
#include "opcolor.h"
#include "runtimeoptimize.h"
using namespace OSL;
using namespace OSL::pvt;


// names of ops we'll be using frequently
static ustring u_aassign("aassign");
static ustring u_add("add");
static ustring u_assign("assign");
static ustring u_cbrt("cbrt");
static ustring u_cell("cell");
static ustring u_cellnoise("cellnoise");
static ustring u_compassign("compassign");
static ustring u_eq("eq");
static ustring u_error("error");
static ustring u_fmt_range_check(
    "Index [%d] out of range %s[0..%d]: %s:%d (group %s, layer %d %s, shader %s)");
static ustring u_fmterror("%s");
static ustring u_if("if");
static ustring u_inversesqrt("inversesqrt");
static ustring u_mul("mul");
static ustring u_mxcompassign("mxcompassign");
static ustring u_nop("nop");
static ustring u_return("return");
static ustring u_sqrt("sqrt");
static ustring u_sub("sub");


OSL_NAMESPACE_BEGIN

namespace pvt {  // OSL::pvt


inline bool
equal_consts(const Symbol& A, const Symbol& B)
{
    return (
        &A == &B
        || (equivalent(A.typespec(), B.typespec())
            && !memcmp(A.data(), B.data(), A.typespec().simpletype().size())));
}



inline bool
unequal_consts(const Symbol& A, const Symbol& B)
{
    return (equivalent(A.typespec(), B.typespec())
            && memcmp(A.data(), B.data(), A.typespec().simpletype().size()));
}



int
constfold_none(RuntimeOptimizer& /*rop*/, int /*opnum*/)
{
    return 0;
}



DECLFOLDER(constfold_add)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& B(*rop.inst()->argsymbol(op.firstarg() + 2));
    if (rop.is_zero(A)) {
        // R = 0 + B  =>   R = B
        rop.turn_into_assign(op, rop.inst()->arg(op.firstarg() + 2),
                             "0 + A => A");
        return 1;
    }
    if (rop.is_zero(B)) {
        // R = A + 0   =>   R = A
        rop.turn_into_assign(op, rop.inst()->arg(op.firstarg() + 1),
                             "A + 0 => A");
        return 1;
    }
    if (A.is_constant() && B.is_constant()) {
        if (A.typespec().is_int() && B.typespec().is_int()) {
            int cind = rop.add_constant(A.get_int() + B.get_int());
            rop.turn_into_assign(op, cind, "const + const");
            return 1;
        } else if (A.typespec().is_float() && B.typespec().is_float()) {
            int cind = rop.add_constant(A.get_float() + B.get_float());
            rop.turn_into_assign(op, cind, "const + const");
            return 1;
        } else if (A.typespec().is_triple_or_float()
                   && B.typespec().is_triple_or_float()) {
            Symbol& R(*rop.inst()->argsymbol(op.firstarg() + 0));
            Vec3 result = A.coerce_vec3() + B.coerce_vec3();
            int cind    = rop.add_constant(R.typespec(), &result);
            rop.turn_into_assign(op, cind, "const + const");
            return 1;
        }
    }
    return 0;
}



DECLFOLDER(constfold_sub)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& B(*rop.inst()->argsymbol(op.firstarg() + 2));
    if (rop.is_zero(B)) {
        // R = A - 0   =>   R = A
        rop.turn_into_assign(op, rop.inst()->arg(op.firstarg() + 1),
                             "A - 0 => A");
        return 1;
    }
    // R = A - B, if both are constants, =>  R = C
    if (A.is_constant() && B.is_constant()) {
        if (A.typespec().is_int() && B.typespec().is_int()) {
            int cind = rop.add_constant(A.get_int() - B.get_int());
            rop.turn_into_assign(op, cind, "const - const");
            return 1;
        } else if (A.typespec().is_float() && B.typespec().is_float()) {
            int cind = rop.add_constant(A.get_float() - B.get_float());
            rop.turn_into_assign(op, cind, "const - const");
            return 1;
        } else if (A.typespec().is_triple_or_float()
                   && B.typespec().is_triple_or_float()) {
            Symbol& R(*rop.inst()->argsymbol(op.firstarg() + 0));
            Vec3 result = A.coerce_vec3() - B.coerce_vec3();
            int cind    = rop.add_constant(R.typespec(), &result);
            rop.turn_into_assign(op, cind, "const - const");
            return 1;
        }
    }
    // R = A - A  =>  R = 0    even if not constant!
    if (&A == &B) {
        rop.turn_into_assign_zero(op, "A - A => 0");
    }
    return 0;
}



DECLFOLDER(constfold_mul)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& B(*rop.inst()->argsymbol(op.firstarg() + 2));
    if (rop.is_one(A)) {
        // R = 1 * B  =>   R = B
        rop.turn_into_assign(op, rop.inst()->arg(op.firstarg() + 2),
                             "1 * A => A");
        return 1;
    }
    if (rop.is_zero(A)) {
        // R = 0 * B  =>   R = 0
        rop.turn_into_assign(op, rop.inst()->arg(op.firstarg() + 1),
                             "0 * A => 0");
        return 1;
    }
    if (rop.is_one(B)) {
        // R = A * 1   =>   R = A
        rop.turn_into_assign(op, rop.inst()->arg(op.firstarg() + 1),
                             "A * 1 => A");
        return 1;
    }
    if (rop.is_zero(B)) {
        // R = A * 0   =>   R = 0
        rop.turn_into_assign(op, rop.inst()->arg(op.firstarg() + 2),
                             "A * 0 => 0");
        return 1;
    }
    if (A.is_constant() && B.is_constant()) {
        if (A.typespec().is_int() && B.typespec().is_int()) {
            int cind = rop.add_constant(A.get_int() * B.get_int());
            rop.turn_into_assign(op, cind, "const * const");
            return 1;
        } else if (A.typespec().is_float() && B.typespec().is_float()) {
            int cind = rop.add_constant(A.get_float() * B.get_float());
            rop.turn_into_assign(op, cind, "const * const");
            return 1;
        } else if (A.typespec().is_triple_or_float()
                   && B.typespec().is_triple_or_float()) {
            Symbol& R(*rop.inst()->argsymbol(op.firstarg() + 0));
            Vec3 result = A.coerce_vec3() * B.coerce_vec3();
            int cind    = rop.add_constant(R.typespec(), &result);
            rop.turn_into_assign(op, cind, "const * const");
            return 1;
        }
    }
    return 0;
}



DECLFOLDER(constfold_div)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& B(*rop.inst()->argsymbol(op.firstarg() + 2));
    if (rop.is_one(B)) {
        // R = A / 1   =>   R = A
        rop.turn_into_assign(op, rop.inst()->arg(op.firstarg() + 1),
                             "A / 1 => A");
        return 1;
    }
    if (rop.is_zero(B)
        && (B.typespec().is_float() || B.typespec().is_triple()
            || B.typespec().is_int())) {
        // R = A / 0   =>   R = 0      because of OSL div by zero rule
        rop.turn_into_assign_zero(op, "A / 0 => 0 (by OSL division rules)");
        return 1;
    }
    if (A.is_constant() && B.is_constant()) {
        int cind = -1;
        if (A.typespec().is_int() && B.typespec().is_int()) {
            cind = rop.add_constant(A.get_int() / B.get_int());
        } else if (A.typespec().is_int_or_float()
                   && B.typespec().is_int_or_float()) {
            cind = rop.add_constant(A.coerce_float() / B.coerce_float());
        } else if (A.typespec().is_triple_or_float()
                   && B.typespec().is_triple_or_float()) {
            Symbol& R(*rop.inst()->argsymbol(op.firstarg() + 0));
            Vec3 result = A.coerce_vec3() / B.coerce_vec3();
            cind        = rop.add_constant(R.typespec(), &result);
        }
        if (cind >= 0) {
            rop.turn_into_assign(op, cind, "const / const");
            return 1;
        }
    }
    return 0;
}



DECLFOLDER(constfold_mod)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& B(*rop.inst()->argsymbol(op.firstarg() + 2));

    if (rop.is_zero(A)) {
        // R = 0 % B  =>   R = 0
        rop.turn_into_assign_zero(op, "0 % A => 0");
        return 1;
    }
    if (rop.is_zero(B)) {
        // R = A % 0   =>   R = 0
        rop.turn_into_assign_zero(op, "A % 0 => 0");
        return 1;
    }
    if (A.is_constant() && B.is_constant() && A.typespec().is_int()
        && B.typespec().is_int()) {
        int a    = A.get_int();
        int b    = B.get_int();
        int cind = rop.add_constant(b ? (a % b) : 0);
        rop.turn_into_assign(op, cind, "const % const");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_dot)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& B(*rop.inst()->argsymbol(op.firstarg() + 2));

    // Dot with (0,0,0) -> 0
    if (rop.is_zero(A) || rop.is_zero(B)) {
        rop.turn_into_assign_zero(op, "dot(a,(0,0,0)) => 0");
        return 1;
    }

    // dot(const,const) -> const
    if (A.is_constant() && B.is_constant()) {
        OSL_DASSERT(A.typespec().is_triple() && B.typespec().is_triple());
        float result = A.get_vec3().dot(B.get_vec3());
        int cind     = rop.add_constant(TypeFloat, &result);
        rop.turn_into_assign(op, cind, "dot(const,const)");
        return 1;
    }

    return 0;
}



DECLFOLDER(constfold_neg)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    if (A.is_constant()) {
        if (A.typespec().is_int()) {
            int result = -A.get_int();
            int cind   = rop.add_constant(A.typespec(), &result);
            rop.turn_into_assign(op, cind, "-const");
            return 1;
        } else if (A.typespec().is_float()) {
            float result = -A.get_float();
            int cind     = rop.add_constant(A.typespec(), &result);
            rop.turn_into_assign(op, cind, "-const");
            return 1;
        } else if (A.typespec().is_triple()) {
            Vec3 result = -A.get_vec3();
            int cind    = rop.add_constant(A.typespec(), &result);
            rop.turn_into_assign(op, cind, "-const");
            return 1;
        }
    }
    return 0;
}



DECLFOLDER(constfold_abs)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    if (A.is_constant()) {
        if (A.typespec().is_int()) {
            int result = std::abs(A.get_int());
            int cind   = rop.add_constant(A.typespec(), &result);
            rop.turn_into_assign(op, cind, "abs(const)");
            return 1;
        } else if (A.typespec().is_float()) {
            float result = std::abs(A.get_float());
            int cind     = rop.add_constant(A.typespec(), &result);
            rop.turn_into_assign(op, cind, "abs(const)");
            return 1;
        } else if (A.typespec().is_triple()) {
            Vec3 result = A.get_vec3();
            result.x    = std::abs(result.x);
            result.y    = std::abs(result.y);
            result.z    = std::abs(result.z);
            int cind    = rop.add_constant(A.typespec(), &result);
            rop.turn_into_assign(op, cind, "abs(const)");
            return 1;
        }
    }
    return 0;
}



DECLFOLDER(constfold_eq)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& B(*rop.inst()->argsymbol(op.firstarg() + 2));
    if (A.is_constant() && B.is_constant()) {
        bool val = false;
        if (equivalent(A.typespec(), B.typespec())) {
            val = equal_consts(A, B);
        } else if (A.typespec().is_float() && B.typespec().is_int()) {
            val = (A.get_float() == B.get_int());
        } else if (A.typespec().is_int() && B.typespec().is_float()) {
            val = (A.get_int() == B.get_float());
        } else {
            return 0;  // unhandled cases
        }
        // Turn the 'eq R A B' into 'assign R X' where X is 0 or 1.
        static const int int_zero = 0, int_one = 1;
        int cind = rop.add_constant(TypeInt, val ? &int_one : &int_zero);
        rop.turn_into_assign(op, cind, "const == const");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_neq)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& B(*rop.inst()->argsymbol(op.firstarg() + 2));
    if (A.is_constant() && B.is_constant()) {
        bool val = false;
        if (equivalent(A.typespec(), B.typespec())) {
            val = !equal_consts(A, B);
        } else if (A.typespec().is_float() && B.typespec().is_int()) {
            val = (A.get_float() != B.get_int());
        } else if (A.typespec().is_int() && B.typespec().is_float()) {
            val = (A.get_int() != B.get_float());
        } else {
            return 0;  // unhandled case
        }
        // Turn the 'neq R A B' into 'assign R X' where X is 0 or 1.
        static const int int_zero = 0, int_one = 1;
        int cind = rop.add_constant(TypeInt, val ? &int_one : &int_zero);
        rop.turn_into_assign(op, cind, "const != const");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_lt)
{
    static const int int_zero = 0, int_one = 1;
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& B(*rop.inst()->argsymbol(op.firstarg() + 2));
    const TypeSpec& ta(A.typespec());
    const TypeSpec& tb(B.typespec());
    if (A.is_constant() && B.is_constant()) {
        // Turn the 'leq R A B' into 'assign R X' where X is 0 or 1.
        bool val = false;
        if (ta.is_float() && tb.is_float()) {
            val = (A.get_float() < B.get_float());
        } else if (ta.is_float() && tb.is_int()) {
            val = (A.get_float() < B.get_int());
        } else if (ta.is_int() && tb.is_float()) {
            val = (A.get_int() < B.get_float());
        } else if (ta.is_int() && tb.is_int()) {
            val = (A.get_int() < B.get_int());
        } else {
            return 0;  // unhandled case
        }
        int cind = rop.add_constant(TypeInt, val ? &int_one : &int_zero);
        rop.turn_into_assign(op, cind, "const < const");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_le)
{
    static const int int_zero = 0, int_one = 1;
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& B(*rop.inst()->argsymbol(op.firstarg() + 2));
    const TypeSpec& ta(A.typespec());
    const TypeSpec& tb(B.typespec());
    if (A.is_constant() && B.is_constant()) {
        // Turn the 'leq R A B' into 'assign R X' where X is 0 or 1.
        bool val = false;
        if (ta.is_float() && tb.is_float()) {
            val = (A.get_float() <= B.get_float());
        } else if (ta.is_float() && tb.is_int()) {
            val = (A.get_float() <= B.get_int());
        } else if (ta.is_int() && tb.is_float()) {
            val = (A.get_int() <= B.get_float());
        } else if (ta.is_int() && tb.is_int()) {
            val = (A.get_int() <= B.get_int());
        } else {
            return 0;  // unhandled case
        }
        int cind = rop.add_constant(TypeInt, val ? &int_one : &int_zero);
        rop.turn_into_assign(op, cind, "const <= const");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_gt)
{
    static const int int_zero = 0, int_one = 1;
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& B(*rop.inst()->argsymbol(op.firstarg() + 2));
    const TypeSpec& ta(A.typespec());
    const TypeSpec& tb(B.typespec());
    if (A.is_constant() && B.is_constant()) {
        // Turn the 'gt R A B' into 'assign R X' where X is 0 or 1.
        bool val = false;
        if (ta.is_float() && tb.is_float()) {
            val = (A.get_float() > B.get_float());
        } else if (ta.is_float() && tb.is_int()) {
            val = (A.get_float() > B.get_int());
        } else if (ta.is_int() && tb.is_float()) {
            val = (A.get_int() > B.get_float());
        } else if (ta.is_int() && tb.is_int()) {
            val = (A.get_int() > B.get_int());
        } else {
            return 0;  // unhandled case
        }
        int cind = rop.add_constant(TypeInt, val ? &int_one : &int_zero);
        rop.turn_into_assign(op, cind, "const > const");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_ge)
{
    static const int int_zero = 0, int_one = 1;
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& B(*rop.inst()->argsymbol(op.firstarg() + 2));
    const TypeSpec& ta(A.typespec());
    const TypeSpec& tb(B.typespec());
    if (A.is_constant() && B.is_constant()) {
        // Turn the 'leq R A B' into 'assign R X' where X is 0 or 1.
        bool val = false;
        if (ta.is_float() && tb.is_float()) {
            val = (A.get_float() >= B.get_float());
        } else if (ta.is_float() && tb.is_int()) {
            val = (A.get_float() >= B.get_int());
        } else if (ta.is_int() && tb.is_float()) {
            val = (A.get_int() >= B.get_float());
        } else if (ta.is_int() && tb.is_int()) {
            val = (A.get_int() >= B.get_int());
        } else {
            return 0;  // unhandled case
        }
        int cind = rop.add_constant(TypeInt, val ? &int_one : &int_zero);
        rop.turn_into_assign(op, cind, "const >= const");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_or)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& B(*rop.inst()->argsymbol(op.firstarg() + 2));
    if (A.is_constant() && B.is_constant()) {
        OSL_DASSERT(A.typespec().is_int() && B.typespec().is_int());
        bool val = A.get_int() || B.get_int();
        // Turn the 'or R A B' into 'assign R X' where X is 0 or 1.
        static const int int_zero = 0, int_one = 1;
        int cind = rop.add_constant(TypeInt, val ? &int_one : &int_zero);
        rop.turn_into_assign(op, cind, "const || const");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_and)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& B(*rop.inst()->argsymbol(op.firstarg() + 2));
    if (A.is_constant() && B.is_constant()) {
        // Turn the 'and R A B' into 'assign R X' where X is 0 or 1.
        OSL_DASSERT(A.typespec().is_int() && B.typespec().is_int());
        bool val                  = A.get_int() && B.get_int();
        static const int int_zero = 0, int_one = 1;
        int cind = rop.add_constant(TypeInt, val ? &int_one : &int_zero);
        rop.turn_into_assign(op, cind, "const && const");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_bitand)
{
    Opcode& op(rop.op(opnum));
    Symbol& A(*rop.opargsym(op, 1));
    Symbol& B(*rop.opargsym(op, 2));
    if (A.is_constant() && B.is_constant()) {
        // Turn the 'bitand R A B' into 'assign R X'.
        OSL_DASSERT(A.typespec().is_int() && B.typespec().is_int());
        int cind = rop.add_constant(A.get_int() & B.get_int());
        rop.turn_into_assign(op, cind, "const & const");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_bitor)
{
    Opcode& op(rop.op(opnum));
    Symbol& A(*rop.opargsym(op, 1));
    Symbol& B(*rop.opargsym(op, 2));
    if (A.is_constant() && B.is_constant()) {
        // Turn the 'bitor R A B' into 'assign R X'.
        OSL_DASSERT(A.typespec().is_int() && B.typespec().is_int());
        int cind = rop.add_constant(A.get_int() | B.get_int());
        rop.turn_into_assign(op, cind, "const | const");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_xor)
{
    Opcode& op(rop.op(opnum));
    Symbol& A(*rop.opargsym(op, 1));
    Symbol& B(*rop.opargsym(op, 2));
    if (A.is_constant() && B.is_constant()) {
        // Turn the 'xor R A B' into 'assign R X'.
        OSL_DASSERT(A.typespec().is_int() && B.typespec().is_int());
        int cind = rop.add_constant(A.get_int() ^ B.get_int());
        rop.turn_into_assign(op, cind, "const ^ const");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_compl)
{
    Opcode& op(rop.op(opnum));
    Symbol& A(*rop.opargsym(op, 1));
    if (A.is_constant()) {
        // Turn the 'compl R A' into 'assign R X'.
        OSL_DASSERT(A.typespec().is_int());
        int cind = rop.add_constant(~(A.get_int()));
        rop.turn_into_assign(op, cind, "~const");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_if)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& C(*rop.inst()->argsymbol(op.firstarg() + 0));
    if (C.is_constant()) {
        int result = -1;  // -1 == we don't know
        if (C.typespec().is_int())
            result = (C.get_int() != 0);
        else if (C.typespec().is_float())
            result = (C.get_float() != 0.0f);
        else if (C.typespec().is_triple())
            result = (C.get_vec3() != Vec3(0, 0, 0));
        else if (C.typespec().is_string()) {
            ustring s = C.get_string();
            result    = (s.length() != 0);
        }
        int changed = 0;
        if (result > 0) {
            changed += rop.turn_into_nop(op.jump(0), op.jump(1),
                                         "elide 'else'");
            changed += rop.turn_into_nop(op, "elide 'else'");
        } else if (result == 0) {
            changed += rop.turn_into_nop(opnum, op.jump(0), "elide 'if'");
        }
        return changed;
    }

    // Eliminate 'if' that contains no statements to execute
    int jump       = op.farthest_jump();
    bool only_nops = true;
    for (int i = opnum + 1; i < jump && only_nops; ++i)
        only_nops &= (rop.inst()->ops()[i].opname() == u_nop);
    if (only_nops) {
        rop.turn_into_nop(op, "'if' with no body");
        return 1;
    }

    return 0;
}



// Is an array known to have all elements having the same value?
static bool
array_all_elements_equal(const Symbol& s)
{
    TypeDesc t  = s.typespec().simpletype();
    size_t size = t.elementsize();
    size_t n    = t.numelements();
    for (size_t i = 1; i < n; ++i)
        if (memcmp((const char*)s.data(), (const char*)s.data() + i * size,
                   size))
            return false;
    return true;
}



DECLFOLDER(constfold_aref)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& R(*rop.inst()->argsymbol(op.firstarg() + 0));
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& Index(*rop.inst()->argsymbol(op.firstarg() + 2));
    OSL_DASSERT(A.typespec().is_array() && Index.typespec().is_int());

    // Try to turn R=A[I] into R=C if A and I are const.
    if (A.is_constant() && Index.is_constant()) {
        TypeSpec elemtype = A.typespec().elementtype();
        OSL_ASSERT(equivalent(elemtype, R.typespec()));
        const int length     = A.typespec().arraylength();
        const int orig_index = Index.get_int(),
                  index      = OIIO::clamp(orig_index, 0, length - 1);
        OSL_DASSERT(index >= 0 && index < length);
        int cind = rop.add_constant(elemtype,
                                    (char*)A.data()
                                        + index * elemtype.simpletype().size());
        rop.turn_into_assign(op, cind, "aref const fold: const_array[const]");
        if (rop.inst()->master()->range_checking() && index != orig_index) {
            // the original index was out of range, and the user cares about reporting errors
            const int args_to_add[]
                = { rop.add_constant(u_fmt_range_check),
                    rop.add_constant(orig_index),
                    rop.add_constant(A.unmangled()),
                    rop.add_constant(length - 1),
                    rop.add_constant(op.sourcefile()),
                    rop.add_constant(op.sourceline()),
                    rop.add_constant(rop.group().name()),
                    rop.add_constant(rop.layer()),
                    rop.add_constant(rop.inst()->layername()),
                    rop.add_constant(ustring(rop.inst()->shadername())) };
            rop.insert_code(opnum, u_error, args_to_add,
                            RuntimeOptimizer::RecomputeRWRanges,
                            RuntimeOptimizer::GroupWithNext);
            Opcode& newop(rop.inst()->ops()[opnum]);
            newop.argreadonly(0);
        }
        return 1;
    }
    // Even if the index isn't constant, we still know the answer if all
    // the array elements are equal!
    if (A.is_constant() && array_all_elements_equal(A)) {
        TypeSpec elemtype = A.typespec().elementtype();
        OSL_ASSERT(equivalent(elemtype, R.typespec()));
        int cind = rop.add_constant(elemtype, (char*)A.data());
        rop.turn_into_assign(op, cind, "aref of elements-equal array");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_arraylength)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_MAYBE_UNUSED Symbol& R(*rop.inst()->argsymbol(op.firstarg() + 0));
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    OSL_DASSERT(R.typespec().is_int() && A.typespec().is_array());

    // Try to turn R=arraylength(A) into R=C if the array length is known
    int len = A.typespec().is_unsized_array() ? A.initializers()
                                              : A.typespec().arraylength();
    if (len > 0) {
        int cind = rop.add_constant(TypeSpec(TypeDesc::INT), &len);
        rop.turn_into_assign(op, cind, "const fold arraylength");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_aassign)
{
    // Array assignment
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol* R(rop.inst()->argsymbol(op.firstarg() + 0));
    Symbol* I(rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol* C(rop.inst()->argsymbol(op.firstarg() + 2));
    if (!I->is_constant() || !C->is_constant())
        return 0;  // not much we can do if not assigning constants
    OSL_DASSERT(R->typespec().is_array() && I->typespec().is_int());

    TypeSpec elemtype = R->typespec().elementtype();
    if (elemtype.is_closure())
        return 0;  // don't worry about closures
    TypeDesc elemsimpletype = elemtype.simpletype();

    // Look for patterns where all array elements are assigned in
    // succession within the same block, in which case we can turn the
    // result into a constant!
    int len = R->typespec().arraylength();
    if (len <= 0)
        return 0;  // don't handle arrays of unknown length
    int elemsize = (int)elemsimpletype.size();
    std::vector<int> index_assigned(len, -1);
    std::vector<char> filled_values(elemsize * len);  // constant storage
    char* fill       = (char*)&filled_values[0];
    int num_assigned = 0;
    int opindex      = opnum;
    int highestop    = opindex;
    for (;;) {
        Opcode& opi(rop.inst()->ops()[opindex]);
        if (opi.opname() != u_aassign)
            break;  // not a successive aassign op
        Symbol* Ri(rop.inst()->argsymbol(opi.firstarg() + 0));
        if (Ri != R)
            break;  // not a compassign to the same variable
        Symbol* Ii(rop.inst()->argsymbol(opi.firstarg() + 1));
        Symbol* Ci(rop.inst()->argsymbol(opi.firstarg() + 2));
        if (!Ii->is_constant() || !Ci->is_constant())
            break;  // not assigning constants
        int indexval = Ii->get_int();
        if (indexval < 0 || indexval >= len)
            break;  // out of range index; let runtime deal with it
        if (equivalent(elemtype, Ci->typespec())) {
            // equivalent types
            memcpy(fill + indexval * elemsize, Ci->data(), elemsize);
        } else if (elemtype.is_float() && Ci->typespec().is_int()) {
            // special case of float[i] = int
            ((float*)fill)[indexval] = Ci->coerce_float();
        } else {
            break;  // a case we don't handle
        }
        if (index_assigned[indexval] < 0)
            ++num_assigned;
        index_assigned[indexval] = opindex;
        highestop                = opindex;
        opindex                  = rop.next_block_instruction(opindex);
        if (!opindex)
            break;
    }
    if (num_assigned == len) {
        // woo-hoo! we had a succession of constant aassign ops to the
        // same variable, filling in all indices. Turn the whole shebang
        // into a single assignment.
        int cind = rop.add_constant(R->typespec(), fill);
        rop.turn_into_assign(op, cind,
                             "replaced element-by-element assignment");
        rop.turn_into_nop(opnum + 1, highestop + 1,
                          "replaced element-by-element assignment");
        return highestop + 1 - opnum;
    }

    return 0;
}



DECLFOLDER(constfold_compassign)
{
    // Component assignment
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol* R(rop.inst()->argsymbol(op.firstarg() + 0));
    Symbol* I(rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol* C(rop.inst()->argsymbol(op.firstarg() + 2));
    if (!I->is_constant() || !C->is_constant())
        return 0;  // not much we can do if not assigning constants
    OSL_DASSERT(R->typespec().is_triple() && I->typespec().is_int()
                && (C->typespec().is_float() || C->typespec().is_int()));

    // We are obviously not assigning to a constant, but it could be
    // that at this point in our current block, the value of A is known,
    // and that will show up as a block alias.
    int Aalias = rop.block_alias(rop.inst()->arg(op.firstarg() + 0));
    Symbol* AA = rop.inst()->symbol(Aalias);
    // N.B. symbol returns NULL if Aalias is < 0

    // Try to simplify A[I]=C if we already know the old value of A as a
    // constant. We can turn it into A[I] = N, where N is the old A but with
    // the Ith component set to C. If it turns out that the old A[I] == C,
    // and thus the assignment doesn't change A's value, we can eliminate
    // the assignment entirely.
    if (AA && AA->is_constant()) {
        OSL_DASSERT(AA->typespec().is_triple());
        int index = I->get_int();
        if (index < 0 || index >= 3) {
            // We are indexing a const triple out of range.  But this
            // isn't necessarily a reportable error, because it may be a
            // code path that will never be taken.  Punt -- don't
            // optimize this op, leave it to the execute-time range
            // check to catch, if indeed it is a problem.
            return 0;
        }
        float* aa = (float*)AA->data();
        float c   = C->coerce_float();
        if (aa[index] == c) {
            // If the component assignment doesn't change that component,
            // just omit the op entirely.
            rop.turn_into_nop(op, "useless compassign");
            return 1;
        }
        // If the previous value of the triple was a constant, and we're
        // assigning a new constant to one component (and the index is
        // also a constant), just turn it into an assignment of a new
        // constant triple.
        Vec3 newval(aa[0], aa[1], aa[2]);
        newval[index] = c;
        int cind      = rop.add_constant(AA->typespec(), &newval);
        rop.turn_into_assign(op, cind, "fold compassign");
        return 1;
    }

    // Look for patterns where all three components are assigned in
    // succession within the same block, in which case we can turn the
    // result into a constant!
    int index_assigned[3] = { -1, -1, -1 };
    float filled_values[3];
    int num_assigned = 0;
    int opindex      = opnum;
    int highestop    = opindex;
    for (;;) {
        Opcode& opi(rop.inst()->ops()[opindex]);
        if (opi.opname() != u_compassign)
            break;  // not a successive compassign op
        Symbol* Ri(rop.inst()->argsymbol(opi.firstarg() + 0));
        if (Ri != R)
            break;  // not a compassign to the same variable
        Symbol* Ii(rop.inst()->argsymbol(opi.firstarg() + 1));
        Symbol* Ci(rop.inst()->argsymbol(opi.firstarg() + 2));
        if (!Ii->is_constant() || !Ci->is_constant())
            break;  // not assigning constants
        int indexval = Ii->get_int();
        if (indexval < 0 || indexval >= 3)
            break;  // out of range index; let runtime deal with it
        filled_values[indexval] = Ci->coerce_float();
        if (index_assigned[indexval] < 0)
            ++num_assigned;
        index_assigned[indexval] = opindex;
        highestop                = opindex;
        opindex                  = rop.next_block_instruction(opindex);
        if (!opindex)
            break;
    }
    if (num_assigned == 3) {
        // woo-hoo! we had a succession of constant compassign ops to the
        // same variable, filling in all indices. Turn the whole shebang
        // into a single assignment.
        int cind = rop.add_constant(R->typespec(), filled_values);
        rop.turn_into_assign(op, cind,
                             "replaced element-by-element assignment");
        rop.turn_into_nop(opnum + 1, highestop + 1,
                          "replaced element-by-element assignment");
        return highestop + 1 - opnum;
    }

    return 0;
}



DECLFOLDER(constfold_mxcompassign)
{
    // Matrix component assignment
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol* R(rop.inst()->argsymbol(op.firstarg() + 0));
    Symbol* J(rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol* I(rop.inst()->argsymbol(op.firstarg() + 2));
    Symbol* C(rop.inst()->argsymbol(op.firstarg() + 3));
    if (!J->is_constant() || !I->is_constant() || !C->is_constant())
        return 0;  // not much we can do if not assigning constants
    OSL_DASSERT(R->typespec().is_matrix() && J->typespec().is_int()
                && I->typespec().is_int()
                && (C->typespec().is_float() || C->typespec().is_int()));

    // We are obviously not assigning to a constant, but it could be
    // that at this point in our current block, the value of A is known,
    // and that will show up as a block alias.
    int Aalias = rop.block_alias(rop.inst()->arg(op.firstarg() + 0));
    Symbol* AA = rop.inst()->symbol(Aalias);
    // N.B. symbol returns NULL if Aalias is < 0

    // Try to simplify A[J,I]=C if we already know the old value of A as a
    // constant. We can turn it into A[J,I] = N, where N is the old A but with
    // the designated component set to C. If it turns out that the old
    // A[J,I] == C, and thus the assignment doesn't change A's value, we can
    // eliminate the assignment entirely.
    if (AA && AA->is_constant()) {
        OSL_DASSERT(AA->typespec().is_matrix());
        int jndex = J->get_int();
        int index = I->get_int();
        if (index < 0 || index >= 3 || jndex < 0 || jndex >= 3) {
            // We are indexing a const matrix out of range.  But this
            // isn't necessarily a reportable error, because it may be a
            // code path that will never be taken.  Punt -- don't
            // optimize this op, leave it to the execute-time range
            // check to catch, if indeed it is a problem.
            return 0;
        }
        Matrix44* aa = (Matrix44*)AA->dataptr();
        float c      = C->coerce_float();
        if ((*aa)[jndex][index] == c) {
            // If the component assignment doesn't change that component,
            // just omit the op entirely.
            rop.turn_into_nop(op, "useless mxcompassign");
            return 1;
        }
        // If the previous value of the matrix was a constant, and we're
        // assigning a new constant to one component (and the index is
        // also a constant), just turn it into an assignment of a new
        // constant triple.
        Matrix44 newval      = *aa;
        newval[jndex][index] = c;
        int cind             = rop.add_constant(AA->typespec(), &newval);
        rop.turn_into_assign(op, cind, "fold mxcompassign");
        return 1;
    }

    // Look for patterns where all 16 components are assigned in
    // succession within the same block, in which case we can turn the
    // result into a constant!
    int index_assigned[4][4] = { { -1, -1, -1, -1 },
                                 { -1, -1, -1, -1 },
                                 { -1, -1, -1, -1 },
                                 { -1, -1, -1, -1 } };
    float filled_values[4][4];
    int num_assigned = 0;
    int opindex      = opnum;
    int highestop    = opindex;
    for (;;) {
        Opcode& opi(rop.inst()->ops()[opindex]);
        if (opi.opname() != u_mxcompassign)
            break;  // not a successive mxcompassign op
        Symbol* Ri(rop.inst()->argsymbol(opi.firstarg() + 0));
        if (Ri != R)
            break;  // not a mxcompassign to the same variable
        Symbol* Ji(rop.inst()->argsymbol(opi.firstarg() + 1));
        Symbol* Ii(rop.inst()->argsymbol(opi.firstarg() + 2));
        Symbol* Ci(rop.inst()->argsymbol(opi.firstarg() + 3));
        if (!Ji->is_constant() || !Ii->is_constant() || !Ci->is_constant())
            break;  // not assigning constants
        int jndexval = Ji->get_int();
        int indexval = Ii->get_int();
        if (jndexval < 0 || jndexval >= 4 || indexval < 0 || indexval >= 4)
            break;  // out of range index; let runtime deal with it
        float c                           = Ci->coerce_float();
        filled_values[jndexval][indexval] = c;
        if (index_assigned[jndexval][indexval] < 0)
            ++num_assigned;
        index_assigned[jndexval][indexval] = opindex;
        highestop                          = opindex;
        opindex = rop.next_block_instruction(opindex);
        if (!opindex)
            break;
    }
    if (num_assigned == 16) {
        // woo-hoo! we had a succession of constant mxcompassign ops to the
        // same variable, filling in all indices. Turn the whole shebang
        // into a single assignment.
        int cind = rop.add_constant(R->typespec(), filled_values);
        rop.turn_into_assign(op, cind,
                             "replaced element-by-element assignment");
        rop.turn_into_nop(opnum + 1, highestop + 1,
                          "replaced element-by-element assignment");
        return highestop + 1 - opnum;
    }

    return 0;
}



DECLFOLDER(constfold_compref)
{
    // Component reference
    // Try to turn R=A[I] into R=C if A and I are const.
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& Index(*rop.inst()->argsymbol(op.firstarg() + 2));
    if (A.is_constant() && Index.is_constant()) {
        OSL_DASSERT(A.typespec().is_triple() && Index.typespec().is_int());
        int index = Index.get_int();
        if (index < 0 || index >= 3) {
            // We are indexing a const triple out of range.  But this
            // isn't necessarily a reportable error, because it may be a
            // code path that will never be taken.  Punt -- don't
            // optimize this op, leave it to the execute-time range
            // check to catch, if indeed it is a problem.
            return 0;
        }
        int cind = rop.add_constant(A.get_float(index));
        rop.turn_into_assign(op, cind, "const_triple[const]");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_strlen)
{
    // Try to turn R=strlen(s) into R=C
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& S(*rop.inst()->argsymbol(op.firstarg() + 1));
    if (S.is_constant()) {
        OSL_DASSERT(S.typespec().is_string());
        int result = (int)(S.get_string()).length();
        int cind   = rop.add_constant(result);
        rop.turn_into_assign(op, cind, "const fold strlen");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_hash)
{
    // Try to turn R=hash(s) into R=C
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol* S(rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol* T(op.nargs() > 2 ? rop.inst()->argsymbol(op.firstarg() + 2) : NULL);
    if (S->is_constant() && (T == NULL || T->is_constant())) {
        int cind = -1;
        if (S->typespec().is_string()) {
            cind = rop.add_constant((int)(S->get_string()).hash());
        } else if (op.nargs() == 2 && S->typespec().is_int()) {
            cind = rop.add_constant(inthashi(S->get_int()));
        } else if (op.nargs() == 2 && S->typespec().is_float()) {
            cind = rop.add_constant(inthashf(S->get_float()));
        } else if (op.nargs() == 3 && S->typespec().is_float()
                   && T->typespec().is_float()) {
            cind = rop.add_constant(inthashf(S->get_float(), T->get_float()));
        } else if (op.nargs() == 2 && S->typespec().is_triple()) {
            cind = rop.add_constant(inthashf((const float*)S->dataptr()));
        } else if (op.nargs() == 3 && S->typespec().is_triple()
                   && T->typespec().is_float()) {
            cind = rop.add_constant(
                inthashf((const float*)S->dataptr(), T->get_float()));
        }
        if (cind >= 0) {
            rop.turn_into_assign(op, cind, "const fold hash");
            return 1;
        }
    }
    return 0;
}



DECLFOLDER(constfold_getchar)
{
    // Try to turn R=getchar(s,i) into R=C
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& S(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& I(*rop.inst()->argsymbol(op.firstarg() + 2));
    if (S.is_constant() && I.is_constant()) {
        OSL_DASSERT(S.typespec().is_string() && I.typespec().is_int());
        int idx    = I.get_int();
        int len    = (int)(S.get_string()).length();
        int result = idx >= 0 && idx < len ? (S.get_string()).c_str()[idx] : 0;
        int cind   = rop.add_constant(result);
        rop.turn_into_assign(op, cind, "const fold getchar");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_endswith)
{
    // Try to turn R=endswith(s,e) into R=C
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& S(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& E(*rop.inst()->argsymbol(op.firstarg() + 2));
    if (S.is_constant() && E.is_constant()) {
        OSL_DASSERT(S.typespec().is_string() && E.typespec().is_string());
        ustring s = S.get_string();
        ustring e = E.get_string();
        int cind  = rop.add_constant(OIIO::Strutil::ends_with(s, e) ? 1 : 0);
        rop.turn_into_assign(op, cind, "const fold endswith");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_startswith)
{
    // Try to turn R=startswith(s,e) into R=C
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& S(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& E(*rop.inst()->argsymbol(op.firstarg() + 2));
    if (S.is_constant() && E.is_constant()) {
        OSL_DASSERT(S.typespec().is_string() && E.typespec().is_string());
        ustring s = S.get_string();
        ustring e = E.get_string();
        int cind  = rop.add_constant(OIIO::Strutil::starts_with(s, e) ? 1 : 0);
        rop.turn_into_assign(op, cind, "const fold startswith");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_stoi)
{
    // Try to turn R=stoi(s) into R=C
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& S(*rop.inst()->argsymbol(op.firstarg() + 1));
    if (S.is_constant()) {
        OSL_DASSERT(S.typespec().is_string());
        ustring s = S.get_string();
        int cind  = rop.add_constant(Strutil::from_string<int>(s));
        rop.turn_into_assign(op, cind, "const fold stoi");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_stof)
{
    // Try to turn R=stof(s) into R=C
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& S(*rop.inst()->argsymbol(op.firstarg() + 1));
    if (S.is_constant()) {
        OSL_DASSERT(S.typespec().is_string());
        ustring s = S.get_string();
        int cind  = rop.add_constant(Strutil::from_string<float>(s));
        rop.turn_into_assign(op, cind, "const fold stof");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_split)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    // Symbol &R (*rop.inst()->argsymbol(op.firstarg()+0));
    Symbol& Str(*rop.opargsym(op, 1));
    Symbol& Results(*rop.opargsym(op, 2));
    Symbol* Sep(rop.opargsym(op, 3));
    Symbol* Maxsplit(rop.opargsym(op, 4));
    if (Str.is_constant() && (!Sep || Sep->is_constant())
        && (!Maxsplit || Maxsplit->is_constant())) {
        // The split string, separator string, and maxsplit are all constants.
        // Compute the results with Strutil::split.
        int resultslen = Results.typespec().arraylength();
        int maxsplit   = Maxsplit ? Maxsplit->get_int() : resultslen;
        maxsplit       = std::min(maxsplit, resultslen);
        std::vector<std::string> splits;
        ustring sep = Sep ? Sep->get_string() : ustring("");
        Strutil::split(Str.get_string(), splits, sep.string(), maxsplit);
        int n = std::min(std::max(0, maxsplit), (int)splits.size());
        // Temporarily stash the index of the symbol holding results
        int resultsarg = rop.inst()->args()[op.firstarg() + 2];
        // Turn the 'split' into a straight assignment of the return value...
        rop.turn_into_assign(op, rop.add_constant(n));
        // Create a constant array holding the split results
        std::vector<ustring> usplits(resultslen);
        for (int i = 0; i < n; ++i)
            usplits[i] = ustring(splits[i]);
        int cind = rop.add_constant(TypeDesc(TypeDesc::STRING, resultslen),
                                    usplits.data());
        // And insert an instruction copying our constant array to the
        // user's results array.
        const int args[] = { resultsarg, cind };
        rop.insert_code(opnum, u_assign, args,
                        RuntimeOptimizer::RecomputeRWRanges,
                        RuntimeOptimizer::GroupWithNext);
        return 1;
    }

    return 0;
}



DECLFOLDER(constfold_concat)
{
    // Try to turn R=concat(s,...) into R=C
    Opcode& op(rop.inst()->ops()[opnum]);
    ustring result;
    for (int i = 1; i < op.nargs(); ++i) {
        Symbol& S(*rop.inst()->argsymbol(op.firstarg() + i));
        if (!S.is_constant())
            return 0;  // something non-constant
        ustring old = result;
        ustring s   = S.get_string();
        result      = ustring::fmtformat("{}{}", old, s);
    }
    // If we made it this far, all args were constants, and the
    // concatenation is in result.
    int cind = rop.add_constant(TypeString, &result);
    rop.turn_into_assign(op, cind, "const fold concat");
    return 1;
}



DECLFOLDER(constfold_format)
{
    // Try to turn R=format(fmt,...) into R=C
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Format(*rop.opargsym(op, 1));
    if (!Format.is_constant())
        return 0;
    ustring fmt = Format.get_string();

    // split fmt into the prefix (the starting part of the string that we
    // haven't yet processed) and the suffix (the ending part that we've
    // fully processed).
    std::string prefix = fmt.string();
    std::string suffix;
    int args_expanded = 0;

    // While there is still a constant argument at the end of the arg list,
    // peel it off and use it to rewrite the format string.
    for (int argnum = op.nargs() - 1; argnum >= 2; --argnum) {
        Symbol& Arg(*rop.opargsym(op, argnum));
        if (!Arg.is_constant())
            break;  // no more constants

        // find the last format specification
        size_t pos = std::string::npos;
        while (1) {
            pos = prefix.find_last_of('%', pos);  // find at or before pos
            if (pos == std::string::npos) {
                // Fewer '%' tokens than arguments? Must be malformed. Punt.
                return 0;
            }
            if (pos == 0 || prefix[pos - 1] != '%') {
                // we found the format specifier
                break;
            }
            // False alarm! Beware of %% which is a literal % rather than a
            // format specifier. Back up and try again.
            if (pos >= 2)
                pos -= 2;  // back up
            else {
                // This can only happen if the %% is at the start of the
                // format string, but it shouldn't be since there are still
                // args to process. Punt.
                return 0;
            }
        }
        OSL_ASSERT(pos < prefix.length() && prefix[pos] == '%');

        // cleave off the last format specification into mid
        std::string mid = std::string(prefix, pos);
        std::string formatted;
        const TypeSpec& argtype = Arg.typespec();
        if (argtype.is_int())
            formatted = Strutil::sprintf(mid.c_str(), Arg.get_int());
        else if (argtype.is_float())
            formatted = Strutil::sprintf(mid.c_str(), Arg.get_float());
        else if (argtype.is_triple()) {
            std::string format = fmtformat("{} {} {}", mid, mid, mid);
            const Vec3& v      = Arg.get_vec3();
            formatted = Strutil::sprintf(format.c_str(), v[0], v[1], v[2]);
        } else if (argtype.is_matrix()) {
            std::string format
                = fmtformat("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
                            mid, mid, mid, mid, mid, mid, mid, mid, mid, mid,
                            mid, mid, mid, mid, mid, mid);
            const float* m = (const float*)Arg.dataptr();
            formatted = Strutil::sprintf(format.c_str(), m[0], m[1], m[2], m[3],
                                         m[4], m[5], m[6], m[7], m[8], m[9],
                                         m[10], m[11], m[12], m[13], m[14],
                                         m[15]);
        } else if (argtype.is_string())
            formatted = Strutil::sprintf(mid.c_str(),
                                         Arg.get_string().string());
        else
            break;  // something else we don't handle -- we're done

        // We were able to format, so rejigger the strings.
        prefix.erase(pos, std::string::npos);
        suffix = formatted + suffix;
        args_expanded += 1;
    }

    // Rewrite the op
    if (args_expanded == op.nargs() - 2) {
        // Special case -- completely expanded, replace with a string
        // assignment
        int cind = rop.add_constant(ustring(prefix + suffix));
        rop.turn_into_assign(op, cind, "fully constant fold format()");
        return 1;
    } else if (args_expanded != 0) {
        // Partially expanded -- rewrite the instruction. It's actually
        // easier to turn this instruction into a nop and insert a new one.
        // Grab the previous arguments, drop the ones we folded, and
        // replace the format string with our new one.
        int* argstart = &rop.inst()->args()[0] + op.firstarg();
        std::vector<int> newargs(argstart,
                                 argstart + op.nargs() - args_expanded);
        newargs[1]     = rop.add_constant(ustring(prefix + suffix));
        ustring opname = op.opname();
        rop.turn_into_nop(op, "partial constant fold format()");
        rop.insert_code(opnum, opname, newargs,
                        RuntimeOptimizer::RecomputeRWRanges);
        return 1;
    }

    return 0;
}



DECLFOLDER(constfold_substr)
{
    // Try to turn R=substr(s,start,len) into R=C
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& S(*rop.opargsym(op, 1));
    Symbol& Start(*rop.opargsym(op, 2));
    Symbol& Len(*rop.opargsym(op, 3));
    if (S.is_constant() && Start.is_constant() && Len.is_constant()) {
        OSL_DASSERT(S.typespec().is_string() && Start.typespec().is_int()
                    && Len.typespec().is_int());
        ustring s = S.get_string();
        int start = Start.get_int();
        int len   = Len.get_int();
        int slen  = s.length();
        int b     = start;
        if (b < 0)
            b += slen;
        b = Imath::clamp(b, 0, slen);
        ustring r(s, b, Imath::clamp(len, 0, slen));
        int cind = rop.add_constant(r);
        rop.turn_into_assign(op, cind, "const fold substr");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_regex_search)
{
    // Try to turn R=regex_search(subj,reg) into R=C
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Subj(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& Reg(*rop.inst()->argsymbol(op.firstarg() + 2));
    if (op.nargs() == 3  // only the 2-arg version without search results
        && Subj.is_constant() && Reg.is_constant()) {
        OSL_DASSERT(Subj.typespec().is_string() && Reg.typespec().is_string());
        ustring s(Subj.get_string());
        ustring r(Reg.get_string());
        std::regex reg(r.string());
        int result = std::regex_search(s.string(), reg);
        int cind   = rop.add_constant(result);
        rop.turn_into_assign(op, cind, "const fold regex_search");
        return 1;
    }
    return 0;
}



inline float
clamp(float x, float minv, float maxv)
{
    if (x < minv)
        return minv;
    else if (x > maxv)
        return maxv;
    else
        return x;
}



DECLFOLDER(constfold_clamp)
{
    // Try to turn R=clamp(x,min,max) into R=C
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& X(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& Min(*rop.inst()->argsymbol(op.firstarg() + 2));
    Symbol& Max(*rop.inst()->argsymbol(op.firstarg() + 3));
    if (X.is_constant() && Min.is_constant() && Max.is_constant()
        && equivalent(X.typespec(), Min.typespec())
        && equivalent(X.typespec(), Max.typespec())
        && (X.typespec().is_float() || X.typespec().is_triple())) {
        const float* x   = (const float*)X.dataptr();
        const float* min = (const float*)Min.dataptr();
        const float* max = (const float*)Max.dataptr();
        float result[3];
        result[0] = clamp(x[0], min[0], max[0]);
        if (X.typespec().is_triple()) {
            result[1] = clamp(x[1], min[1], max[1]);
            result[2] = clamp(x[2], min[2], max[2]);
        }
        int cind = rop.add_constant(X.typespec(), &result);
        rop.turn_into_assign(op, cind, "const fold clamp");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_mix)
{
    // Try to turn R=mix(a,b,x) into
    //   R = c             if a,b,x are all are constant
    //   R = a             if x is constant and x == 0
    //   R = b             if x is constant and x == 1
    //   R = a             if a and b are the same (even if not constant)
    //
    Opcode& op(rop.inst()->ops()[opnum]);
    int Rind = rop.oparg(op, 0);
    int Aind = rop.oparg(op, 1);
    int Bind = rop.oparg(op, 2);
    int Xind = rop.oparg(op, 3);
    Symbol& R(*rop.inst()->symbol(Rind));
    Symbol& A(*rop.inst()->symbol(Aind));
    Symbol& B(*rop.inst()->symbol(Bind));
    Symbol& X(*rop.inst()->symbol(Xind));
    // Everything better be a float or triple
    if (!((A.typespec().is_float() || A.typespec().is_triple())
          && (B.typespec().is_float() || B.typespec().is_triple())
          && (X.typespec().is_float() || X.typespec().is_triple())))
        return 0;
    if (X.is_constant() && A.is_constant() && B.is_constant()) {
        // All three constants
        float result[3];
        const float* a = (const float*)A.dataptr();
        const float* b = (const float*)B.dataptr();
        const float* x = (const float*)X.dataptr();
        bool atriple   = A.typespec().is_triple();
        bool btriple   = B.typespec().is_triple();
        bool xtriple   = X.typespec().is_triple();
        bool rtriple   = R.typespec().is_triple();
        int ncomps     = rtriple ? 3 : 1;
        for (int i = 0; i < ncomps; ++i) {
            float xval = x[xtriple * i];
            result[i]  = (1.0f - xval) * a[atriple * i] + xval * b[btriple * i];
        }
        int cind = rop.add_constant(R.typespec(), &result);
        rop.turn_into_assign(op, cind, "const fold mix");
        return 1;
    }

    // Two special cases... X is 0, X is 1
    if (rop.is_zero(X)) {  // mix(A,B,0) == A
        rop.turn_into_assign(op, Aind, "mix(a,b,0) => a");
        return 1;
    }
    if (rop.is_one(X)) {  // mix(A,B,1) == B
        rop.turn_into_assign(op, Bind, "mix(a,b,1) => b");
        return 1;
    }

    if (rop.is_zero(A)
        && (!B.connected() || !rop.opt_mix() || rop.optimization_pass() > 2)) {
        // mix(0,b,x) == b*x, but only do this if b is not connected.
        // Because if b is connected, it may pull on something expensive.
        rop.turn_into_new_op(op, u_mul, Rind, Bind, Xind, "mix(0,b,x) => b*x");
        return 1;
    }
#if 0
    // This seems to almost never happen, so don't worry about it
    if (rop.is_zero(B) && ! A.connected()) {
        // mix(a,0,x) == (1-x)*a, but only do this if b is not connected
    }
#endif

    // mix (a, a, x) is a, regardless of x and even if none are constants
    if (Aind == Bind) {
        rop.turn_into_assign(op, Aind, "const fold: mix(a,a,x) -> a");
    }

    // Special sauce: mix(a,b,x) is implemented as a*(1-x)+b*x.  But
    // consider cases where x is not constant (thus not foldable), but
    // nonetheless turns out to be 0 or 1 much of the time.  If a and b
    // are short local computations, it's not so bad, but if they are
    // shader parameters connected to other layers, this affair may
    // needlessly evaluate other layers for no purpose other than to
    // multiply their results by zero.  So we try to ameliorate that
    // case with some extra tests here.  N.B. we delay doing this until
    // a few optimization passes in, to give enough time to optimize
    // away the inputs in other ways before introducing the 'if'.
    if (rop.opt_mix() && rop.optimization_pass() > 1 && !X.is_constant()
        && (A.connected() || B.connected())) {
        // A or B are connected, and thus presumed expensive, so turn into:
        //    if (X == 0)  // But eliminate this clause if B not connected
        //        R = A;
        //    else if (X == 1)  // But eliminate this clause if A not connected
        //        R = B;
        //    else
        //        R = A*(1-X) + B*X;
        int if0op = -1;  // Op where we have the 'if' for testing x==0
        int if1op = -1;  // Op where we have the 'if' for testing x==1
        if (B.connected()) {
            // Add the test and conditional for X==0, in which case we can
            // just R=A and not have to access B
            int cond  = rop.add_temp(TypeInt);
            int fzero = rop.add_constant(0.0f);
            rop.insert_code(opnum++, u_eq, RuntimeOptimizer::GroupWithNext,
                            cond, Xind, fzero);
            if0op = opnum;
            rop.insert_code(opnum++, u_if, RuntimeOptimizer::GroupWithNext,
                            cond);
            rop.op(if0op).argreadonly(0);
            rop.symbol(cond)->mark_rw(if0op, true, false);
            // Add the true (R=A) clause
            rop.insert_code(opnum++, u_assign, RuntimeOptimizer::GroupWithNext,
                            Rind, Aind);
        }
        int if0op_false = opnum;  // Where we jump if the 'if x==0' is false
        if (A.connected()) {
            // Add the test and conditional for X==1, in which case we can
            // just R=B and not have to access A
            int cond = rop.add_temp(TypeInt);
            int fone = rop.add_constant(1.0f);
            rop.insert_code(opnum++, u_eq, RuntimeOptimizer::GroupWithNext,
                            cond, Xind, fone);
            if1op = opnum;
            rop.insert_code(opnum++, u_if, RuntimeOptimizer::GroupWithNext,
                            cond);
            rop.op(if1op).argreadonly(0);
            rop.symbol(cond)->mark_rw(if1op, true, false);
            // Add the true (R=B) clause
            rop.insert_code(opnum++, u_assign, RuntimeOptimizer::GroupWithNext,
                            Rind, Bind);
        }
        int if1op_false = opnum;  // Where we jump if the 'if x==1' is false
        // Add the (R=A*(1-X)+B*X) clause -- always need that
        int one_minus_x = rop.add_temp(X.typespec());
        int temp1       = rop.add_temp(A.typespec());
        int temp2       = rop.add_temp(B.typespec());
        int fone        = rop.add_constant(1.0f);
        rop.insert_code(opnum++, u_sub, RuntimeOptimizer::GroupWithNext,
                        one_minus_x, fone, Xind);
        rop.insert_code(opnum++, u_mul, RuntimeOptimizer::GroupWithNext, temp1,
                        Aind, one_minus_x);
        rop.insert_code(opnum++, u_mul, RuntimeOptimizer::GroupWithNext, temp2,
                        Bind, Xind);
        rop.insert_code(opnum++, u_add, RuntimeOptimizer::GroupWithNext, Rind,
                        temp1, temp2);
        // Now go back and patch the 'if' ops with the right jump addresses
        if (if0op >= 0)
            rop.op(if0op).set_jump(if0op_false, opnum);
        if (if1op >= 0)
            rop.op(if1op).set_jump(if1op_false, opnum);
        // The next op is the original mix, make it nop
        rop.turn_into_nop(rop.op(opnum), "smart 'mix'");
        return 1;
    }

    return 0;
}



DECLFOLDER(constfold_select)
{
    // Try to turn R=select(a,b,cond) into (per component):
    //   R[c] = a          if cond is constant and zero
    //   R[c] = b          if cond is constant and nonzero
    //   R = a             if a == b (even if nothing is constant)
    //
    Opcode& op(rop.inst()->ops()[opnum]);
    // int Rind = rop.oparg(op,0);
    int Aind = rop.oparg(op, 1);
    int Bind = rop.oparg(op, 2);
    int Cind = rop.oparg(op, 3);
    Symbol& C(*rop.inst()->symbol(Cind));

    if (C.is_constant() && rop.is_zero(C)) {
        rop.turn_into_assign(op, Aind, "select(A,B,0) => A");
        return 1;
    }
    if (C.is_constant() && rop.is_nonzero(C)) {
        rop.turn_into_assign(op, Bind, "select(A,B,non-0) => B");
        return 1;
    }
    if (Aind == Bind) {
        rop.turn_into_assign(op, Aind, "select(c,a,a) -> a");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_min)
{
    // Try to turn R=min(x,y) into R=C
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& X(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& Y(*rop.inst()->argsymbol(op.firstarg() + 2));
    if (X.is_constant() && Y.is_constant()
        && equivalent(X.typespec(), Y.typespec())) {
        if (X.typespec().is_float() || X.typespec().is_triple()) {
            const float* x = (const float*)X.dataptr();
            const float* y = (const float*)Y.dataptr();
            float result[3];
            result[0] = std::min(x[0], y[0]);
            if (X.typespec().is_triple()) {
                result[1] = std::min(x[1], y[1]);
                result[2] = std::min(x[2], y[2]);
            }
            int cind = rop.add_constant(X.typespec(), &result);
            rop.turn_into_assign(op, cind, "const fold min");
            return 1;
        }
        if (X.typespec().is_int()) {
            const int* x = (const int*)X.dataptr();
            const int* y = (const int*)Y.dataptr();
            int result   = std::min(x[0], y[0]);
            int cind     = rop.add_constant(result);
            rop.turn_into_assign(op, cind, "const fold min");
            return 1;
        }
    }
    return 0;
}



DECLFOLDER(constfold_max)
{
    // Try to turn R=max(x,y) into R=C
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& X(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& Y(*rop.inst()->argsymbol(op.firstarg() + 2));
    if (X.is_constant() && Y.is_constant()
        && equivalent(X.typespec(), Y.typespec())) {
        if (X.typespec().is_float() || X.typespec().is_triple()) {
            const float* x = (const float*)X.dataptr();
            const float* y = (const float*)Y.dataptr();
            float result[3];
            result[0] = std::max(x[0], y[0]);
            if (X.typespec().is_triple()) {
                result[1] = std::max(x[1], y[1]);
                result[2] = std::max(x[2], y[2]);
            }
            int cind = rop.add_constant(X.typespec(), &result);
            rop.turn_into_assign(op, cind, "const fold max");
            return 1;
        }
        if (X.typespec().is_int()) {
            const int* x = (const int*)X.dataptr();
            const int* y = (const int*)Y.dataptr();
            int result   = std::max(x[0], y[0]);
            int cind     = rop.add_constant(result);
            rop.turn_into_assign(op, cind, "const fold max");
            return 1;
        }
    }
    return 0;
}



// Handy macro for automatically constructing a constant-folder for
// a simple function of one argument that can be float or triple
// and returns the same type as its argument.
#define AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(name, impl)                     \
    DECLFOLDER(constfold_##name)                                        \
    {                                                                   \
        /* Try to turn R=f(x) into R=C */                               \
        Opcode& op(rop.inst()->ops()[opnum]);                           \
        Symbol& X(*rop.inst()->argsymbol(op.firstarg() + 1));           \
        if (X.is_constant()                                             \
            && (X.typespec().is_float() || X.typespec().is_triple())) { \
            const float* x = (const float*)X.dataptr();                 \
            float result[3];                                            \
            result[0] = impl(x[0]);                                     \
            if (X.typespec().is_triple()) {                             \
                result[1] = impl(x[1]);                                 \
                result[2] = impl(x[2]);                                 \
            }                                                           \
            int cind = rop.add_constant(X.typespec(), &result);         \
            rop.turn_into_assign(op, cind, "const fold " #name);        \
            return 1;                                                   \
        }                                                               \
        return 0;                                                       \
    }



AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(sqrt, OIIO::safe_sqrt)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(inversesqrt, OIIO::safe_inversesqrt)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(degrees, OIIO::degrees)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(radians, OIIO::radians)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(floor, floorf)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(ceil, ceilf)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(erf, OIIO::fast_erf)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(erfc, OIIO::fast_erfc)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(logb, OIIO::fast_logb)
#if OSL_FAST_MATH
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(cos, OIIO::fast_cos)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(sin, OIIO::fast_sin)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(acos, OIIO::fast_acos)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(asin, OIIO::fast_asin)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(exp, OIIO::fast_exp)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(exp2, OIIO::fast_exp2)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(expm1, OIIO::fast_expm1)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(log, OIIO::fast_log)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(log10, OIIO::fast_log10)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(log2, OIIO::fast_log2)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(cbrt, OIIO::fast_cbrt)
#else
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(cos, cosf)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(sin, sinf)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(acos, OIIO::safe_acos)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(asin, OIIO::safe_asin)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(exp, expf)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(exp2, exp2f)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(expm1, expm1f)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(log, OIIO::safe_log)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(log10, OIIO::safe_log10)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(log2, OIIO::safe_log2)
AUTO_DECLFOLDER_FLOAT_OR_TRIPLE(cbrt, cbrtf)
#endif

DECLFOLDER(constfold_pow)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& X(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& Y(*rop.inst()->argsymbol(op.firstarg() + 2));

    if (rop.is_zero(Y)) {
        // x^0 == 1
        rop.turn_into_assign_one(op, "pow(x,0) => 1");
        return 1;
    }
    if (rop.is_one(Y)) {
        // x^1 == x
        rop.turn_into_assign(op, rop.inst()->arg(op.firstarg() + 1),
                             "pow(x,1) => x");
        return 1;
    }
    if (rop.is_zero(X)) {
        // 0^y == 0
        rop.turn_into_assign_zero(op, "pow(0,x) => 0");
        return 1;
    }
    if (X.is_constant() && Y.is_constant()) {
        // if x and y are both constant, pre-compute x^y
        const float* x = (const float*)X.dataptr();
        const float* y = (const float*)Y.dataptr();
        int nxcomps    = X.typespec().is_triple() ? 3 : 1;
        int nycomps    = Y.typespec().is_triple() ? 3 : 1;
        float result[3];
        for (int i = 0; i < nxcomps; ++i) {
            int j = std::min(i, nycomps - 1);
#if OSL_FAST_MATH
            result[i] = OIIO::fast_safe_pow(x[i], y[j]);
#else
            result[i] = OIIO::safe_pow(x[i], y[j]);
#endif
        }
        int cind = rop.add_constant(X.typespec(), &result);
        rop.turn_into_assign(op, cind, "const fold pow");
        return 1;
    }

    // A few special cases of constant y:
    if (Y.is_constant() && Y.typespec().is_float()) {
        int resultarg = rop.inst()->args()[op.firstarg() + 0];
        int xarg      = rop.inst()->args()[op.firstarg() + 1];
        float yval    = Y.get_float();
        if (yval == 2.0f) {
            rop.turn_into_new_op(op, u_mul, resultarg, xarg, xarg,
                                 "pow(x,2) => x*x");
            return 1;
        }
        if (yval == 0.5f) {
            rop.turn_into_new_op(op, u_sqrt, resultarg, xarg, -1,
                                 "pow(x,0.5) => sqrt(x)");
            return 1;
        }
        if (yval == -0.5f) {
            rop.turn_into_new_op(op, u_inversesqrt, resultarg, xarg, -1,
                                 "pow(x,-0.5) => inversesqrt(x)");
            return 1;
        }
        if (yval == 1.0f / 3.0f) {
            rop.turn_into_new_op(op, u_cbrt, resultarg, xarg, -1,
                                 "pow(x,1.0/3.0) => cbrt(x)");
            return 1;
        }
    }

    return 0;
}



DECLFOLDER(constfold_sincos)
{
    // Try to turn sincos(const_angle,s,c) into s=sin_a, c = cos_a
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 0));
    if (A.is_constant()) {
        int sinarg  = rop.inst()->args()[op.firstarg() + 1];
        int cosarg  = rop.inst()->args()[op.firstarg() + 2];
        float angle = A.get_float();
        float s, c;
#if OSL_FAST_MATH
        OIIO::fast_sincos(angle, &s, &c);
#else
        OIIO::sincos(angle, &s, &c);
#endif
        // Turn this op into the sin assignment
        rop.turn_into_new_op(op, u_assign, sinarg, rop.add_constant(s), -1,
                             "const fold sincos");
        // And insert a new op for the cos assignment
        const int args_to_add[] = { cosarg, rop.add_constant(c) };
        rop.insert_code(opnum, u_assign, args_to_add,
                        RuntimeOptimizer::RecomputeRWRanges,
                        RuntimeOptimizer::GroupWithNext);
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_normalize)
{
    // Try to turn R=normalize(x) into R=C
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& X(*rop.inst()->argsymbol(op.firstarg() + 1));
    OSL_DASSERT(X.typespec().is_triple());
    if (X.is_constant()) {
        Vec3 result = X.get_vec3();
        result.normalize();
        int cind = rop.add_constant(X.typespec(), &result);
        rop.turn_into_assign(op, cind, "const fold normalize");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_triple)
{
    // Turn R=triple(a,b,c) into R=C if the components are all constants
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 4 || op.nargs() == 5);
    bool using_space = (op.nargs() == 5);
    Symbol& R(*rop.inst()->argsymbol(op.firstarg() + 0));
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1 + using_space));
    Symbol& B(*rop.inst()->argsymbol(op.firstarg() + 2 + using_space));
    Symbol& C(*rop.inst()->argsymbol(op.firstarg() + 3 + using_space));
    if (using_space) {
        // If we're using a space name and it's equivalent to "common",
        // just pretend it doesn't exist.
        Symbol& Space(*rop.inst()->argsymbol(op.firstarg() + 1));
        if (Space.is_constant()
            && (Space.get_string() == Strings::common
                || Space.get_string() == rop.shadingsys().commonspace_synonym()))
            using_space = false;
    }
    if (A.is_constant() && A.typespec().is_float() && B.is_constant()
        && C.is_constant() && !using_space) {
        OSL_DASSERT(A.typespec().is_float() && B.typespec().is_float()
                    && C.typespec().is_float());
        float result[3];
        result[0] = A.get_float();
        result[1] = B.get_float();
        result[2] = C.get_float();
        int cind  = rop.add_constant(R.typespec(), &result);
        rop.turn_into_assign(op, cind,
                             "triple(const,const,const) => triple constant");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_matrix)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    int nargs       = op.nargs();
    int using_space = rop.opargsym(op, 1)->typespec().is_string() ? 1 : 0;
    if (using_space && nargs > 2 && rop.opargsym(op, 2)->typespec().is_string())
        using_space = 2;
    int nfloats = nargs - 1 - using_space;
    OSL_DASSERT(nfloats == 1 || nfloats == 16
                || (nfloats == 0 && using_space == 2));
    if (nargs == 3 && using_space == 2) {
        // Try to simplify R=matrix(from,to) in cases of an identify
        // transform: if From and To are the same variable (even if not a
        // constant), or if their values are the same, or if one is "common"
        // and the other is the designated common space synonym.
        Symbol& From(*rop.inst()->argsymbol(op.firstarg() + 1));
        Symbol& To(*rop.inst()->argsymbol(op.firstarg() + 2));
        ustring from = From.is_constant() ? From.get_string()
                                          : ustring("$unknown1$");
        ustring to = To.is_constant() ? To.get_string() : ustring("$unknown2$");
        ustring commonsyn = rop.inst()->shadingsys().commonspace_synonym();
        if (&From == &To || from == to
            || ((from == Strings::common && to == commonsyn)
                || (from == commonsyn && to == Strings::common))) {
            static Matrix44 ident(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                  1);
            rop.turn_into_assign(op, rop.add_constant(ident),
                                 "matrix(spaceA,spaceA) => identity matrix");
            return 1;
        }
        // Try to simplify R=matrix(from,to) in cases of an constant (but
        // different) names -- do the matrix retrieval now, if not time-
        // varying matrices.
        if (!(From.is_constant() && To.is_constant()))
            return 0;
        // Shader and object spaces will vary from execution to execution,
        // so we can't optimize those away.
        if (from == Strings::shader || from == Strings::object
            || to == Strings::shader || to == Strings::object)
            return 0;
        // But whatever spaces are left *may* be optimizable if they are
        // not time-varying.
        RendererServices* rs = rop.shadingsys().renderer();
        Matrix44 Mfrom, Mto;
        bool ok = true;
        if (from == Strings::common || from == commonsyn)
            Mfrom.makeIdentity();
        else
            ok &= rs->get_matrix(rop.shaderglobals(), Mfrom, from);
        if (to == Strings::common || to == commonsyn)
            Mto.makeIdentity();
        else
            ok &= rs->get_inverse_matrix(rop.shaderglobals(), Mto, to);
        if (ok) {
            // The from-to matrix is known and not time-varying, so just
            // turn it into a constant rather than calling getmatrix at
            // execution time.
            Matrix44 Mresult = Mfrom * Mto;
            int cind         = rop.add_constant(TypeMatrix, &Mresult);
            rop.turn_into_assign(op, cind, "const fold matrix");
            return 1;
        }
    }
    if (using_space == 1 && nfloats == 1) {
        // Turn matrix("common",1) info identity matrix.
        Symbol& From(*rop.inst()->argsymbol(op.firstarg() + 1));
        Symbol& Val(*rop.inst()->argsymbol(op.firstarg() + 2));
        if (From.is_constant() && Val.is_constant()
            && Val.get_float() == 1.0f) {
            ustring from = From.get_string();
            if (from == Strings::common
                || from == rop.inst()->shadingsys().commonspace_synonym()) {
                static Matrix44 ident(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                                      0, 1);
                rop.turn_into_assign(op, rop.add_constant(ident),
                                     "matrix(\"common\",1) => identity matrix");
            }
        }
    }
    if (nfloats == 16 && using_space == 0) {
        // Try to turn matrix(...16 float consts...) into just a const
        // matrix assign.
        bool all_const = true;
        float M[16];
        for (int i = 0; i < 16; ++i) {
            Symbol& Val(*rop.inst()->argsymbol(op.firstarg() + 1 + i));
            if (Val.is_constant())
                M[i] = Val.get_float();
            else {
                all_const = false;
                break;
            }
        }
        if (all_const) {
            rop.turn_into_assign(op, rop.add_constant(TypeMatrix, M),
                                 "const fold matrix");
            return 1;
        }
    }
    if (nfloats == 1 && using_space == 0) {
        // Try to turn matrix(const float) into just a const matrix assign.
        Symbol& Val(*rop.inst()->argsymbol(op.firstarg() + 1));
        if (Val.is_constant()) {
            float val = Val.get_float();
            Matrix44 M(val, 0, 0, 0, 0, val, 0, 0, 0, 0, val, 0, 0, 0, 0, val);
            rop.turn_into_assign(op, rop.add_constant(M), "const fold matrix");
            return 1;
        }
    }
    return 0;
}



DECLFOLDER(constfold_getmatrix)
{
    // Try to turn R=getmatrix(from,to,M) into R=1,M=const if it's an
    // identity transform or if the result is a non-time-varying matrix.
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& From(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& To(*rop.inst()->argsymbol(op.firstarg() + 2));
    if (!(From.is_constant() && To.is_constant()))
        return 0;
    // OK, From and To are constant strings.
    ustring from      = From.get_string();
    ustring to        = To.get_string();
    ustring commonsyn = rop.inst()->shadingsys().commonspace_synonym();

    // Shader and object spaces will vary from execution to execution,
    // so we can't optimize those away.
    if (from == Strings::shader || from == Strings::object
        || to == Strings::shader || to == Strings::object)
        return 0;

    // But whatever spaces are left *may* be optimizable if they are
    // not time-varying.
    RendererServices* rs = rop.shadingsys().renderer();
    Matrix44 Mfrom, Mto;
    bool ok = true;
    if (from == Strings::common || from == commonsyn || from == to)
        Mfrom.makeIdentity();
    else
        ok &= rs->get_matrix(rop.shaderglobals(), Mfrom, from);
    if (to == Strings::common || to == commonsyn || from == to)
        Mto.makeIdentity();
    else
        ok &= rs->get_inverse_matrix(rop.shaderglobals(), Mto, to);
    if (ok) {
        // The from-to matrix is known and not time-varying, so just
        // turn it into a constant rather than calling getmatrix at
        // execution time.
        int resultarg = rop.inst()->args()[op.firstarg() + 0];
        int dataarg   = rop.inst()->args()[op.firstarg() + 3];
        // Make data the first argument
        rop.inst()->args()[op.firstarg() + 0] = dataarg;
        // Now turn it into an assignment
        Matrix44 Mresult = Mfrom * Mto;
        int cind         = rop.add_constant(TypeMatrix, &Mresult);
        rop.turn_into_assign(op, cind, "getmatrix of known matrix");

        // Now insert a new instruction that assigns 1 to the
        // original return result of getmatrix.
        const int one           = 1;
        const int args_to_add[] = { resultarg,
                                    rop.add_constant(TypeInt, &one) };
        rop.insert_code(opnum, u_assign, args_to_add,
                        RuntimeOptimizer::RecomputeRWRanges,
                        RuntimeOptimizer::GroupWithNext);
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_transform)
{
    // Try to turn identity transforms into assignments
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& M(*rop.inst()->argsymbol(op.firstarg() + 1));
    if (op.nargs() == 3 && M.typespec().is_matrix() && rop.is_one(M)) {
        rop.turn_into_assign(op, rop.inst()->arg(op.firstarg() + 2),
                             "transform by identity");
        return 1;
    }
    if (op.nargs() == 4) {
        Symbol& T(*rop.inst()->argsymbol(op.firstarg() + 2));
        if (M.is_constant() && T.is_constant()) {
            OSL_DASSERT(M.typespec().is_string() && T.typespec().is_string());
            ustring from = M.get_string();
            ustring to   = T.get_string();
            ustring syn  = rop.shadingsys().commonspace_synonym();
            if (from == syn)
                from = Strings::common;
            if (to == syn)
                to = Strings::common;
            if (from == to) {
                rop.turn_into_assign(op, rop.inst()->arg(op.firstarg() + 3),
                                     "transform by identity");
                return 1;
            }
        }
    }
    return 0;
}



DECLFOLDER(constfold_transformc)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    // Symbol &Result = *rop.opargsym (op, 0);
    Symbol& From = *rop.opargsym(op, 1);
    Symbol& To   = *rop.opargsym(op, 2);
    Symbol& C    = *rop.opargsym(op, 3);

    if (From.is_constant() && To.is_constant()) {
        ustring from = From.get_string();
        ustring to   = To.get_string();
        if (from == Strings::RGB)
            from = Strings::rgb;
        if (to == Strings::RGB)
            to = Strings::rgb;
        if (from == to) {
            rop.turn_into_assign(op, rop.inst()->arg(op.firstarg() + 3),
                                 "transformc by identity");
            return 1;
        }
        if (C.is_constant()) {
            Color3 Cin(C.get_float(0), C.get_float(1), C.get_float(2));
            Color3 result = rop.shadingsys().colorsystem().transformc(
                from, to, Cin, rop.shaderglobals()->context, nullptr);
            rop.turn_into_assign(op, rop.add_constantc(result),
                                 "transformc => constant");
            return 1;
        }
    }
    return 0;
}



DECLFOLDER(constfold_setmessage)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Name(*rop.inst()->argsymbol(op.firstarg() + 0));

    // Record that the inst set a message
    if (Name.is_constant()) {
        OSL_DASSERT(Name.typespec().is_string());
        rop.register_message(Name.get_string());
    } else {
        rop.register_unknown_message();
    }

    return 0;
}



DECLFOLDER(constfold_getmessage)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    int has_source = (op.nargs() == 4);
    if (has_source)
        return 0;  // Don't optimize away sourced getmessage
    Symbol& Name(*rop.inst()->argsymbol(op.firstarg() + 1 + (int)has_source));
    if (Name.is_constant()) {
        OSL_DASSERT(Name.typespec().is_string());
        if (!rop.message_possibly_set(Name.get_string())) {
            // If the messages could not have been sent, get rid of the
            // getmessage op, leave the destination value alone, and
            // assign 0 to the returned status of getmessage.
            rop.turn_into_assign_zero(op, "impossible getmessage");
            return 1;
        }
    }
    return 0;
}



DECLFOLDER(constfold_getattribute)
{
    if (!rop.shadingsys().fold_getattribute())
        return 0;

    // getattribute() has eight "flavors":
    //   * getattribute (attribute_name, value)
    //   * getattribute (attribute_name, value[])
    //   * getattribute (attribute_name, index, value)
    //   * getattribute (attribute_name, index, value[])
    //   * getattribute (object, attribute_name, value)
    //   * getattribute (object, attribute_name, value[])
    //   * getattribute (object, attribute_name, index, value)
    //   * getattribute (object, attribute_name, index, value[])
    Opcode& op(rop.inst()->ops()[opnum]);
    int nargs = op.nargs();
    OSL_DASSERT(nargs >= 3 && nargs <= 5);
    bool array_lookup  = rop.opargsym(op, nargs - 2)->typespec().is_int();
    bool object_lookup = rop.opargsym(op, 2)->typespec().is_string()
                         && nargs >= 4;
    int object_slot = (int)object_lookup;
    int attrib_slot = object_slot + 1;
    int index_slot  = nargs - 2;
    int dest_slot   = nargs - 1;

    //    Symbol& Result      = *rop.opargsym (op, 0);
    Symbol& ObjectName
        = *rop.opargsym(op, object_slot);  // only valid if object_slot is true
    Symbol& Attribute = *rop.opargsym(op, attrib_slot);
    Symbol& Index
        = *rop.opargsym(op, index_slot);  // only valid if array_lookup is true
    Symbol& Destination = *rop.opargsym(op, dest_slot);

    if (!Attribute.is_constant() || (object_lookup && !ObjectName.is_constant())
        || (array_lookup && !Index.is_constant()))
        return 0;  // Non-constant things prevent a fold
    if (Destination.typespec().is_array())
        return 0;  // Punt on arrays for now

    ustring attr_name       = Attribute.get_string();
    const size_t maxbufsize = 1024;
    char buf[maxbufsize];
    TypeDesc attr_type = Destination.typespec().simpletype();
    if (attr_type.size() > maxbufsize)
        return 0;  // Don't constant fold humongous things

    bool found = false;

    // Check global things first
    if (attr_name == "osl:version" && attr_type == TypeInt) {
        int* val = (int*)(char*)buf;
        *val     = OSL_VERSION;
        found    = true;
    } else if (attr_name == "shader:shadername" && attr_type == TypeString) {
        ustring* up = (ustring*)(char*)buf;
        *up         = ustring(rop.inst()->shadername());
        found       = true;
    } else if (attr_name == "shader:layername" && attr_type == TypeString) {
        ustring* up = (ustring*)(char*)buf;
        *up         = rop.inst()->layername();
        found       = true;
    } else if (attr_name == "shader:groupname" && attr_type == TypeString) {
        ustring* up = (ustring*)(char*)buf;
        *up         = rop.group().name();
        found       = true;
    }

    if (!found) {
        // If the object name is not supplied, it implies that we are
        // supposed to search the shaded object first, then if that fails,
        // the scene-wide namespace.  We can't do that yet, have to wait
        // until shade time.
        ustring obj_name;
        if (object_lookup)
            obj_name = ObjectName.get_string();
        if (obj_name.empty())
            return 0;

        found = array_lookup
                    ? rop.renderer()->get_array_attribute(NULL, false, obj_name,
                                                          attr_type, attr_name,
                                                          Index.get_int(), buf)
                    : rop.renderer()->get_attribute(NULL, false, obj_name,
                                                    attr_type, attr_name, buf);
    }

    if (found) {
        // Now we turn the existing getattribute op into this for success:
        //       assign result 1
        //       assign data [retrieved values]
        // but if it fails, don't change anything, because we want it to
        // issue errors at runtime.

        // Make the data destination be the first argument
        int oldresultarg = rop.inst()->args()[op.firstarg() + 0];
        int dataarg      = rop.inst()->args()[op.firstarg() + dest_slot];
        rop.inst()->args()[op.firstarg() + 0] = dataarg;
        // Now turn it into an assignment
        int cind = rop.add_constant(attr_type, &buf);
        rop.turn_into_assign(op, cind, "const fold getattribute");
        // Now insert a new instruction that assigns 1 to the
        // original return result of getattribute.
        const int one           = 1;
        const int args_to_add[] = { oldresultarg,
                                    rop.add_constant(TypeInt, &one) };
        rop.insert_code(opnum, u_assign, args_to_add,
                        RuntimeOptimizer::RecomputeRWRanges,
                        RuntimeOptimizer::GroupWithNext);
        return 1;
    } else {
        return 0;
    }
}



DECLFOLDER(constfold_gettextureinfo)
{
    Opcode& op(rop.inst()->ops()[opnum]);

    // FIXME: For now, punt on constant folding the variety of gettextureinfo
    // that is passed texture coordinates. Eventually, if we can determine that
    // the filename is constant and is known to not be UDIM, we can fall back
    // to the other case.
    bool use_coords = (op.nargs() == 6);
    if (use_coords)
        return 0;

    OSL_MAYBE_UNUSED Symbol& Result(*rop.inst()->argsymbol(op.firstarg() + 0));
    Symbol& Filename(*rop.inst()->argsymbol(op.firstarg() + 1));
    Symbol& Dataname(
        *rop.inst()->argsymbol(op.firstarg() + (use_coords ? 4 : 2)));
    Symbol& Data(*rop.inst()->argsymbol(op.firstarg() + (use_coords ? 5 : 3)));
    OSL_DASSERT(Result.typespec().is_int() && Filename.typespec().is_string()
                && Dataname.typespec().is_string());

    if (Filename.is_constant() && Dataname.is_constant()) {
        ustring filename = Filename.get_string();
        ustring dataname = Dataname.get_string();
        TypeDesc t       = Data.typespec().simpletype();
        void* mydata     = OSL_ALLOCA(char, t.size());
        // FIXME(ptex) -- exclude folding of ptex, since these things
        // can vary per face.
        ustringhash em;
        int result = rop.renderer()->get_texture_info(
            filename, nullptr, rop.shadingcontext()->texture_thread_info(),
            rop.shaderglobals(), 0 /* TODO: subimage? */, dataname, t, mydata,
            &em);
        // Results from RendererServices for TypeDesc::STRING will actaully be a ustringhash
        // as that is what we expect during execution.  We will need to convert it back to
        // a regular ustring before continuing since this is the compile/optimization phase.
        if (result && t == TypeDesc::STRING) {
            auto uh_mydata = reinterpret_cast<ustringhash*>(mydata)[0];
            reinterpret_cast<ustring*>(mydata)[0] = ustring_from(uh_mydata);
        }
        ustring errormessage = ustring_from(em);
        // Now we turn
        //       gettextureinfo result filename dataname data
        // into this for success:
        //       assign data [retrieved values]
        //       assign result 1
        // into this for failure:
        //       error "%s" errormessage
        //       assign result 0
        if (result) {
            int oldresultarg = rop.inst()->args()[op.firstarg() + 0];
            int dataarg      = rop.inst()->args()[op.firstarg() + 3];
            // Make data the first argument
            rop.inst()->args()[op.firstarg() + 0] = dataarg;
            // Now turn it into an assignment
            int cind = rop.add_constant(Data.typespec(), mydata);
            rop.turn_into_assign(op, cind, "const fold gettextureinfo");

            // Now insert a new instruction that assigns 1 to the
            // original return result of gettextureinfo.
            int one                 = 1;
            const int args_to_add[] = { oldresultarg,
                                        rop.add_constant(TypeInt, &one) };
            rop.insert_code(opnum, u_assign, args_to_add,
                            RuntimeOptimizer::RecomputeRWRanges,
                            RuntimeOptimizer::GroupWithNext);
            return 1;
        } else {
            // Constant fold to 0
            rop.turn_into_assign_zero(op, "const fold gettextureinfo");
            if (errormessage.size()) {
                // display the error message if control flow ever reaches here
                const int args_to_add[] = { rop.add_constant(u_fmterror),
                                            rop.add_constant(errormessage) };
                rop.insert_code(opnum, u_error, args_to_add,
                                RuntimeOptimizer::RecomputeRWRanges,
                                RuntimeOptimizer::GroupWithNext);
                Opcode& newop(rop.inst()->ops()[opnum]);
                newop.argreadonly(0);
                newop.argreadonly(1);
            }
            return 1;
        }
    }
    return 0;
}



// texture -- we can eliminate a lot of superfluous setting of optional
// parameters to their default values.
DECLFOLDER(constfold_texture)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    // Symbol &Result = *rop.opargsym (op, 0);
    // Symbol &Filename = *rop.opargsym (op, 1);
    // Symbol &S = *rop.opargsym (op, 2);
    // Symbol &T = *rop.opargsym (op, 3);

    int first_optional_arg = 4;
    if (op.nargs() > 4 && rop.opargsym(op, 4)->typespec().is_float()) {
        //user_derivs = true;
        first_optional_arg = 8;
        OSL_DASSERT(rop.opargsym(op, 5)->typespec().is_float());
        OSL_DASSERT(rop.opargsym(op, 6)->typespec().is_float());
        OSL_DASSERT(rop.opargsym(op, 7)->typespec().is_float());
    }

    TextureOpt opt;  // So we can check the defaults
    bool swidth_set = false, twidth_set = false, rwidth_set = false;
    bool sblur_set = false, tblur_set = false, rblur_set = false;
    bool swrap_set = false, twrap_set = false, rwrap_set = false;
    bool firstchannel_set = false, fill_set = false, interp_set = false;
    bool any_elided = false;
    for (int i = first_optional_arg; i < op.nargs() - 1; i += 2) {
        Symbol& Name  = *rop.opargsym(op, i);
        Symbol& Value = *rop.opargsym(op, i + 1);
        OSL_DASSERT(Name.typespec().is_string());
        if (Name.is_constant() && Value.is_constant()) {
            ustring name       = Name.get_string();
            bool elide         = false;
            void* value        = Value.dataptr();
            TypeDesc valuetype = Value.typespec().simpletype();

// Keep from repeating the same tedious code for {s,t,r, }{width,blur,wrap}
#define CHECK(field, ctype, osltype)                                      \
    if (name == Strings::field && !field##_set) {                         \
        if (valuetype == osltype && *(ctype*)value == (ctype)opt.field)   \
            elide = true;                                                 \
        else if (osltype == TypeDesc::FLOAT && valuetype == TypeDesc::INT \
                 && *(int*)value == (int)opt.field)                       \
            elide = true;                                                 \
        else                                                              \
            field##_set = true;                                           \
    }
#define CHECK_str(field, ctype, osltype)                                       \
    CHECK(s##field, ctype, osltype)                                            \
    else CHECK(t##field, ctype,                                                \
               osltype) else CHECK(r##field, ctype,                            \
                                   osltype) else if (name == Strings::field    \
                                                     && !s##field##_set        \
                                                     && !t##field##_set        \
                                                     && !r##field##_set)       \
    {                                                                          \
        if (valuetype == osltype) {                                            \
            ctype* v = (ctype*)value;                                          \
            if (*v == (ctype)opt.s##field && *v == (ctype)opt.t##field         \
                && *v == (ctype)opt.r##field)                                  \
                elide = true;                                                  \
            else {                                                             \
                s##field##_set = true;                                         \
                t##field##_set = true;                                         \
                r##field##_set = true;                                         \
            }                                                                  \
        } else if (osltype == TypeDesc::FLOAT && valuetype == TypeDesc::INT) { \
            int* v = (int*)value;                                              \
            if (*v == (ctype)opt.s##field && *v == (ctype)opt.t##field         \
                && *v == (ctype)opt.r##field)                                  \
                elide = true;                                                  \
            else {                                                             \
                s##field##_set = true;                                         \
                t##field##_set = true;                                         \
                r##field##_set = true;                                         \
            }                                                                  \
        }                                                                      \
    }

#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wtautological-compare"
#endif
            CHECK_str(width, float, TypeDesc::FLOAT) else CHECK_str(
                blur, float,
                TypeDesc::FLOAT) else CHECK(firstchannel, int,
                                            TypeDesc::
                                                INT) else CHECK(fill, float,
                                                                TypeDesc::FLOAT)

                else if ((name == Strings::wrap || name == Strings::swrap
                          || name == Strings::twrap || name == Strings::rwrap)
                         && value && valuetype == TypeDesc::STRING)
            {
                // Special trick is needed for wrap modes because the input
                // is a string but the field we're setting is an int enum.
                OIIO::Tex::Wrap wrapmode = OIIO::Tex::decode_wrapmode(
                    *(ustring*)value);
                void* value = &wrapmode;
                CHECK_str(wrap, int, TypeDesc::INT);
            }
#ifdef __clang__
#    pragma clang diagnostic pop
#endif
#undef CHECK_STR
#undef CHECK

            // Cases that don't fit the pattern
            else if (name == Strings::interp && !interp_set)
            {
                if (value && valuetype == TypeDesc::STRING
                    && tex_interp_to_code(*(ustring*)value)
                           == (int)opt.interpmode)
                    elide = true;
                else
                    interp_set = true;
            }

            if (elide) {
                // Just turn the param name into empty string and it will
                // be skipped.
                ustring empty;
                int cind = rop.add_constant(TypeString, &empty);
                rop.inst()->args()[op.firstarg() + i]     = cind;
                rop.inst()->args()[op.firstarg() + i + 1] = cind;
                any_elided                                = true;
            }
        }
    }
    return any_elided;
}



DECLFOLDER(constfold_pointcloud_search)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() >= 5);
    int result_sym     = rop.oparg(op, 0);
    Symbol& Filename   = *rop.opargsym(op, 1);
    Symbol& Center     = *rop.opargsym(op, 2);
    Symbol& Radius     = *rop.opargsym(op, 3);
    Symbol& Max_points = *rop.opargsym(op, 4);
    OSL_DASSERT(Filename.typespec().is_string() && Center.typespec().is_triple()
                && Radius.typespec().is_float()
                && Max_points.typespec().is_int());

    // Can't constant fold unless all the required input args are constant
    if (!(Filename.is_constant() && Center.is_constant() && Radius.is_constant()
          && Max_points.is_constant()))
        return 0;

    // Handle the optional 'sort' flag, and don't bother constant folding
    // if sorted results may be required.
    int attr_arg_offset = 5;  // where the opt attrs begin
    if (op.nargs() > 5 && rop.opargsym(op, 5)->typespec().is_int()) {
        // Sorting requested
        Symbol* Sort = rop.opargsym(op, 5);
        if (!Sort->is_constant() || Sort->get_int())
            return 0;  // forget it if sorted data might be requested
        ++attr_arg_offset;
    }
    int nattrs = (op.nargs() - attr_arg_offset) / 2;

    // First pass through the optional arguments: gather the query names,
    // types, and destinations.  If any of the query names are not known
    // constants, we can't optimize this call so just return.
    std::vector<ustring> names;
    std::vector<int> value_args;
    std::vector<TypeDesc> value_types;
    for (int i = 0, num_queries = 0; i < nattrs; ++i) {
        Symbol& Name  = *rop.opargsym(op, attr_arg_offset + i * 2);
        Symbol& Value = *rop.opargsym(op, attr_arg_offset + i * 2 + 1);
        OSL_ASSERT(Name.typespec().is_string());
        if (!Name.is_constant())
            return 0;  // unknown optional argument, punt
        if (++num_queries > RuntimeOptimizer::max_new_consts_per_fold)
            return 0;
        names.push_back(Name.get_string());
        value_args.push_back(rop.oparg(op, attr_arg_offset + i * 2 + 1));
        value_types.push_back(Value.typespec().simpletype());
    }

    // We're doing a fixed query, so instead of running at every shade,
    // perform the search now.
    const int maxconst = 256;   // Max number of points to consider a constant
    int indices[maxconst + 1];  // Make room for one more!
    float distances[maxconst + 1];
    int maxpoints    = std::min(maxconst + 1, Max_points.get_int());
    ustring filename = Filename.get_string();
    int count        = 0;
    if (!filename.empty()) {
        count = rop.renderer()->pointcloud_search(rop.shaderglobals(), filename,
                                                  Center.get_vec3(),
                                                  Radius.get_float(), maxpoints,
                                                  false, indices, distances, 0);
        rop.shadingsys().pointcloud_stats(1, 0, count);
    }

    // If it returns few enough results (256 points or less), just fold
    // those results into constant arrays.  If more than that, let the
    // query happen at runtime to avoid tying up a bunch of memory.
    if (count > maxconst)
        return 0;

    // If the query returned no matching points, just turn the whole
    // pointcloud_search call into an assignment of 0 to the 'result'.
    if (count < 1) {
        rop.turn_into_assign_zero(op,
                                  "Folded constant pointcloud_search lookup");
        return 1;
    }

    // From here on out, we are able to fold the query (it returned
    // results, but not too many).  Start by removing the original
    // pointcloud_search call itself from the shader code.
    rop.turn_into_nop(op, "Folded constant pointcloud_search lookup");

    // Now, for each optional individual query, do a pointcloud_get NOW
    // to retrieve it, create a constant array for the shader to hold
    // those results, and add to the shader an array copy to move it
    // from the constant into the place the shader wanted the query
    // results to go.  (This assignment can be further optimized later
    // on as well, depending on how it's used.)  If any of the individual
    // queries fail now, we will return a failed result in the end.
    std::vector<char> tmp;  // temporary data
    for (int i = 0; i < nattrs; ++i) {
        // We had stashed names, data types, and destinations earlier.
        // Retrieve them now to build a query.
        if (names[i].empty())
            continue;
        void* const_data       = NULL;
        TypeDesc const_valtype = value_types[i];
        tmp.clear();
        tmp.resize(const_valtype.size(), 0);
        const_data = &tmp[0];
        if (names[i] == "index") {
            // "index" is a special case -- it's retrieving the hit point
            // indices, not data on those hit points.
            //
            const_data = indices;
        } else {
            // Named queries.
            bool ok = rop.renderer()->pointcloud_get(rop.shaderglobals(),
                                                     filename, indices, count,
                                                     names[i], const_valtype,
                                                     const_data);
            rop.shadingsys().pointcloud_stats(0, 1, 0);
            if (!ok) {
                count = 0;  // Make it look like an error in the end
                break;
            }
        }
        // Now make a constant array for those results we just retrieved...
        int const_array_sym = rop.add_constant(const_valtype, const_data);
        // ... and add an instruction to copy the constant into the
        // original destination for the query.
        const int args_to_add[] = { value_args[i], const_array_sym };
        rop.insert_code(opnum, u_assign, args_to_add,
                        RuntimeOptimizer::RecomputeRWRanges,
                        RuntimeOptimizer::GroupWithNext);
    }

    // Query results all copied.  The only thing left to do is to assign
    // status (query result count) to the original "result".
    const int args_to_add[] = { result_sym, rop.add_constant(TypeInt, &count) };
    rop.insert_code(opnum, u_assign, args_to_add,
                    RuntimeOptimizer::RecomputeRWRanges,
                    RuntimeOptimizer::GroupWithNext);

    return 1;
}



DECLFOLDER(constfold_pointcloud_get)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    // Symbol& Result     = *rop.opargsym (op, 0);
    Symbol& Filename  = *rop.opargsym(op, 1);
    Symbol& Indices   = *rop.opargsym(op, 2);
    Symbol& Count     = *rop.opargsym(op, 3);
    Symbol& Attr_name = *rop.opargsym(op, 4);
    Symbol& Data      = *rop.opargsym(op, 5);
    if (!(Filename.is_constant() && Indices.is_constant() && Count.is_constant()
          && Attr_name.is_constant()))
        return 0;

    // All inputs are constants -- we can just turn this into an array
    // assignment.

    ustring filename = Filename.get_string();
    int count        = Count.get_int();
    if (filename.empty() || count < 1) {
        rop.turn_into_assign_zero(op, "Folded constant pointcloud_get");
        return 1;
    }

    if (count >= 1024)  // Too many, don't bother folding
        return 0;

    TypeDesc valtype = Data.typespec().simpletype();
    std::vector<char> data(valtype.size());
    int ok = rop.renderer()->pointcloud_get(rop.shaderglobals(), filename,
                                            (int*)Indices.data(), count,
                                            Attr_name.get_string(), valtype,
                                            &data[0]);
    rop.shadingsys().pointcloud_stats(0, 1, 0);

    rop.turn_into_assign(op, rop.add_constant(TypeInt, &ok),
                         "Folded constant pointcloud_get");

    // Now make a constant array for those results we just retrieved...
    if (valtype == TypeDesc::STRING) {
        // Renderer services treat strings as ustringhash,
        // so we need to convert it back to ustring before continuing
        auto uh_data = reinterpret_cast<ustringhash*>(data.data())[0];
        reinterpret_cast<ustring*>(data.data())[0] = ustring_from(uh_data);
    }
    int const_array_sym = rop.add_constant(valtype, &data[0]);
    // ... and add an instruction to copy the constant into the
    // original destination for the query.
    const int args_to_add[] = { rop.oparg(op, 5) /* Data symbol*/,
                                const_array_sym };
    rop.insert_code(opnum, u_assign, args_to_add,
                    RuntimeOptimizer::RecomputeRWRanges,
                    RuntimeOptimizer::GroupWithNext);
    return 1;
}



DECLFOLDER(constfold_noise)
{
    Opcode& op(rop.inst()->ops()[opnum]);

    // Decode some info about which noise function we're dealing with
    //    bool periodic = (op.opname() == Strings::pnoise);
    int arg        = 0;  // Next arg to read
    Symbol& Result = *rop.opargsym(op, arg++);
    int outdim     = Result.typespec().is_triple() ? 3 : 1;
    Symbol* Name   = rop.opargsym(op, arg++);
    ustring name;
    if (Name->typespec().is_string()) {
        if (Name->is_constant())
            name = Name->get_string();
    } else {
        // Not a string, must be the old-style noise/pnoise
        --arg;  // forget that arg
        Name = NULL;
        name = op.opname();
    }

    // Noise with name that is not a constant at osl-compile-time was marked
    // as taking the derivs of its coordinate arguments. If at this point we
    // can determine that the name is known and not "gabor", when we can
    // turn its derivative taking off.
    if (op.argtakesderivs_all() && name.length() && name != "gabor")
        op.argtakesderivs_all(0);

    // Gabor noise is the only one that takes optional arguments, so
    // optimize them away for other noise types.
    if (name.length() && name != "gabor") {
        for (int a = arg; a < op.nargs(); ++a) {
            // Advance until we hit a string argument, which will be the
            // first optional token/value pair. Then just turn all arguments
            // from that point on into empty strings, which will later be
            // skipped, and in the mean time will eliminate the dependencies
            // on whatever values were previously passed.
            if (rop.opargsym(op, a)->typespec().is_string()) {
                for (; a < op.nargs(); a += 2) {
                    OSL_ASSERT(a + 1 < op.nargs());
                    int cind = rop.add_constant(ustring());
                    rop.inst()->args()[op.firstarg() + a]     = cind;
                    rop.inst()->args()[op.firstarg() + a + 1] = cind;
                }
            }
        }
    }

    // Early out: for now, we only fold cell noise
    if (name != u_cellnoise && name != u_cell)
        return 0;

    // Take an early out if any args are not constant (other than the result)
    for (int i = 1; i < op.nargs(); ++i)
        if (!rop.opargsym(op, i)->is_constant())
            return 0;

    // Extract the constant input coordinates
    float input[4];
    int indim = 0;
    for (; arg < op.nargs() && indim < 4; ++arg) {
        Symbol* in = rop.opargsym(op, arg);
        if (in->typespec().is_float()) {
            input[indim++] = in->get_float(0);
        } else if (in->typespec().is_triple()) {
            input[indim++] = in->get_float(0);
            input[indim++] = in->get_float(1);
            input[indim++] = in->get_float(2);
        } else
            return 0;  // optional args starting, we don't fold them yet
    }

#if OSL_GNUC_VERSION >= 90000
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    if (name == u_cellnoise || name == u_cell) {
        CellNoise cell;
        if (outdim == 1) {
            float n;
            if (indim == 1)
                cell(n, input[0]);
            else if (indim == 2)
                cell(n, input[0], input[1]);
            else if (indim == 3)
                cell(n, Vec3(input[0], input[1], input[2]));
            else
                cell(n, Vec3(input[0], input[1], input[2]), input[3]);
            int cind = rop.add_constant(n);
            rop.turn_into_assign(op, cind, "const fold cellnoise");
            return 1;
        } else {
            OSL_DASSERT(outdim == 3);
            Vec3 n;
            if (indim == 1)
                cell(n, input[0]);
            else if (indim == 2)
                cell(n, input[0], input[1]);
            else if (indim == 3)
                cell(n, Vec3(input[0], input[1], input[2]));
            else
                cell(n, Vec3(input[0], input[1], input[2]), input[3]);
            int cind = rop.add_constant(TypePoint, &n);
            rop.turn_into_assign(op, cind, "const fold cellnoise");
            return 1;
        }
    }
#if OSL_GNUC_VERSION >= 90000
#    pragma GCC diagnostic pop
#endif

    return 0;
}



DECLFOLDER(constfold_functioncall)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    // Make a "functioncall" block disappear if the only non-nop statements
    // inside it is 'return'.
    bool has_return        = false;
    bool has_anything_else = false;
    for (int i = opnum + 1, e = op.jump(0); i < e; ++i) {
        Opcode& op(rop.inst()->ops()[i]);
        if (op.opname() == u_return)
            has_return = true;
        else if (op.opname() != u_nop)
            has_anything_else = true;
    }
    int changed = 0;
    if (!has_anything_else) {
        // Possibly due to optimizations, there's nothing in the
        // function body but the return.  So just eliminate the whole
        // block of ops.
        for (int i = opnum, e = op.jump(0); i < e; ++i) {
            if (rop.inst()->ops()[i].opname() != u_nop) {
                rop.turn_into_nop(rop.inst()->ops()[i], "empty function");
                ++changed;
            }
        }
    } else if (!has_return) {
        // The function is just a straight-up execution, no return
        // statement, so kill the "function" op.
        if (rop.keep_no_return_function_calls()) {
            rop.turn_into_functioncall_nr(
                op, "'functioncall' transmuted to 'no return' version");
        } else {
            rop.turn_into_nop(op, "'function' not necessary");
        }
        ++changed;
    }

    return changed;
}



DECLFOLDER(constfold_useparam)
{
    // Just eliminate useparam (from shaders compiled with old oslc)
    Opcode& op(rop.inst()->ops()[opnum]);
    rop.turn_into_nop(op);
    return 1;
}



DECLFOLDER(constfold_assign)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol* B(rop.inst()->argsymbol(op.firstarg() + 1));
    int Aalias = rop.block_alias(rop.inst()->arg(op.firstarg() + 0));
    Symbol* AA = rop.inst()->symbol(Aalias);
    // N.B. symbol() returns NULL if alias is < 0

    if (B->is_constant() && AA && AA->is_constant()) {
        // Try to turn A=C into nop if A already is C
        if (AA->typespec().is_int() && B->typespec().is_int()) {
            if (AA->get_int() == B->get_int()) {
                rop.turn_into_nop(op, "reassignment of current value");
                return 1;
            }
        } else if (AA->typespec().is_float() && B->typespec().is_float()) {
            if (AA->get_float() == B->get_float()) {
                rop.turn_into_nop(op, "reassignment of current value");
                return 1;
            }
        } else if (AA->typespec().is_float() && B->typespec().is_int()) {
            if (AA->get_float() == B->get_int()) {
                rop.turn_into_nop(op, "reassignment of current value");
                return 1;
            }
        } else if (AA->typespec().is_triple() && B->typespec().is_triple()) {
            if (AA->get_vec3() == B->get_vec3()) {
                rop.turn_into_nop(op, "reassignment of current value");
                return 1;
            }
        } else if (AA->typespec().is_triple() && B->typespec().is_float()) {
            float b = B->get_float();
            if (AA->get_vec3() == Vec3(b, b, b)) {
                rop.turn_into_nop(op, "reassignment of current value");
                return 1;
            }
        }
    }
    return 0;
}



DECLFOLDER(constfold_warning)
{
    if (rop.shadingsys().max_warnings_per_thread() == 0) {
        Opcode& op(rop.inst()->ops()[opnum]);
        rop.turn_into_nop(op,
                          "warnings disabled by max_warnings_per_thread == 0");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_deriv)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    if (A.is_constant()) {
        rop.turn_into_assign_zero(op, "deriv of constant => 0");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_isconstant)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& A(*rop.inst()->argsymbol(op.firstarg() + 1));
    // If at this point we know it's a constant, it's certainly a constant,
    // so we can constant fold it. Note that if it's not known to be a
    // constant at this point, that doesn't mean we won't detect it to be
    // constant after further optimization, so we never fold this to 0.
    if (A.is_constant()) {
        rop.turn_into_assign_one(op, "isconstant => 1");
        return 1;
    }
    return 0;
}



DECLFOLDER(constfold_raytype)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Name = *rop.opargsym(op, 1);
    OSL_DASSERT(Name.typespec().is_string());
    if (!Name.is_constant())
        return 0;  // Can't optimize non-constant raytype name

    int bit = rop.shadingsys().raytype_bit(Name.get_string());
    if (bit & rop.raytypes_on()) {
        rop.turn_into_assign_one(op, "raytype => 1");
        return 1;
    }
    if (bit & rop.raytypes_off()) {
        rop.turn_into_assign_zero(op, "raytype => 0");
        return 1;
    }
    return 0;  // indeterminate until execution time
}


};  // namespace pvt
OSL_NAMESPACE_END
