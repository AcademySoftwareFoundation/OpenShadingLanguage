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

#ifndef OSLOPS_H
#define OSLOPS_H

#include "OpenImageIO/typedesc.h"

#include "oslexec.h"
#include "osl_pvt.h"
#include "oslexec_pvt.h"


namespace OSL {
namespace pvt {


/// Macro that defines the arguments to shading opcode implementations
///
#define OPARGSDECL     ShadingExecution *exec, int nargs, const int *args, \
                       Runflag *runflags, int beginpoint, int endpoint

/// Macro that defines the full declaration of a shading opcode
/// implementation
#define DECLOP(name)   void name (OPARGSDECL)


// Declarations of all our shader opcodes follow:

DECLOP (OP_add);
DECLOP (OP_assign);
DECLOP (OP_div);
DECLOP (OP_end);
DECLOP (OP_mul);
DECLOP (OP_neg);
DECLOP (OP_sub);

DECLOP (OP_missing);



// Heavy lifting of the math and other binary ops, this is a templated
// version that knows the types of the arguments and the operation to
// perform (given by a functor).
template <class RET, class ATYPE, class BTYPE, class FUNCTION>
DECLOP (binary_op)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, A.is_varying() | B.is_varying(),
                          A.data() == Result.data() || B.data() == Result.data());

    // Loop over points, do the operation
    VaryingRef<RET> result ((RET *)Result.data(), Result.step());
    VaryingRef<ATYPE> a ((ATYPE *)A.data(), A.step());
    VaryingRef<BTYPE> b ((BTYPE *)B.data(), B.step());
    FUNCTION function;
    if (result.is_uniform()) {
        // Uniform case
        *result = function (*a, *b);
    } else if (A.is_uniform() && B.is_uniform()) {
        // Operands are uniform but we're assigning to a varying (it can
        // happen if we're in a conditional).  Take a shortcut by doing
        // the operation only once.
        RET r = function (*a, *b);
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = r;
    } else {
        // Fully varying case
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = function (a[i], b[i]);
    }
}



// Heavy lifting of the math and other unary ops, this is a templated
// version that knows the types of the arguments and the operation to
// perform (given by a functor).
template <class RET, class ATYPE, class FUNCTION>
DECLOP (unary_op)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, A.is_varying(), A.data() == Result.data());

    // Loop over points, do the operation
    VaryingRef<RET> result ((RET *)Result.data(), Result.step());
    VaryingRef<ATYPE> a ((ATYPE *)A.data(), A.step());
    FUNCTION function;
    if (result.is_uniform()) {
        // Uniform case
        *result = function (*a);
    } else if (A.is_uniform()) {
        // Operands are uniform but we're assigning to a varying (it can
        // happen if we're in a conditional).  Take a shortcut by doing
        // the operation only once.
        RET r = function (*a);
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = r;
    } else {
        // Fully varying case
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = function (a[i]);
    }
}





}; // namespace pvt
}; // namespace OSL


#endif /* OSLOPS_H */
