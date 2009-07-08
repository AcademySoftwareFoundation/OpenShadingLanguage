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

#include <iostream>

#include "oslexec_pvt.h"
#include "oslops.h"

#include "OpenImageIO/varyingref.h"


namespace OSL {
namespace pvt {



// Heavy lifting of 'assign', this is a specialized version that assumes
// the types of 
template <class RET, class SRC>
static DECLOP (specialized_assign)
{
    if (exec->debug())
        std::cout << "Executing specialized_assign!\n";
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &Src (exec->sym (args[1]));

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, Src.is_varying(),
                          Result.data() == Src.data());

    // Loop over points, do the assignment.
    VaryingRef<RET> result ((RET *)Result.data(), Result.step());
    VaryingRef<SRC> src ((SRC *)Src.data(), Src.step());
    if (result.is_uniform()) {
        // Result (and src) are uniform
        *result = RET (*src);
    } else {
        // Potentially varying case
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = RET (src[i]);
    }
    if (exec->debug()) {
        std::cout << "After assignment, new values are:\n";
        if (result.is_uniform())
            std::cout << "\tuniform " << result[0] << "\n";
        else {
            for (int i = beginpoint;  i < endpoint;  ++i) {
                std::cout << "\t" << i << ": " << (result[i]) << "\n";
                if (i == beginpoint || (i%16 == 0))
                    std::cout << "\t" << i << ": ";
                std::cout << result[i] << ' ';
                if (i == endpoint-1 || (i%16 == 15))
                    std::cout << "\n";
            }
        }
    }
}



// Syntax of matrix assignment of scalar is slightly different because
// Imath::Matrix44 defines operator= as setting all components to the
// scalar, whereas OSL defines the operation as just setting the
// diagonal to the scalar (i.e. m=f is equiv to m = Identity * f)
template <class SRC>
static DECLOP (specialized_assign_matrix_scalar)
{
    if (exec->debug())
        std::cout << "Executing specialized_assign for matrix!\n";
    typedef Imath::M44f RET;
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &Src (exec->sym (args[1]));

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, Src.is_varying(),
                          Result.data() == Src.data());

    // Loop over points, do the assignment.
    VaryingRef<RET> result ((RET *)Result.data(), Result.step());
    VaryingRef<SRC> src ((SRC *)Src.data(), Src.step());
    if (result.is_uniform()) {
        // Result (and src) are uniform
        *result = 1.0f;
        *result *= *src;
    } else {
        // Potentially varying case
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i]) {
                result[i] = 1.0f;
                result[i] *= (src[i]);
            }
    }
    if (exec->debug()) {
        std::cout << "After assignment, new values are:\n";
        for (int i = beginpoint;  i < endpoint;  ++i)
            std::cout << "\t" << i << ": " << (result[i]) << "\n";
    }
}



DECLOP (OP_assign)
{
    if (exec->debug())
        std::cout << "Executing assign!\n";
    ASSERT (nargs == 2);
    Symbol &Result (exec->sym (args[0]));
    Symbol &Src (exec->sym (args[1]));
    if (exec->debug()) {
        std::cout << "  Result is " << Result.typespec().string() 
                  << " " << Result.mangled() << " @ " << (void *)Result.data() << "\n";
        std::cout << "  Src is " << Src.typespec().string() 
                  << " " << Src.mangled() << " @ " << (void*)Src.data() << "\n";
    }
    ASSERT (! Result.typespec().is_closure() &&
            ! Result.typespec().is_structure() &&
            ! Result.typespec().is_array());   // Not yet
    ASSERT (! Src.typespec().is_closure() &&
            ! Src.typespec().is_structure() &&
            ! Src.typespec().is_array());   // Not yet
    OpImpl impl = NULL;
    if (Result.typespec().is_closure()) {
        // FIXME -- not handled yet
    } else if (Result.typespec().is_structure()) {
        // FIXME -- not handled yet
    } else if (Result.typespec().is_float()) {
        if (Src.typespec().is_float()) {
            impl = specialized_assign<float,float>;
        } if (Src.typespec().is_int()) {
            impl = specialized_assign<float,int>;
        } else {
            // Nothing else can be assigned to a float
        }
    } else if (Result.typespec().is_triple()) {
        if (Src.typespec().is_triple()) {
            impl = specialized_assign<Imath::V3f,Imath::V3f>;
        } else if (Src.typespec().is_float()) {
            impl = specialized_assign<Imath::V3f,float>;
        } if (Src.typespec().is_int()) {
            impl = specialized_assign<Imath::V3f,int>;
        } else {
            // Nothing else can be assigned to a float
        }
    } else if (Result.typespec().is_matrix()) {
        if (Src.typespec().is_matrix()) {
            impl = specialized_assign<Imath::M44f,Imath::M44f>;
        } else if (Src.typespec().is_float()) {
            impl = specialized_assign_matrix_scalar<float>;
        } if (Src.typespec().is_int()) {
            impl = specialized_assign_matrix_scalar<int>;
        } else {
            // Nothing else can be assigned to a float
        }
    }
    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
        return;
    }
    ASSERT (0 && "Assignment types can't be handled");
}



}; // namespace pvt
}; // namespace OSL
