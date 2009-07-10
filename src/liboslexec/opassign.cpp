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


// Proxy type that derives from Matrix44 but allows assignment of a float
// to mean f*Identity.
class MatrixProxy : public Matrix44 {
public:
    MatrixProxy (float a, float b, float c, float d,
                 float e, float f, float g, float h,
                 float i, float j, float k, float l,
                 float m, float n, float o, float p)
        : Matrix44 (a,b,c,d, e,f,g,h, i,j,k,l, m,n,o,p) { }

    MatrixProxy (float f) : Matrix44 (f,0,0,0, 0,f,0,0, 0,0,f,0, 0,0,0,f) { }

    const MatrixProxy& operator= (float f) {
        *this = MatrixProxy (f);
        return *this;
    }
};



// Heavy lifting of 'assign', this is a specialized version that knows
// the types of the arguments.
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
        // Uniform case
        *result = RET (*src);
    } else {
        // Potentially varying case
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = RET (src[i]);
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
            impl = specialized_assign<Vec3,Vec3>;
        } else if (Src.typespec().is_float()) {
            impl = specialized_assign<Vec3,float>;
        } if (Src.typespec().is_int()) {
            impl = specialized_assign<Vec3,int>;
        } else {
            // Nothing else can be assigned to a triple
        }
    } else if (Result.typespec().is_matrix()) {
        if (Src.typespec().is_matrix()) {
            impl = specialized_assign<Matrix44,Matrix44>;
        } else if (Src.typespec().is_float()) {
            impl = specialized_assign<MatrixProxy,float>;
        } if (Src.typespec().is_int()) {
            impl = specialized_assign<MatrixProxy,int>;
        } else {
            // Nothing else can be assigned to a matrix
        }
    } else if (Result.typespec().is_string()) {
        if (Src.typespec().is_string())
            impl = specialized_assign<ustring,ustring>;
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
