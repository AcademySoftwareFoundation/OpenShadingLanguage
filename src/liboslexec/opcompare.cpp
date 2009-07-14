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


namespace {


template<class A, class B>
class Less {
public:
    inline int operator() (const A &a, const B &b) { return (a < b); }
};

template<class A, class B>
class LessEqual {
public:
    inline int operator() (const A &a, const B &b) { return (a <= b); }
};

template<class A, class B>
class Greater {
public:
    inline int operator() (const A &a, const B &b) { return (a > b); }
};

template<class A, class B>
class GreaterEqual {
public:
    inline int operator() (const A &a, const B &b) { return (a >= b); }
};

};  // End anonymous namespace




DECLOP (OP_lt)
{
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    ASSERT (! Result.typespec().is_closure() &&
            ! Result.typespec().is_structure() &&
            ! Result.typespec().is_array() &&
            Result.typespec().is_int());
    ASSERT (! A.typespec().is_closure() &&
            ! A.typespec().is_structure() &&
            ! A.typespec().is_array());
    ASSERT (! B.typespec().is_closure() &&
            ! B.typespec().is_structure() &&
            ! B.typespec().is_array());
    OpImpl impl = NULL;

    if (Result.typespec().is_int()) {
        if (A.typespec().is_float() && B.typespec().is_float())
            impl = binary_op<int,float,float, Less<float,float> >;
        else if (A.typespec().is_float() && B.typespec().is_int())
            impl = binary_op<int,float,int, Less<float,int> >;
        else if (A.typespec().is_int() && B.typespec().is_float())
            impl = binary_op<int,int,float, Less<int,float> >;
        else if (A.typespec().is_int() && B.typespec().is_int())
            impl = binary_op<int,int,int, Less<int,int> >;
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        std::cerr << "Don't know how to compare " << Result.typespec().string()
                  << " = " << A.typespec().string() 
                  << " < " << B.typespec().string() << "\n";
        ASSERT (0 && "comparison types can't be handled");
    }
}



DECLOP (OP_le)
{
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    ASSERT (! Result.typespec().is_closure() &&
            ! Result.typespec().is_structure() &&
            ! Result.typespec().is_array() &&
            Result.typespec().is_int());
    ASSERT (! A.typespec().is_closure() &&
            ! A.typespec().is_structure() &&
            ! A.typespec().is_array());
    ASSERT (! B.typespec().is_closure() &&
            ! B.typespec().is_structure() &&
            ! B.typespec().is_array());
    OpImpl impl = NULL;

    if (Result.typespec().is_int()) {
        if (A.typespec().is_float() && B.typespec().is_float())
            impl = binary_op<int,float,float, LessEqual<float,float> >;
        else if (A.typespec().is_float() && B.typespec().is_int())
            impl = binary_op<int,float,int, LessEqual<float,int> >;
        else if (A.typespec().is_int() && B.typespec().is_float())
            impl = binary_op<int,int,float, LessEqual<int,float> >;
        else if (A.typespec().is_int() && B.typespec().is_int())
            impl = binary_op<int,int,int, LessEqual<int,int> >;
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        std::cerr << "Don't know how to compare " << Result.typespec().string()
                  << " = " << A.typespec().string() 
                  << " <= " << B.typespec().string() << "\n";
        ASSERT (0 && "comparison types can't be handled");
    }
}



DECLOP (OP_gt)
{
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    ASSERT (! Result.typespec().is_closure() &&
            ! Result.typespec().is_structure() &&
            ! Result.typespec().is_array() &&
            Result.typespec().is_int());
    ASSERT (! A.typespec().is_closure() &&
            ! A.typespec().is_structure() &&
            ! A.typespec().is_array());
    ASSERT (! B.typespec().is_closure() &&
            ! B.typespec().is_structure() &&
            ! B.typespec().is_array());
    OpImpl impl = NULL;

    if (Result.typespec().is_int()) {
        if (A.typespec().is_float() && B.typespec().is_float())
            impl = binary_op<int,float,float, Greater<float,float> >;
        else if (A.typespec().is_float() && B.typespec().is_int())
            impl = binary_op<int,float,int, Greater<float,int> >;
        else if (A.typespec().is_int() && B.typespec().is_float())
            impl = binary_op<int,int,float, Greater<int,float> >;
        else if (A.typespec().is_int() && B.typespec().is_int())
            impl = binary_op<int,int,int, Greater<int,int> >;
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        std::cerr << "Don't know how to compare " << Result.typespec().string()
                  << " = " << A.typespec().string() 
                  << " > " << B.typespec().string() << "\n";
        ASSERT (0 && "comparison types can't be handled");
    }
}



DECLOP (OP_ge)
{
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    ASSERT (! Result.typespec().is_closure() &&
            ! Result.typespec().is_structure() &&
            ! Result.typespec().is_array() &&
            Result.typespec().is_int());
    ASSERT (! A.typespec().is_closure() &&
            ! A.typespec().is_structure() &&
            ! A.typespec().is_array());
    ASSERT (! B.typespec().is_closure() &&
            ! B.typespec().is_structure() &&
            ! B.typespec().is_array());
    OpImpl impl = NULL;

    if (Result.typespec().is_int()) {
        if (A.typespec().is_float() && B.typespec().is_float())
            impl = binary_op<int,float,float, GreaterEqual<float,float> >;
        else if (A.typespec().is_float() && B.typespec().is_int())
            impl = binary_op<int,float,int, GreaterEqual<float,int> >;
        else if (A.typespec().is_int() && B.typespec().is_float())
            impl = binary_op<int,int,float, GreaterEqual<int,float> >;
        else if (A.typespec().is_int() && B.typespec().is_int())
            impl = binary_op<int,int,int, GreaterEqual<int,int> >;
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        std::cerr << "Don't know how to compare " << Result.typespec().string()
                  << " = " << A.typespec().string() 
                  << " >= " << B.typespec().string() << "\n";
        ASSERT (0 && "comparison types can't be handled");
    }
}


}; // namespace pvt
}; // namespace OSL
