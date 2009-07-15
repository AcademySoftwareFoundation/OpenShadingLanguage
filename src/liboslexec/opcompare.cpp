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
class Equal {
public:
    inline int operator() (const A &a, const B &b) { return (a == b); }
};

template<class A, class B>
class NotEqual {
public:
    inline int operator() (const A &a, const B &b) { return (a != b); }
};

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




DECLOP (OP_eq)
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
        if (A.typespec().is_triple()) {
            if (B.typespec().is_triple())
                impl = binary_op<int,Vec3,Vec3, Equal<Vec3,Vec3> >;
            else if (B.typespec().is_float())
                impl = binary_op<int,VecProxy,float, Equal<VecProxy,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<int,VecProxy,int, Equal<VecProxy,int> >;
        } else if (A.typespec().is_float()) {
            if (B.typespec().is_triple())
                impl = binary_op<int,float,VecProxy, Equal<float,VecProxy> >;
            else if (B.typespec().is_float())
                impl = binary_op<int,float,float, Equal<float,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<int,float,int, Equal<float,int> >;
            else if (B.typespec().is_matrix())
                impl = binary_op<int,float,MatrixProxy, Equal<float,MatrixProxy> >;
        } if (A.typespec().is_int()) {
            if (B.typespec().is_triple())
                impl = binary_op<int,int,VecProxy, Equal<int,VecProxy> >;
            else if (B.typespec().is_float())
                impl = binary_op<int,int,float, Equal<int,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<int,int,int, Equal<int,int> >;
            else if (B.typespec().is_matrix())
                impl = binary_op<int,int,MatrixProxy, Equal<int,MatrixProxy> >;
        } if (A.typespec().is_matrix()) {
            if (B.typespec().is_float())
                impl = binary_op<int,MatrixProxy,float, Equal<MatrixProxy,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<int,MatrixProxy,int, Equal<MatrixProxy,int> >;
            else if (B.typespec().is_matrix())
                impl = binary_op<int,Matrix44,Matrix44, Equal<Matrix44,Matrix44> >;
        } if (A.typespec().is_string()) {
            if (B.typespec().is_string())
                impl = binary_op<int,ustring,ustring, Equal<ustring,ustring> >;
        }
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
                  << " == " << B.typespec().string() << "\n";
        ASSERT (0 && "comparison types can't be handled");
    }
}



DECLOP (OP_neq)
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
        if (A.typespec().is_triple()) {
            if (B.typespec().is_triple())
                impl = binary_op<int,Vec3,Vec3, NotEqual<Vec3,Vec3> >;
            else if (B.typespec().is_float())
                impl = binary_op<int,VecProxy,float, NotEqual<VecProxy,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<int,VecProxy,int, NotEqual<VecProxy,int> >;
        } else if (A.typespec().is_float()) {
            if (B.typespec().is_triple())
                impl = binary_op<int,float,VecProxy, NotEqual<float,VecProxy> >;
            else if (B.typespec().is_float())
                impl = binary_op<int,float,float, NotEqual<float,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<int,float,int, NotEqual<float,int> >;
            else if (B.typespec().is_matrix())
                impl = binary_op<int,float,MatrixProxy, NotEqual<float,MatrixProxy> >;
        } if (A.typespec().is_int()) {
            if (B.typespec().is_triple())
                impl = binary_op<int,int,VecProxy, NotEqual<int,VecProxy> >;
            else if (B.typespec().is_float())
                impl = binary_op<int,int,float, NotEqual<int,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<int,int,int, NotEqual<int,int> >;
            else if (B.typespec().is_matrix())
                impl = binary_op<int,int,MatrixProxy, NotEqual<int,MatrixProxy> >;
        } if (A.typespec().is_matrix()) {
            if (B.typespec().is_float())
                impl = binary_op<int,MatrixProxy,float, NotEqual<MatrixProxy,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<int,MatrixProxy,int, NotEqual<MatrixProxy,int> >;
            else if (B.typespec().is_matrix())
                impl = binary_op<int,Matrix44,Matrix44, NotEqual<Matrix44,Matrix44> >;
        } if (A.typespec().is_string()) {
            if (B.typespec().is_string())
                impl = binary_op<int,ustring,ustring, NotEqual<ustring,ustring> >;
        }
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
                  << " != " << B.typespec().string() << "\n";
        ASSERT (0 && "comparison types can't be handled");
    }
}



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
