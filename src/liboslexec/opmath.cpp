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

// Proxy type that derives from Vec3 but allows some additional operations
// not normally supported by Imath::Vec3.
class VecProxy : public Vec3 {
public:
    VecProxy (float a, float b, float c) : Vec3(a,b,c) { }
    VecProxy (const Vec3& v) : Vec3(v) { }

    friend VecProxy operator+ (const Vec3 &v, float f) {
        return VecProxy (v.x+f, v.y+f, v.z+f);
    }
    friend VecProxy operator+ (float f, const Vec3 &v) {
        return VecProxy (v.x+f, v.y+f, v.z+f);
    }
    friend VecProxy operator- (const Vec3 &v, float f) {
        return VecProxy (v.x-f, v.y-f, v.z-f);
    }
    friend VecProxy operator- (float f, const Vec3 &v) {
        return VecProxy (v.x-f, v.y-f, v.z-f);
    }
    friend VecProxy operator* (const Vec3 &v, int f) {
        return VecProxy (v.x*f, v.y*f, v.z*f);
    }
    friend VecProxy operator* (int f, const Vec3 &v) {
        return VecProxy (v.x*f, v.y*f, v.z*f);
    }
};


// Make a templated functor that encapsulates addition.
template<class R, class A, class B>
class Add {
public:
    inline R operator() (const A &a, const B &b) { return R (a + b); }
};


// Make a templated functor that encapsulates subtraction.
template<class R, class A, class B>
class Sub {
public:
    inline R operator() (const A &a, const B &b) { return R (a - b); }
};


// Make a templated functor that encapsulates multiplication.
template<class R, class A, class B>
class Mul {
public:
    inline R operator() (const A &a, const B &b) { return R (a * b); }
};

template<>
class Mul<Matrix44,Matrix44,int> {
public:
    inline Matrix44 operator() (const Matrix44 &a, int b) {
        return Matrix44 (a * (float)b);
    }
};

template<>
class Mul<Matrix44,int,Matrix44> {
public:
    inline Matrix44 operator() (int a, const Matrix44 &b) {
        return Matrix44 ((float)a * b);
    }
};

// Specialized version for matrix = scalar * scalar
class ScalarMatrixMul {
public:
    inline Matrix44 operator() (float a, float b) {
        float f = a * b;
        return Matrix44 (f,0,0,0, 0,f,0,0, 0,0,f,0, 0,0,0,f);
    }
};

};  // End anonymous namespace




DECLOP (OP_add)
{
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    if (exec->debug()) {
        std::cout << "Executing add!\n";
        std::cout << "  Result is " << Result.typespec().string() 
                  << " " << Result.mangled() << " @ " << (void *)Result.data() << "\n";
        std::cout << "  A is " << A.typespec().string() 
                  << " " << A.mangled() << " @ " << (void*)A.data() << "\n";
        std::cout << "  B is " << B.typespec().string() 
                  << " " << B.mangled() << " @ " << (void*)B.data() << "\n";
    }
    ASSERT (! Result.typespec().is_closure() &&
            ! Result.typespec().is_structure() &&
            ! Result.typespec().is_array());   // Not yet
    ASSERT (! A.typespec().is_closure() &&
            ! A.typespec().is_structure() &&
            ! A.typespec().is_array());   // Not yet
    ASSERT (! B.typespec().is_closure() &&
            ! B.typespec().is_structure() &&
            ! B.typespec().is_array());   // Not yet
    OpImpl impl = NULL;

    if (Result.typespec().is_closure()) {
        // FIXME -- not handled yet
    }

    else if (Result.typespec().is_triple()) {
        if (A.typespec().is_triple()) {
            if (B.typespec().is_triple())
                impl = binary_op<Vec3,Vec3,Vec3, Add<Vec3,Vec3,Vec3> >;
            else if (B.typespec().is_float())
                impl = binary_op<VecProxy,VecProxy,float,
                                 Add<VecProxy,VecProxy,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<VecProxy,VecProxy,int,
                                 Add<VecProxy,VecProxy,int> >;
        } else if (A.typespec().is_float()) {
            if (B.typespec().is_triple())
                impl = binary_op<VecProxy,float,VecProxy,
                                 Add<VecProxy,float,VecProxy> >;
        } if (A.typespec().is_int()) {
            if (B.typespec().is_triple())
                impl = binary_op<VecProxy,int,VecProxy,
                                 Add<VecProxy,int,VecProxy> >;
        }
    } 

    else if (Result.typespec().is_float()) {
        if (A.typespec().is_float() && B.typespec().is_float())
            impl = binary_op<float,float,float, Add<float,float,float> >;
        else if (A.typespec().is_float() && B.typespec().is_int())
            impl = binary_op<float,float,int, Add<float,float,int> >;
        else if (A.typespec().is_int() && B.typespec().is_float())
            impl = binary_op<float,int,float, Add<float,int,float> >;
    }

    else if (Result.typespec().is_int()) {
        if (A.typespec().is_int() && B.typespec().is_int())
            impl = binary_op<int,int,int, Add<int,int,int> >;
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        std::cerr << "Don't know how to add " << Result.typespec().string()
                  << " = " << A.typespec().string() 
                  << " + " << B.typespec().string() << "\n";
        ASSERT (0 && "Addition types can't be handled");
    }
}



DECLOP (OP_sub)
{
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    if (exec->debug()) {
        std::cout << "Executing sub!\n";
        std::cout << "  Result is " << Result.typespec().string() 
                  << " " << Result.mangled() << " @ " << (void *)Result.data() << "\n";
        std::cout << "  A is " << A.typespec().string() 
                  << " " << A.mangled() << " @ " << (void*)A.data() << "\n";
        std::cout << "  B is " << B.typespec().string() 
                  << " " << B.mangled() << " @ " << (void*)B.data() << "\n";
    }
    ASSERT (! Result.typespec().is_closure() &&
            ! Result.typespec().is_structure() &&
            ! Result.typespec().is_array());   // Not yet
    ASSERT (! A.typespec().is_closure() &&
            ! A.typespec().is_structure() &&
            ! A.typespec().is_array());   // Not yet
    ASSERT (! B.typespec().is_closure() &&
            ! B.typespec().is_structure() &&
            ! B.typespec().is_array());   // Not yet
    OpImpl impl = NULL;

    if (Result.typespec().is_closure()) {
        // FIXME -- not handled yet
    }

    else if (Result.typespec().is_triple()) {
        if (A.typespec().is_triple()) {
            if (B.typespec().is_triple())
                impl = binary_op<Vec3,Vec3,Vec3, Sub<Vec3,Vec3,Vec3> >;
            else if (B.typespec().is_float())
                impl = binary_op<VecProxy,VecProxy,float,
                                 Sub<VecProxy,VecProxy,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<VecProxy,VecProxy,int,
                                 Sub<VecProxy,VecProxy,int> >;
        } else if (A.typespec().is_float()) {
            if (B.typespec().is_triple())
                impl = binary_op<VecProxy,float,VecProxy,
                                 Sub<VecProxy,float,VecProxy> >;
        } if (A.typespec().is_int()) {
            if (B.typespec().is_triple())
                impl = binary_op<VecProxy,int,VecProxy,
                                 Sub<VecProxy,int,VecProxy> >;
        }
    } 

    else if (Result.typespec().is_float()) {
        if (A.typespec().is_float() && B.typespec().is_float())
            impl = binary_op<float,float,float, Sub<float,float,float> >;
        else if (A.typespec().is_float() && B.typespec().is_int())
            impl = binary_op<float,float,int, Sub<float,float,int> >;
        else if (A.typespec().is_int() && B.typespec().is_float())
            impl = binary_op<float,int,float, Sub<float,int,float> >;
    }

    else if (Result.typespec().is_int()) {
        if (A.typespec().is_int() && B.typespec().is_int())
            impl = binary_op<int,int,int, Sub<int,int,int> >;
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        std::cerr << "Don't know how to sub " << Result.typespec().string()
                  << " = " << A.typespec().string() 
                  << " + " << B.typespec().string() << "\n";
        ASSERT (0 && "Subtraction types can't be handled");
    }
}



DECLOP (OP_mul)
{
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    if (exec->debug()) {
        std::cout << "Executing mul!\n";
        std::cout << "  Result is " << Result.typespec().string() 
                  << " " << Result.mangled() << " @ " << (void *)Result.data() << "\n";
        std::cout << "  A is " << A.typespec().string() 
                  << " " << A.mangled() << " @ " << (void*)A.data() << "\n";
        std::cout << "  B is " << B.typespec().string() 
                  << " " << B.mangled() << " @ " << (void*)B.data() << "\n";
    }
    ASSERT (! Result.typespec().is_closure() &&
            ! Result.typespec().is_structure() &&
            ! Result.typespec().is_array());   // Not yet
    ASSERT (! A.typespec().is_closure() &&
            ! A.typespec().is_structure() &&
            ! A.typespec().is_array());   // Not yet
    ASSERT (! B.typespec().is_closure() &&
            ! B.typespec().is_structure() &&
            ! B.typespec().is_array());   // Not yet
    OpImpl impl = NULL;

    if (Result.typespec().is_closure()) {
        // FIXME -- not handled yet
    }

    else if (Result.typespec().is_triple()) {
        if (A.typespec().is_triple()) {
            if (B.typespec().is_triple())
                impl = binary_op<Vec3,Vec3,Vec3, Mul<Vec3,Vec3,Vec3> >;
            else if (B.typespec().is_float())
                impl = binary_op<VecProxy,VecProxy,float,
                                 Mul<VecProxy,VecProxy,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<VecProxy,VecProxy,int,
                                 Mul<VecProxy,VecProxy,int> >;
        } else if (A.typespec().is_float()) {
            if (B.typespec().is_triple())
                impl = binary_op<VecProxy,float,VecProxy,
                                 Mul<VecProxy,float,VecProxy> >;
        } if (A.typespec().is_int()) {
            if (B.typespec().is_triple())
                impl = binary_op<VecProxy,int,VecProxy,
                                 Mul<VecProxy,int,VecProxy> >;
        }
    } 

    else if (Result.typespec().is_float()) {
        if (A.typespec().is_float() && B.typespec().is_float())
            impl = binary_op<float,float,float, Mul<float,float,float> >;
        else if (A.typespec().is_float() && B.typespec().is_int())
            impl = binary_op<float,float,int, Mul<float,float,int> >;
        else if (A.typespec().is_int() && B.typespec().is_float())
            impl = binary_op<float,int,float, Mul<float,int,float> >;
    }

    else if (Result.typespec().is_int()) {
        if (A.typespec().is_int() && B.typespec().is_int())
            impl = binary_op<int,int,int, Mul<int,int,int> >;
    }

    else if (Result.typespec().is_matrix()) {
        if (A.typespec().is_float()) {
            if (B.typespec().is_float())
                impl = binary_op<Matrix44,float,float, ScalarMatrixMul>;
            else if (B.typespec().is_int())
                impl = binary_op<Matrix44,float,int, ScalarMatrixMul>;
            else if (B.typespec().is_matrix())
                impl = binary_op<Matrix44,float,Matrix44, Mul<Matrix44,float,Matrix44> >;
        } if (A.typespec().is_int()) {
            if (B.typespec().is_float())
                impl = binary_op<Matrix44,int,float, ScalarMatrixMul>;
            else if (B.typespec().is_int())
                impl = binary_op<Matrix44,int,int, ScalarMatrixMul>;
            else if (B.typespec().is_matrix())
                impl = binary_op<Matrix44,int,Matrix44, Mul<Matrix44,int,Matrix44> >;
        } if (A.typespec().is_matrix()) {
            if (B.typespec().is_float())
                impl = binary_op<Matrix44,Matrix44,float, Mul<Matrix44,Matrix44,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<Matrix44,Matrix44,int, Mul<Matrix44,Matrix44,int> >;
            else if (B.typespec().is_matrix())
                impl = binary_op<Matrix44,Matrix44,Matrix44, Mul<Matrix44,Matrix44,Matrix44> >;
        }
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        std::cerr << "Don't know how to mul " << Result.typespec().string()
                  << " = " << A.typespec().string() 
                  << " + " << B.typespec().string() << "\n";
        ASSERT (0 && "Multiplication types can't be handled");
    }
}



#if 0
    // Canonical:
    if (Result.typespec().is_XXX()) { // triple, float, int, matrix, string
        if (A.typespec().is_triple()) {
            if (B.typespec().is_triple())
                impl = binary_op<XXX,Vec3,Vec3, FUNC<XXX,Vec3,Vec3> >;
            else if (B.typespec().is_float())
                impl = binary_op<XXX,Vec3,float, FUNC<XXX,Vec3,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<XXX,Vec3,int, FUNC<XXX,Vec3,int> >;
            else if (B.typespec().is_matrix())
                impl = binary_op<XXX,Vec3,Matrix44, FUNC<XXX,Vec3,Matrix44> >;
        } else if (A.typespec().is_float()) {
            if (B.typespec().is_triple())
                impl = binary_op<XXX,float,Vec3, FUNC<XXX,float,Vec3> >;
            else if (B.typespec().is_float())
                impl = binary_op<XXX,float,float, FUNC<XXX,float,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<XXX,float,int, FUNC<XXX,float,int> >;
            else if (B.typespec().is_matrix())
                impl = binary_op<XXX,float,Matrix44, FUNC<XXX,float,Matrix44> >;
        } if (A.typespec().is_int()) {
            if (B.typespec().is_triple())
                impl = binary_op<XXX,int,Vec3, FUNC<XXX,int,Vec3> >;
            else if (B.typespec().is_float())
                impl = binary_op<XXX,int,float, FUNC<XXX,int,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<XXX,int,int, FUNC<XXX,int,int> >;
            else if (B.typespec().is_matrix())
                impl = binary_op<XXX,int,Matrix44, FUNC<XXX,int,Matrix44> >;
        } if (A.typespec().is_matrix()) {
            if (B.typespec().is_triple())
                impl = binary_op<XXX,Matrix44,Vec3, FUNC<XXX,Matrix44,Vec3> >;
            else if (B.typespec().is_float())
                impl = binary_op<XXX,Matrix44,float, FUNC<XXX,Matrix44,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<XXX,Matrix44,int, FUNC<XXX,Matrix44,int> >;
            else if (B.typespec().is_matrix())
                impl = binary_op<XXX,Matrix44,Matrix44, FUNC<XXX,Matrix44,Matrix44> >;
        } if (A.typespec().is_string()) {
            if (B.typespec().is_string())
                impl = binary_op<XXX,ustring,ustring, FUNC<XXX,ustring,ustring> >;
        }
#endif




}; // namespace pvt
}; // namespace OSL
