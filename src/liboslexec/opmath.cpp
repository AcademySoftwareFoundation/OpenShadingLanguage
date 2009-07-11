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

// Proxy type that derives from Vec3 but allows addition and subtraction
// of float.
class VecProxy : public Vec3 {
public:
    VecProxy (float a, float b, float c) : Vec3(a,b,c) { }

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
};


// Make a templated functor that encapsulates addition.
template<class R, class A, class B>
class Adder {
public:
    inline R operator() (const A &a, const B &b) { return R (a + b); }
};


// Make a templated functor that encapsulates addition.
template<class R, class A, class B>
class Subber {
public:
    inline R operator() (const A &a, const B &b) { return R (a - b); }
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
                impl = binary_op<Vec3,Vec3,Vec3, Adder<Vec3,Vec3,Vec3> >;
            else if (B.typespec().is_float())
                impl = binary_op<VecProxy,VecProxy,float,
                                 Adder<VecProxy,VecProxy,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<VecProxy,VecProxy,int,
                                 Adder<VecProxy,VecProxy,int> >;
        } else if (A.typespec().is_float()) {
            if (B.typespec().is_triple())
                impl = binary_op<VecProxy,float,VecProxy,
                                 Adder<VecProxy,float,VecProxy> >;
        } if (A.typespec().is_int()) {
            if (B.typespec().is_triple())
                impl = binary_op<VecProxy,int,VecProxy,
                                 Adder<VecProxy,int,VecProxy> >;
        }
    } 

    else if (Result.typespec().is_float()) {
        if (A.typespec().is_float() && B.typespec().is_float())
            impl = binary_op<float,float,float, Adder<float,float,float> >;
        else if (A.typespec().is_float() && B.typespec().is_int())
            impl = binary_op<float,float,int, Adder<float,float,int> >;
        else if (A.typespec().is_int() && B.typespec().is_float())
            impl = binary_op<float,int,float, Adder<float,int,float> >;
    }

    else if (Result.typespec().is_int()) {
        if (A.typespec().is_int() && B.typespec().is_int())
            impl = binary_op<int,int,int, Adder<int,int,int> >;
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
                impl = binary_op<Vec3,Vec3,Vec3, Subber<Vec3,Vec3,Vec3> >;
            else if (B.typespec().is_float())
                impl = binary_op<VecProxy,VecProxy,float,
                                 Subber<VecProxy,VecProxy,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<VecProxy,VecProxy,int,
                                 Subber<VecProxy,VecProxy,int> >;
        } else if (A.typespec().is_float()) {
            if (B.typespec().is_triple())
                impl = binary_op<VecProxy,float,VecProxy,
                                 Subber<VecProxy,float,VecProxy> >;
        } if (A.typespec().is_int()) {
            if (B.typespec().is_triple())
                impl = binary_op<VecProxy,int,VecProxy,
                                 Subber<VecProxy,int,VecProxy> >;
        }
    } 

    else if (Result.typespec().is_float()) {
        if (A.typespec().is_float() && B.typespec().is_float())
            impl = binary_op<float,float,float, Subber<float,float,float> >;
        else if (A.typespec().is_float() && B.typespec().is_int())
            impl = binary_op<float,float,int, Subber<float,float,int> >;
        else if (A.typespec().is_int() && B.typespec().is_float())
            impl = binary_op<float,int,float, Subber<float,int,float> >;
    }

    else if (Result.typespec().is_int()) {
        if (A.typespec().is_int() && B.typespec().is_int())
            impl = binary_op<int,int,int, Subber<int,int,int> >;
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



#if 0
    // Canonical:
    if (Result.typespec().is_XXX()) { // triple, float, int, matrix, string
        if (A.typespec().is_triple()) {
            if (B.typespec().is_triple())
                impl = specialized_add<XXX,Vec3,Vec3>;
            else if (B.typespec().is_float())
                impl = specialized_add<XXX,Vec3,float>;
            else if (B.typespec().is_int())
                impl = specialized_add<XXX,Vec3,int>;
            else if (B.typespec().is_matrix())
                impl = specialized_add<XXX,Vec3,Matrix44>;
        } else if (A.typespec().is_float()) {
            if (B.typespec().is_triple())
                impl = specialized_add<XXX,float,Vec3>;
            else if (B.typespec().is_float())
                impl = specialized_add<XXX,float,float>;
            else if (B.typespec().is_int())
                impl = specialized_add<XXX,float,int>;
            else if (B.typespec().is_matrix())
                impl = specialized_add<XXX,float,Matrix44>;
        } if (A.typespec().is_int()) {
            if (B.typespec().is_triple())
                impl = specialized_add<XXX,int,Vec3>;
            else if (B.typespec().is_float())
                impl = specialized_add<XXX,int,float>;
            else if (B.typespec().is_int())
                impl = specialized_add<XXX,int,int>;
            else if (B.typespec().is_matrix())
                impl = specialized_add<XXX,int,Matrix44>;
        } if (A.typespec().is_matrix()) {
            if (B.typespec().is_triple())
                impl = specialized_add<XXX,Matrix44,Vec3>;
            else if (B.typespec().is_float())
                impl = specialized_add<XXX,Matrix44,float>;
            else if (B.typespec().is_int())
                impl = specialized_add<XXX,Matrix44,int>;
            else if (B.typespec().is_matrix())
                impl = specialized_add<XXX,Matrix44,Matrix44>;
        } if (A.typespec().is_string()) {
            if (B.typespec().is_string())
                impl = specialized_add<XXX,Vec3,ustring>;
        }
#endif




}; // namespace pvt
}; // namespace OSL
