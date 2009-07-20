/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
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

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of basic math operators 
/// such as +, -, *, /, %.
///
/////////////////////////////////////////////////////////////////////////


#include <iostream>

#include "oslexec_pvt.h"
#include "oslops.h"

#include "OpenImageIO/varyingref.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {


namespace {

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

// Make a templated functor that encapsulates division.
template<class R, class A, class B>
class Div {
public:
    inline R operator() (const A &a, const B &b) {
        return (b == 0) ? R (0.0) : R (a / b);
    }
};

// Specialized version for matrix = matrix / matrix
template<>
class Div<Matrix44,Matrix44,Matrix44>
{
public:
    inline Matrix44 operator() (const Matrix44 &a, const Matrix44 &b) {
        return a * b.inverse();
    }
};

// Specialized version for matrix = float / matrix
template<>
class Div<Matrix44,float,Matrix44>
{
public:
    inline Matrix44 operator() (float a, const Matrix44 &b) {
        return a * b.inverse();
    }
};

// Specialized version for matrix = int / matrix
template<>
class Div<Matrix44,int,Matrix44>
{
public:
    inline Matrix44 operator() (int a, const Matrix44 &b) {
        return (float)a * b.inverse();
    }
};

// Specialized version for matrix = matrix / int
template<>
class Div<Matrix44,Matrix44,int>
{
public:
    inline Matrix44 operator() (const Matrix44 &a, int b) {
        return a / (float)b;
    }
};

// Specialized version for matrix = scalar / scalar
class ScalarMatrixDiv {
public:
    inline Matrix44 operator() (float a, float b) {
        float f = (b == 0) ? 0.0 : (a / b);
        return Matrix44 (f,0,0,0, 0,f,0,0, 0,0,f,0, 0,0,0,f);
    }
};

// Make a functor that encapsulates modulus
class Mod {
public:
    inline int operator() (int a, int b) { return (b == 0) ? 0 : (a % b); }
};

// Make a templated functor that encapsulates negation.
template<class R, class A>
class Neg {
public:
    inline R operator() (const A &a) { return R (-a); }
};

};  // End anonymous namespace




DECLOP (OP_add)
{
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
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



DECLOP (OP_div)
{
    // FIXME -- maybe we can speed up div for the case where A is varying
    // and B is uniform, by taking 1/b and mutiplying.

    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
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
                impl = binary_op<Vec3,Vec3,Vec3, Div<Vec3,Vec3,Vec3> >;
            else if (B.typespec().is_float())
                impl = binary_op<VecProxy,VecProxy,float,
                                 Div<VecProxy,VecProxy,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<VecProxy,VecProxy,int,
                                 Div<VecProxy,VecProxy,int> >;
        } else if (A.typespec().is_float()) {
            if (B.typespec().is_triple())
                impl = binary_op<VecProxy,float,VecProxy,
                                 Div<VecProxy,float,VecProxy> >;
        } if (A.typespec().is_int()) {
            if (B.typespec().is_triple())
                impl = binary_op<VecProxy,int,VecProxy,
                                 Div<VecProxy,int,VecProxy> >;
        }
    } 

    else if (Result.typespec().is_float()) {
        if (A.typespec().is_float() && B.typespec().is_float())
            impl = binary_op<float,float,float, Div<float,float,float> >;
        else if (A.typespec().is_float() && B.typespec().is_int())
            impl = binary_op<float,float,int, Div<float,float,int> >;
        else if (A.typespec().is_int() && B.typespec().is_float())
            impl = binary_op<float,int,float, Div<float,int,float> >;
    }

    else if (Result.typespec().is_int()) {
        if (A.typespec().is_int() && B.typespec().is_int())
            impl = binary_op<int,int,int, Div<int,int,int> >;
    }

    else if (Result.typespec().is_matrix()) {
        if (A.typespec().is_float()) {
            if (B.typespec().is_float())
                impl = binary_op<Matrix44,float,float, ScalarMatrixDiv>;
            else if (B.typespec().is_int())
                impl = binary_op<Matrix44,float,int, ScalarMatrixDiv>;
            else if (B.typespec().is_matrix())
                impl = binary_op<Matrix44,float,Matrix44, Div<Matrix44,float,Matrix44> >;
        } if (A.typespec().is_int()) {
            if (B.typespec().is_float())
                impl = binary_op<Matrix44,int,float, ScalarMatrixDiv>;
            else if (B.typespec().is_int())
                impl = binary_op<Matrix44,int,int, ScalarMatrixDiv>;
            else if (B.typespec().is_matrix())
                impl = binary_op<Matrix44,int,Matrix44, Div<Matrix44,int,Matrix44> >;
        } if (A.typespec().is_matrix()) {
            if (B.typespec().is_float())
                impl = binary_op<Matrix44,Matrix44,float, Div<Matrix44,Matrix44,float> >;
            else if (B.typespec().is_int())
                impl = binary_op<Matrix44,Matrix44,int, Div<Matrix44,Matrix44,int> >;
            else if (B.typespec().is_matrix())
                impl = binary_op<Matrix44,Matrix44,Matrix44, Div<Matrix44,Matrix44,Matrix44> >;
        }
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        std::cerr << "Don't know how to div " << Result.typespec().string()
                  << " = " << A.typespec().string() 
                  << " + " << B.typespec().string() << "\n";
        ASSERT (0 && "Division types can't be handled");
    }
}



DECLOP (OP_mod)
{
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
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

    if (Result.typespec().is_int() && A.typespec().is_int() &&
            B.typespec().is_int()) {
        impl = binary_op<int,int,int, Mod >;
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        std::cerr << "Don't know how to mod " << Result.typespec().string()
                  << " = " << A.typespec().string() 
                  << " + " << B.typespec().string() << "\n";
        ASSERT (0 && "Division types can't be handled");
    }
}



DECLOP (OP_neg)
{
    ASSERT (nargs == 2);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    ASSERT (! Result.typespec().is_closure() &&
            ! Result.typespec().is_structure() &&
            ! Result.typespec().is_array());   // Not yet
    ASSERT (! A.typespec().is_closure() &&
            ! A.typespec().is_structure() &&
            ! A.typespec().is_array());   // Not yet
    OpImpl impl = NULL;

    if (Result.typespec().is_closure()) {
        // FIXME -- not handled yet
    }

    else if (Result.typespec().is_triple()) {
        if (A.typespec().is_triple())
            impl = unary_op<Vec3,Vec3, Neg<Vec3,Vec3> >;
        else if (A.typespec().is_float())
            impl = unary_op<VecProxy,float, Neg<VecProxy,float> >;
        else if (A.typespec().is_int())
            impl = unary_op<VecProxy,int, Neg<VecProxy,int> >;
    } 

    else if (Result.typespec().is_float()) {
        if (A.typespec().is_float())
            impl = unary_op<float,float, Neg<float,float> >;
        else if (A.typespec().is_int())
            impl = unary_op<float,int, Neg<float,int> >;
    }

    else if (Result.typespec().is_int() && A.typespec().is_int()) {
        impl = unary_op<int,int, Neg<int,int> >;
    }

    else if (Result.typespec().is_matrix()) {
        if (A.typespec().is_matrix())
            impl = unary_op<Matrix44,Matrix44, Neg<Matrix44,Matrix44> >;
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        std::cerr << "Don't know how to neg " << Result.typespec().string()
                  << " = -" << A.typespec().string() << "\n";
        ASSERT (0 && "Negation types can't be handled");
    }
}


}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
