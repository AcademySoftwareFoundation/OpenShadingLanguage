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
/// Shader interpreter implementation of simple math functions, such
/// as cos, sqrt, log, etc.
///
/////////////////////////////////////////////////////////////////////////


/************ 
 * Instructions for adding new simple math shadeops:
 * 
 * 1. Pick a function.  Here are some that need to be implemented
 *    (please edit this list as you add them):
 *    trig (cos, sin, tan, acos, asin, atan), ceil/floor/round, sign,
 *    degrees/radians, cosh/sinh/tanh, erf/erfc, exp/exp2/expm,
 *    log/log2/log10/logb, isnan/isinf/isfinite, sqrt/inversesqrt.
 *
 * 2. Clone the implementation of OP_cos, rename the function.
 *
 * 3. You'll need to make one or more functors, model them after Cos.
 *
 * 4. Uncomment the DECLOP for your shadeop, in oslops.h.
 *
 * 5. Add your shadeop to the op_name_entries table in master.cpp.
 *
 * 6. Create a test (or add to an existing test of closely-related
 *    functions).  Start by cloning testsuite/tests/trig/...
 *    And don't forget to add the test name to src/CMakeLists.txt
 *    where it says 'TESTSUITE'.
 ************/
 
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

// Functors for the math functions

// regular trigonometric functions

class Cos {
public:
    inline float operator() (float x) { return cosf (x); }
    inline Vec3 operator() (const Vec3 &x) {
        return Vec3 (cosf (x[0]), cosf (x[1]), cosf (x[2]));
    }
};

class Sin {
public:
    inline float operator() (float x) { return sinf (x); }
    inline Vec3 operator() (const Vec3 &x) {
        return Vec3 (sinf (x[0]), sinf (x[1]), sinf (x[2]));
    }
};

class Tan {
public:
    inline float operator() (float x) { return tanf (x); }
    inline Vec3 operator() (const Vec3 &x) {
        return Vec3 (tanf (x[0]), tanf (x[1]), tanf (x[2]));
    }
};

// inverse trigonometric functions

class ACos {
    static inline float safe_acosf(float x) {
        if (x >=  1.0f) return 0.0f;
        if (x <= -1.0f) return M_PI;
        return acosf(x);
    }
public:
    inline float operator() (float x) { return safe_acosf (x); }
    inline Vec3 operator() (const Vec3 &x) {
        return Vec3 (safe_acosf (x[0]), safe_acosf (x[1]), safe_acosf (x[2]));
    }
};



// Generic template for implementing "T func(T)" where T can be either
// float or triple.  This expands to a function that checks the arguments
// for valid type combinations, then dispatches to a further specialized
// one for the individual types (but that doesn't do any more polymorphic
// resolution or sanity checks).
template<class FUNCTION>
DECLOP (generic_unary_function_shadeop)
{
    // 2 args, result and input.
    ASSERT (nargs == 2);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    ASSERT (! Result.typespec().is_closure() && ! A.typespec().is_closure());
    OpImpl impl = NULL;

    // We allow two flavors: float = func (float), and triple = func (triple)
    if (Result.typespec().is_triple() && A.typespec().is_triple()) {
        impl = unary_op<Vec3,Vec3, FUNCTION >;
    }
    else if (Result.typespec().is_float() && A.typespec().is_float()) {
        impl = unary_op<float,float, FUNCTION >;
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        std::cerr << "Don't know how compute " << Result.typespec().string()
                  << " = " << exec->op().opname() << "(" 
                  << A.typespec().string() << ")\n";
        ASSERT (0 && "Function arg type can't be handled");
    }
}

};  // End anonymous namespace




DECLOP (OP_cos)
{
    generic_unary_function_shadeop<Cos> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_sin)
{
    generic_unary_function_shadeop<Sin> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_tan)
{
    generic_unary_function_shadeop<Tan> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_acos)
{
    generic_unary_function_shadeop<ACos> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
