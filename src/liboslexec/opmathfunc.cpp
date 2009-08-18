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
    Cos (ShadingExecution *) { }
    inline float operator() (float x) { return cosf (x); }
    inline Vec3 operator() (const Vec3 &x) {
        return Vec3 (cosf (x[0]), cosf (x[1]), cosf (x[2]));
    }
};

class Sin {
public:
    Sin (ShadingExecution *) { }
    inline float operator() (float x) { return sinf (x); }
    inline Vec3 operator() (const Vec3 &x) {
        return Vec3 (sinf (x[0]), sinf (x[1]), sinf (x[2]));
    }
};

class Tan {
public:
    Tan (ShadingExecution *) { }
    inline float operator() (float x) { return tanf (x); }
    inline Vec3 operator() (const Vec3 &x) {
        return Vec3 (tanf (x[0]), tanf (x[1]), tanf (x[2]));
    }
};

// inverse trigonometric functions

class ACos {
public:
    ACos (ShadingExecution *) { }
    inline float operator() (float x) { return safe_acosf (x); }
    inline Vec3 operator() (const Vec3 &x) {
        return Vec3 (safe_acosf (x[0]), safe_acosf (x[1]), safe_acosf (x[2]));
    }
private:
    inline float safe_acosf (float x) {
        if (x >=  1.0f) return 0.0f;
        if (x <= -1.0f) return M_PI;
        return acosf (x);
    }
};

class ASin {
public:
    ASin (ShadingExecution *) { }
    inline float operator() (float x) { return safe_asinf (x); }
    inline Vec3 operator() (const Vec3 &x) {
        return Vec3 (safe_asinf (x[0]), safe_asinf (x[1]), safe_asinf (x[2]));
    }
private:
    static inline float safe_asinf (float x) {
        if (x >=  1.0f) return  M_PI/2;
        if (x <= -1.0f) return -M_PI/2;
        return asinf (x);
    }
};

class ATan {
public:
    ATan (ShadingExecution *) { }
    inline float operator() (float x) { return atanf (x); }
    inline Vec3 operator() (const Vec3 &x) {
        return Vec3 (atanf (x[0]), atanf (x[1]), atanf (x[2]));
    }
};

// Degrees/Radians

class Degrees {
public:
    Degrees (ShadingExecution *) { }
    inline float operator() (float x) { return x*180.0/M_PI; }
    inline Vec3 operator() (const Vec3 &x) { return x*Float(180.0/M_PI); }
};

class Radians {
public:
    Radians (ShadingExecution *) { }
    inline float operator() (float x) { return x*M_PI/180.0; }
    inline Vec3 operator() (Vec3 &x) { return x*Float(M_PI/180.0); }
};


// hyperbolic functions

class Cosh {
public:
    Cosh (ShadingExecution *) { }
    inline float operator() (float x) { return coshf (x); }
    inline Vec3 operator() (const Vec3 &x) {
        return Vec3 (coshf (x[0]), coshf (x[1]), coshf (x[2]));
    }
};

class Sinh {
public:
    Sinh (ShadingExecution *) { }
    inline float operator() (float x) { return sinhf (x); }
    inline Vec3 operator() (const Vec3 &x) {
        return Vec3 (sinhf (x[0]), sinhf (x[1]), sinhf (x[2]));
    }
};

class Tanh {
public:
    Tanh (ShadingExecution *) { }
    inline float operator() (float x) { return tanhf (x); }
    inline Vec3 operator() (const Vec3 &x) {
        return Vec3 (tanhf (x[0]), tanhf (x[1]), tanhf (x[2]));
    }
};

// logarithmic/exponential functions

class Log {
public:
    Log (ShadingExecution *exec) : m_exec(exec) { }
    inline float operator() (float x)          { return safe_log (x, M_E);}
    inline float operator() (float x, float b) { return safe_log (x, b);  }
    inline Vec3  operator() (const Vec3 &x)    { return safe_log (x, M_E);} 
    inline Vec3  operator() (const Vec3 &x, float b) { return safe_log (x, b);}
private:
    inline float safe_log (float f, float b) {
        if (f <= 0.0f || b <= 0.0f || b == 1.0f) {
            m_exec->error ("attempted to compute log(%g, %g)", f, b);
            if (b == 1.0) 
                return std::numeric_limits<float>::max();
            else
                return -std::numeric_limits<float>::max();
        } else {
            // OPT: faster to check if (b==M_E)?
            return logf (f)/ logf (b);
        }
    }
    inline Vec3 safe_log (const Vec3 &x, float b) {
        if (x[0] <= 0.0f || x[1] <= 0.0f || x[2] <= 0.0f || b <= 0.0f || b == 1.0f) {
            m_exec->error ("attempted to compute log(%g %g %g, %g)", x[0], x[1], x[2], b);
            if (b == 0.0) {
                const float neg_flt_max = -std::numeric_limits<float>::max();
                return Vec3 (neg_flt_max, neg_flt_max, neg_flt_max);
            } else if (b == 1.0) {
                const float flt_max = std::numeric_limits<float>::max();
                return Vec3 (flt_max, flt_max, flt_max);
            } else {
                float inv_log_b = 1.0/logf (b);
                float x0 = (x[0] <= 0) ? -std::numeric_limits<float>::max() : logf (x[0])*inv_log_b;
                float x1 = (x[1] <= 0) ? -std::numeric_limits<float>::max() : logf (x[1])*inv_log_b;
                float x2 = (x[2] <= 0) ? -std::numeric_limits<float>::max() : logf (x[2])*inv_log_b;
                return Vec3 (x0, x1, x2);
            }
        } else {
            float inv_log_b = 1.0/logf (b);
            return Vec3 (logf (x[0])*inv_log_b, logf (x[1])*inv_log_b, logf (x[2])*inv_log_b);
        }
    }
    ShadingExecution *m_exec;
};

class Log2 {
public:
    Log2 (ShadingExecution *exec) : m_exec(exec) { }
    inline float operator() (float x) { return safe_log2f (x); }
    inline Vec3  operator() (const Vec3 &x) { return safe_log2f (x); }
private:
    inline float safe_log2f (float f) {
        if (f <= 0.0f) {
            m_exec->error ("attempted to compute log2(%g)", f);
            return -std::numeric_limits<float>::max();
        } else {
            return log2f (f);
        }
    }
    inline Vec3 safe_log2f (const Vec3 &x) {
        if (x[0] <= 0.0f || x[1] <= 0.0f || x[2] <= 0.0f) {
            m_exec->error ("attempted to compute log2(%g %g %g)", x[0], x[1], x[2]);
            float x0 = (x[0] <= 0) ? -std::numeric_limits<float>::max() : log2f (x[0]);
            float x1 = (x[1] <= 0) ? -std::numeric_limits<float>::max() : log2f (x[1]);
            float x2 = (x[2] <= 0) ? -std::numeric_limits<float>::max() : log2f (x[2]);
            return Vec3 (x0, x1, x2);
        } else {
            return Vec3( log2f (x[0]), log2f (x[1]), log2f (x[2]));
        }
    }
    ShadingExecution *m_exec;
};

class Log10 {
public:
    Log10 (ShadingExecution *exec) : m_exec(exec) { }
    inline float operator() (float x) { return safe_log10f (x); }
    inline Vec3  operator() (const Vec3 &x) { return safe_log10f (x); }
private:
    inline float safe_log10f (float f) {
        if (f <= 0.0f) {
            m_exec->error ("attempted to compute log10(%g)", f);
            return -std::numeric_limits<float>::max();
        } else {
            return log10f (f);
        }
    }
    inline Vec3 safe_log10f (const Vec3 &x) {
        if (x[0] <= 0.0f || x[1] <= 0.0f || x[2] <= 0.0f) {
            m_exec->error ("attempted to compute log10(%g %g %g)", x[0], x[1], x[2]);
            float x0 = (x[0] <= 0) ? -std::numeric_limits<float>::max() : log10f (x[0]);
            float x1 = (x[1] <= 0) ? -std::numeric_limits<float>::max() : log10f (x[1]);
            float x2 = (x[2] <= 0) ? -std::numeric_limits<float>::max() : log10f (x[2]);
            return Vec3 (x0, x1, x2);
        } else {
            return Vec3 (log10f (x[0]), log10f (x[1]), log10f (x[2]));
        }
    }
    ShadingExecution *m_exec;
};

class Logb {
public:
    Logb (ShadingExecution *exec) : m_exec(exec) { }
    inline float operator() (float x) { return safe_logbf (x); }
    inline Vec3 operator() (const Vec3 &x) { return safe_logbf (x); }
private:
    inline float safe_logbf (float f) {
        if (f == 0.0f) {
            m_exec->error ("attempted to compute logb(%g)", f);
            return -std::numeric_limits<float>::max();
        } else {
            return logbf (f);
        }
    }
    inline Vec3 safe_logbf (const Vec3 &x) {
        if (x[0] == 0.0f || x[1] == 0.0f || x[2] == 0.0f) {
            m_exec->error ("attempted to compute logb(%g %g %g)", x[0], x[1], x[2]);
            float x0 = (x[0] == 0) ? -std::numeric_limits<float>::max() : logbf (x[0]);
            float x1 = (x[1] == 0) ? -std::numeric_limits<float>::max() : logbf (x[1]);
            float x2 = (x[2] == 0) ? -std::numeric_limits<float>::max() : logbf (x[2]);
            return Vec3 (x0, x1, x2);
        } else {
            return Vec3 (logbf (x[0]), logbf (x[1]), logbf (x[2]));
        }
    }
    ShadingExecution *m_exec;
};

class Exp {
public:
    Exp (ShadingExecution *) { }
    inline float operator() (float x) { return expf (x); }
    inline Vec3 operator() (const Vec3 &x) {
        return Vec3 (expf (x[0]), expf (x[1]), expf (x[2]));
    }
};

class Exp2 {
public:
    Exp2 (ShadingExecution *) { }
    inline float operator() (float x) { return exp2f (x); }
    inline Vec3 operator() (const Vec3 &x) {
        return Vec3 (exp2f (x[0]), exp2f (x[1]), exp2f (x[2]));
    }
};

class Expm1 {
public:
    Expm1 (ShadingExecution *) { }
    inline float operator() (float x) { return expm1f (x); }
    inline Vec3 operator() (const Vec3 &x) {
        return Vec3 (expm1f (x[0]), expm1f (x[1]), expm1f (x[2]));
    }
};

// miscellaneous math ops

class FAbs {
public:
    FAbs (ShadingExecution *) { }
    inline float operator() (float x) { return fabsf (x); }
    inline Vec3 operator() (const Vec3 &x) {
        return Vec3 (fabsf (x[0]), fabsf (x[1]), fabsf (x[2]));
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

DECLOP (OP_asin)
{
    generic_unary_function_shadeop<ASin> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_atan)
{
    generic_unary_function_shadeop<ATan> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_degrees)
{
    generic_unary_function_shadeop<Degrees> (exec, nargs, args, 
                                             runflags, beginpoint, endpoint);
}

DECLOP (OP_radians)
{
    generic_unary_function_shadeop<Radians> (exec, nargs, args, 
                                             runflags, beginpoint, endpoint);
}

DECLOP (OP_cosh)
{
    generic_unary_function_shadeop<Cosh> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_sinh)
{
    generic_unary_function_shadeop<Sinh> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_tanh)
{
    generic_unary_function_shadeop<Tanh> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

// log() function can two forms:
//   T = log(T)
//   T = log(T, float)
//  where T is float or 3-tuple (color/vector...)
DECLOP (OP_log)
{
    ASSERT (nargs == 2 || nargs == 3);
    OpImpl impl = NULL;
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));

    // T = log(T) case
    if (nargs == 2) {
        ASSERT (! Result.typespec().is_closure() && ! A.typespec().is_closure());
        if (Result.typespec().is_triple() && A.typespec().is_triple()) {
            impl = unary_op<Vec3,Vec3, Log>;
        }
        else if (Result.typespec().is_float() && A.typespec().is_float()){
            impl = unary_op<float,float, Log>;
        }
        else {
            std::cerr << "Don't know how compute " << Result.typespec().string()
                      << " = " << exec->op().opname() << "(" 
                      << A.typespec().string() << ")\n";
            ASSERT (0 && "Function arg type can't be handled");
        }
    }

    // T = log(T, float) case
    else if (nargs == 3) {
        Symbol &B (exec->sym (args[2]));
        ASSERT (! Result.typespec().is_closure() && ! A.typespec().is_closure() && ! B.typespec().is_closure());
        if (Result.typespec().is_triple() && A.typespec().is_triple() && B.typespec().is_float()) {
            impl = binary_op<Vec3,Vec3,float, Log>;
        }
        else if (Result.typespec().is_float() && A.typespec().is_float() && B.typespec().is_float()){
            impl = binary_op<float,float,float, Log>;
        }
        else {
            std::cerr << "Don't know how compute " << Result.typespec().string()
                      << " = " << exec->op().opname() << "(" 
                      << A.typespec().string() << ", "
                      << B.typespec().string() << ")\n";
            ASSERT (0 && "Function arg type can't be handled");
        }
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } 
}

DECLOP (OP_log2)
{
    generic_unary_function_shadeop<Log2> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_log10)
{
    generic_unary_function_shadeop<Log10> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_logb)
{
    generic_unary_function_shadeop<Logb> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_exp)
{
    generic_unary_function_shadeop<Exp> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_exp2)
{
    generic_unary_function_shadeop<Exp2> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_expm1)
{
    generic_unary_function_shadeop<Expm1> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_fabs)
{
    generic_unary_function_shadeop<FAbs> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
