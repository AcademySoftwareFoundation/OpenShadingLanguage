/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
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
#include "OpenImageIO/fmath.h"

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {


namespace {

// this helper template calls the scalar versions of the function, Func, for
// each of the Vec'3 components.
template <typename Func> class Vec3Adaptor 
{
public:
    Vec3Adaptor (ShadingExecution *) { }
    inline void operator() (Vec3 &result, const Vec3 &x) { 
       Func func;
       func (result.x, x.x);
       func (result.y, x.y);
       func (result.z, x.z);
    }
    inline void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &a)
    {
        Func func;
        Dual2<float> ax, ay, az;
        func (ax, Dual2<float> (a.val().x, a.dx().x, a.dy().x));
        func (ay, Dual2<float> (a.val().y, a.dx().y, a.dy().y));
        func (az, Dual2<float> (a.val().z, a.dx().z, a.dy().z));
        result.set (Vec3( ax.val(), ay.val(), az.val()),
                    Vec3( ax.dx(),  ay.dx(),  az.dx() ),
                    Vec3( ax.dy(),  ay.dy(),  az.dy() ));
    }
    inline void operator() (Vec3 &result, const Vec3 &a, float &b) { 
       Func func;
       // OPT: this leads to 3 evaluations(?) of b when we really only need 1!!!
       func (result.x, a.x, b);
       func (result.y, a.y, b);
       func (result.z, a.z, b);
    }
    inline void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &a, const Dual2<float> &b)
    {
        Func func;
        Dual2<float> ax, ay, az;
        // OPT: this leads to 3 evaluations(?) of b when we really only need 1!!!
        func (ax, Dual2<float> (a.val().x, a.dx().x, a.dy().x), b);
        func (ay, Dual2<float> (a.val().y, a.dx().y, a.dy().y), b);
        func (az, Dual2<float> (a.val().z, a.dx().z, a.dy().z), b);
        result.set (Vec3( ax.val(), ay.val(), az.val()),
                    Vec3( ax.dx(),  ay.dx(),  az.dx() ),
                    Vec3( ax.dy(),  ay.dy(),  az.dy() ));
    }
    inline void operator() (Vec3 &result, const Vec3 &a, const Vec3 &b) { 
       Func func;
       func (result.x, a.x, b.x);
       func (result.y, a.y, b.y);
       func (result.z, a.z, b.z);
    }
    inline void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &a, const Dual2<Vec3> &b)
    {
        Func func;
        Dual2<float> ax, ay, az;
        // OPT: this leads to 3 evaluations(?) of b when we really only need 1!!!
        func (ax, Dual2<float> (a.val().x, a.dx().x, a.dy().x),
                  Dual2<float> (b.val().x, b.dx().x, b.dy().x) );
        func (ay, Dual2<float> (a.val().y, a.dx().y, a.dy().y),
                  Dual2<float> (b.val().y, b.dx().y, b.dy().y) );
        func (az, Dual2<float> (a.val().z, a.dx().z, a.dy().z),
                  Dual2<float> (b.val().z, b.dx().z, b.dy().z) );
        result.set (Vec3( ax.val(), ay.val(), az.val()),
                    Vec3( ax.dx(),  ay.dx(),  az.dx() ),
                    Vec3( ax.dy(),  ay.dy(),  az.dy() ));
    }
    inline void operator() (Vec3 &result, const Vec3 &a, const Vec3 &b, float c) { 
       Func func;
       func (result.x, a.x, b.x, c);
       func (result.y, a.y, b.y, c);
       func (result.z, a.z, b.z, c);
    }
    inline void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &a, const Dual2<Vec3> &b, const Dual2<float> &c)
    {
        Func func;
        Dual2<float> ax, ay, az;
        // OPT: leads to three potential evaluations of c
        func (ax, Dual2<float> (a.val().x, a.dx().x, a.dy().x),
                  Dual2<float> (b.val().x, b.dx().x, b.dy().x),
                  Dual2<float> (c.val(),   c.dx(),   c.dy()) );
        func (ay, Dual2<float> (a.val().y, a.dx().y, a.dy().y),
                  Dual2<float> (b.val().y, b.dx().y, b.dy().y),
                  Dual2<float> (c.val(),   c.dx(),   c.dy()) );
        func (az, Dual2<float> (a.val().z, a.dx().z, a.dy().z),
                  Dual2<float> (b.val().z, b.dx().z, b.dy().z),
                  Dual2<float> (c.val(),   c.dx(),   c.dy()) );
        result.set (Vec3( ax.val(), ay.val(), az.val()),
                    Vec3( ax.dx(),  ay.dx(),  az.dx() ),
                    Vec3( ax.dy(),  ay.dy(),  az.dy() ));
    }
    inline void operator() (Vec3 &result, const Vec3 &a, const Vec3 &b, const Vec3 &c) { 
       Func func;
       func (result.x, a.x, b.x, c.x);
       func (result.y, a.y, b.y, c.y);
       func (result.z, a.z, b.z, c.z);
    }
    inline void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &a, const Dual2<Vec3> &b, const Dual2<Vec3> &c)
    {
        Func func;
        Dual2<float> ax, ay, az;

        func (ax, Dual2<float> (a.val().x, a.dx().x, a.dy().x),
                  Dual2<float> (b.val().x, b.dx().x, b.dy().x),
                  Dual2<float> (c.val().x, c.dx().x, c.dy().x) );
        func (ay, Dual2<float> (a.val().y, a.dx().y, a.dy().y),
                  Dual2<float> (b.val().y, b.dx().y, b.dy().y),
                  Dual2<float> (c.val().y, c.dx().y, c.dy().y) );
        func (az, Dual2<float> (a.val().z, a.dx().z, a.dy().z),
                  Dual2<float> (b.val().z, b.dx().z, b.dy().z),
                  Dual2<float> (c.val().z, c.dx().z, c.dy().z) );
        result.set (Vec3( ax.val(), ay.val(), az.val()),
                    Vec3( ax.dx(),  ay.dx(),  az.dx() ),
                    Vec3( ax.dy(),  ay.dy(),  az.dy() ));
    }
};

// Functors for the math functions

// regular trigonometric functions

class Cos {
public:
    Cos (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x) { result = std::cos (x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = cos (x); }
};

class Sin {
public:
    Sin (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x) { result = std::sin (x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = sin (x); }
};

class SinCos {
public:
    SinCos () { }

    inline void operator() (float x, float &sine, float &cosine) {
        sincos(x, &sine, &cosine);
    }
    inline void operator() (const Dual2<float> &x, Dual2<float> &sine, Dual2<float> &cosine) {
        float s_f, c_f;
        sincos(x.val(), &s_f, &c_f);
        float xdx = x.dx(), xdy = x.dy(); // x might be aliased
        sine   = Dual2<float>(s_f,  c_f * xdx,  c_f * xdy);
        cosine = Dual2<float>(c_f, -s_f * xdx, -s_f * xdy);
    }
    inline void operator() (const Dual2<float> &x, float &sine, Dual2<float> &cosine) {
        float s_f, c_f;
        sincos(x.val(), &s_f, &c_f);
        float xdx = x.dx(), xdy = x.dy(); // x might be aliased
        sine   = s_f;
        cosine = Dual2<float>(c_f, -s_f * xdx, -s_f * xdy);
    }
    inline void operator() (const Dual2<float> &x, Dual2<float> &sine, float &cosine) {
        float s_f, c_f;
        sincos(x.val(), &s_f, &c_f);
        float xdx = x.dx(), xdy = x.dy(); // x might be aliased
        sine   = Dual2<float>(s_f,  c_f * xdx,  c_f * xdy);
        cosine = c_f;
    }
    inline void operator() (const Vec3 &x, Vec3 &sine, Vec3 &cosine) {
        for (int i = 0; i < 3; i++)
            sincos(x[i], &sine[i], &cosine[i]);
    }
    inline void operator() (const Dual2<Vec3> &x, Dual2<Vec3> &sine, Dual2<Vec3> &cosine) {
        for (int i = 0; i < 3; i++) {
            float s_f, c_f;
            sincos(x.val()[i], &s_f, &c_f);
            float xdx = x.dx()[i], xdy = x.dy()[i]; // x might be aliased
              sine.val()[i] = s_f;   sine.dx()[i] =  c_f * xdx;   sine.dy()[i] =  c_f * xdy;
            cosine.val()[i] = c_f; cosine.dx()[i] = -s_f * xdx; cosine.dy()[i] = -s_f * xdy;
        }
    }
    inline void operator() (const Dual2<Vec3> &x, Vec3 &sine, Dual2<Vec3> &cosine) {
        for (int i = 0; i < 3; i++) {
            float s_f, c_f;
            sincos(x.val()[i], &s_f, &c_f);
            float xdx = x.dx()[i], xdy = x.dy()[i]; // x might be aliased
            sine[i] = s_f;
            cosine.val()[i] = c_f; cosine.dx()[i] = -s_f * xdx; cosine.dy()[i] = -s_f * xdy;
        }
    }
    inline void operator() (const Dual2<Vec3> &x, Dual2<Vec3> &sine, Vec3 &cosine) {
        for (int i = 0; i < 3; i++) {
            float s_f, c_f;
            sincos(x.val()[i], &s_f, &c_f);
            float xdx = x.dx()[i], xdy = x.dy()[i]; // x might be aliased
            sine.val()[i] = s_f; sine.dx()[i] =  c_f * xdx; sine.dy()[i] =  c_f * xdy;
            cosine[i] = c_f;
        }
    }
};

class Tan {
public:
    Tan (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x) { result = std::tan (x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = tan (x); }
};

// inverse trigonometric functions

class ACos {
public:
    ACos (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x) { result = safe_acosf (x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = acos (x); }
private:
    inline float safe_acosf (float x) {
        if (x >=  1.0f) return 0.0f;
        if (x <= -1.0f) return M_PI;
        return std::acos (x);
    }
};

class ASin {
public:
    ASin (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x) { result = safe_asinf (x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = asin (x); }
private:
    inline float safe_asinf (float x) {
        if (x >=  1.0f) return  M_PI/2;
        if (x <= -1.0f) return -M_PI/2;
        return std::asin (x);
    }
};

class ATan {
public:
    ATan (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x) { result = std::atan (x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = atan (x); }
};

class ATan2 {
public:
    ATan2 (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float y, float x) { result = std::atan2 (y, x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &y, const Dual2<float> &x) {
        result = atan2 (y,x);
    }
};

// Degrees/Radians

class Degrees {
public:
    Degrees (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x) { result = x*180.0/M_PI; }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = x*Dual2<float>(180.0f/M_PI); }
};

class Radians {
public:
    Radians (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x) { result = x*M_PI/180.0; }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = x*Dual2<float>(M_PI/180.0); }
};


// hyperbolic functions

class Cosh {
public:
    Cosh (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x) { result = std::cosh (x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = cosh (x); }
};

class Sinh {
public:
    Sinh (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x) { result = std::sinh (x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = sinh (x); }
};

class Tanh {
public:
    Tanh (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x) { result = std::tanh (x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = tanh (x); }
};

// logarithmic/exponential functions

class Log {
public:
    Log (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x) { result = safe_log (x, M_E);}
    inline void operator() (float &result, float x, float b) { result = safe_log (x, b);  }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { 
        result = log (x);
    }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x, const Dual2<float> &b) { 
        result = log (x, b);
    }
private:
    inline float safe_log (float f, float b) {
        if (f <= 0.0f || b <= 0.0f || b == 1.0f) {
            if (b == 1.0) 
                return std::numeric_limits<float>::max();
            else
                return -std::numeric_limits<float>::max();
        } else {
            // OPT: faster to check if (b==M_E)?
            return std::log (f)/ std::log (b);
        }
    }
};

class Log2 {
public:
    Log2 (ShadingExecution *exec = NULL)  { }
    inline void operator() (float &result, float x) { result = safe_log2f (x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = log2 (x); }
    inline float safe_log2f(float x) {
        if (x <= 0.0f)
            return -std::numeric_limits<float>::max();
        else
            return log2f(x);
    }
};

class Log10 {
public:
    Log10 (ShadingExecution *exec = NULL)  { }
    inline void operator() (float &result, float x) { result = safe_log10f (x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = log10 (x); }
private:
    inline float safe_log10f(float x) {
        if (x <= 0.0f)
            return -std::numeric_limits<float>::max();
        else
            return log10f(x);
    }
};

class Logb {
public:
    Logb (ShadingExecution *exec) : m_exec(exec) { }
    inline void operator() (float &result, float x) { result = safe_logbf (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) { result = safe_logbf (x); }
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
    Exp (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x) { result = std::exp (x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = exp (x); }
};

class Exp2 {
public:
    Exp2 (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x) { result = exp2f (x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = exp2 (x); }
};

class Expm1 {
public:
    Expm1 (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x) { result = expm1f (x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = expm1 (x); }
};

class Pow {
public:
    Pow (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x, float y) { result = safe_pow (x, y); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x, const Dual2<float> &y) { 
        result = pow (x, y);
    }
private:
    inline float safe_pow (float x, float y) {
        if (x == 0.0f) 
        {
           if (y == 0.0f)
              return 1.0f;
           else
              return 0.0f;
        }
        else 
        {
           if (x < 0.0f && truncf (y) != y) 
              return 0.0f;
           else 
              return std::pow (x,y);
        }
    }
};

class Erf {
public:
    Erf (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = erff (x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = erf (x); }
};

class Erfc {
public:
    Erfc (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = erfcf (x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = erfc (x); }
};


// miscellaneous math ops

class FAbs {
public:
    FAbs (ShadingExecution *exec = NULL) { }
    inline void operator() (int &result, int x) { result = abs (x); }
    inline void operator() (float &result, float x) { result = fabsf (x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) {
       result = x.val() >= 0 ? x : -x;
    }
};

class Floor {
public:
    Floor (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = floorf (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) {
        result = Vec3 (floorf (x[0]), floorf (x[1]), floorf (x[2]));
    }
};

class Ceil {
public:
    Ceil (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = ceilf (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) {
        result = Vec3 (ceilf (x[0]), ceilf (x[1]), ceilf (x[2]));
    }
};

class Round {
public:
    Round (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = roundf (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) {
        result = Vec3 (roundf (x[0]), roundf (x[1]), roundf (x[2]));
    }
};

class Trunc {
public:
    Trunc (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = truncf (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) {
        result = Vec3 (truncf (x[0]), truncf (x[1]), truncf (x[2]));
    }
};

class Sign {
public:
    Sign (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = sign(x); }
    inline void operator() (Vec3 &result, const Vec3 &x) { result = Vec3(sign(x[0]), sign(x[1]), sign(x[2])); }
private:
    inline float sign (float x) { 
        if (x > 0) return 1.0f; 
        else if (x < 0) return -1.0f; 
        else return 0.0; 
    }
};

class Sqrt {
public:
    Sqrt (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x) { result = safe_sqrt (x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = sqrt(x); }
    inline float safe_sqrt (float f) {
        if (f <= 0.0f) {
            return 0.0f;
        } else {
            return std::sqrt (f);
        }
    }
};

class InverseSqrt {
public:
    InverseSqrt (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x) { result = safe_inversesqrt (x); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x) { result = inversesqrt(x); }
private:
    inline float safe_inversesqrt (float f) {
        if (f <= 0.0f) {
            return 0.0f;
        } else {
            return 1.0f/sqrtf (f);
        }
    }
};

class IsNan {
public:
    IsNan (ShadingExecution *) { }
    inline void operator() (int &result, float x) { result = std::isnan (x); }
};

class IsInf {
public:
    IsInf (ShadingExecution *) { }
    inline void operator() (int &result, float x) { result = std::isinf (x); }
};

class IsFinite {
public:
    IsFinite (ShadingExecution *) { }
    inline void operator() (int &result, float x) { result = std::isfinite (x); }
};

class Clamp {
public:
    Clamp (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x, float minv, float maxv) {
        result = clamp(x, minv, maxv);
    }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x, const Dual2<float> &minx, const Dual2<float> &maxv) { 
        result = dual_clamp(x, minx, maxv);
    }
private:
    inline float clamp(float x, float minv, float maxv) {
        if (x < minv) return minv;
        else if (x > maxv) return maxv;
        else return x;
    }
};

class Max {
public:
    Max (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x, float y) { result = max(x,y); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x, const Dual2<float> &y) { 
        result = dual_max(x, y);
    }
private:
    inline float max (float x, float y) { 
        if (x > y) return x;
        else return y;
    }
};

class Min {
public:
    Min (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x, float y) { result = min(x,y); }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x, const Dual2<float> &y) { 
        result = dual_min (x, y);
    }
private:
    inline float min (float x, float y) { 
        if (x > y) return y;
        else return x;
    }
};

class Mix {
public:
    Mix (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x, float y, float a) {
        result = x*(1.0f-a) + y*a;
    }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x, const Dual2<float> &y, const Dual2<float> &a) {
        result = mix(x, y, a);
    }
};

class Step {
public:
    Step (ShadingExecution *) { }
    inline void operator() (float &result, float edge, float x) { 
        result = (x < edge) ? 0.0f : 1.0f;
    }
};

class Hypot {
public:
    Hypot (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float x, float y) {
        result = std::sqrt (x*x + y*y);
    }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x, const Dual2<float> &y) {
        result = dual_hypot(x, y);
    }
    inline void operator() (float &result, float x, float y, float z) {
        result = std::sqrt (x*x + y*y + z*z);
    }
    inline void operator() (Dual2<float> &result, const Dual2<float> &x, const Dual2<float> &y, const Dual2<float> &z) {
        result = dual_hypot(x, y, z);
    }
};

class Smoothstep {
public:
    Smoothstep (ShadingExecution *exec = NULL) { }
    inline void operator() (float &result, float edge0, float edge1, float x) { 
        if (x < edge0) result = 0.0f;
        else if (x >= edge1) result = 1.0f;
        else {
            float t = (x - edge0)/(edge1 - edge0);
            result = (3.0f-2.0f*t)*(t*t);
        }
    }
    inline void operator() (Dual2<float> &result, const Dual2<float> &edge0, const Dual2<float> &edge1, const Dual2<float> &x) {
        result = smoothstep(edge0, edge1, x);
    }
};

//  vector operations

static inline Vec3 reflect(const Vec3 &I, const Vec3 &N) {
   Vec3 R = I - 2.0f*(N.dot(I))*N;
   return R;
}

static inline Vec3 refract(const Vec3 &I, const Vec3 &N, float eta) {
    if (eta == 1.0f)
       return I;

    Vec3  T;
    float n = 1.0*eta;
    float c1 = -I.dot(N);
    float c2_sqr = 1.0 - n*n*(1.0 - c1*c1);

    if (c2_sqr < 0.0f) {
        // total-internal reflection
        T = Vec3(0,0,0);
    }
    else {
        // refraction
        float c2 = sqrtf(c2_sqr);
        T = n*I + (n*c1 - c2)*N;
    }
    return T;
}

class Reflect {
public:
    Reflect (ShadingExecution *) { }
    inline void operator() (Vec3 &result, const Vec3 &I, const Vec3 &N) {
        result = reflect(I, N);
    }
};

class Refract {
public:
    Refract (ShadingExecution *) { }
    inline void operator() (Vec3 &result, const Vec3 &I, const Vec3 &N, float eta) {
        result = refract(I, N, eta);
    }
};

class Fresnel {
public:
    Fresnel (ShadingExecution *) { }
    inline void operator() (const Vec3 &I, const Vec3 &N, float eta,
                            float &Kr) {
        float Kt;
        Vec3 R, T;
        fresnel (I, N, eta, Kr, Kt, R, T, false);
    }
    inline void operator() (const Vec3 &I, const Vec3 &N, float eta,
                            float &Kr, float &Kt, Vec3 &R, Vec3 &T) {
        fresnel (I, N, eta, Kr, Kt, R, T, true);
    }
private:
    inline float sqr(float x) { return x*x; }
    // Implementation of Fresnel.  See, e.g., Roy Hall, "Illumination and Color
    // in Computer Generated Imagery."
    inline void fresnel (const Vec3 &I, const Vec3 &N, float eta,
                            float &Kr, float &Kt, Vec3 &R, Vec3 &T, bool compute_all) {
        float c = I.dot(N);
        if (c < 0)
            c = -c;
        if (compute_all)
           R = reflect(I, N);
        float g = 1.0f / sqr(eta) - 1.0f + c * c;
        if (g >= 0.0f) {
            g = sqrtf (g);
            float beta = g - c;
            float F = (c * (g+c) - 1.0f) / (c * beta + 1.0f);
            F = 0.5f * (1.0f + sqr(F));
            F *= sqr (beta / (g+c));
            Kr = F;
            if (compute_all) {
                Kt = (1.0f - Kr) * eta*eta;
                // OPT: the following recomputes some of the above values, but it 
                // gives us the same result as if the shader-writer called refract()
                T = refract(I, N, eta);
            }
        } else {
            // total internal reflection
            Kr = 1.0f;
            Kt = 0.0f;
            T = Vec3 (0,0,0);
        }
    }
};

// Generic template for implementing "T func(T)" where T can be either
// float or triple.  This expands to a function that checks the arguments
// for valid type combinations, then dispatches to a further specialized
// one for the individual types (but that doesn't do any more polymorphic
// resolution or sanity checks).
template<class FUNCTION>
DECLOP (generic_unary_function_shadeop_noderivs)
{
    // 2 args, result and input.
    ASSERT (nargs == 2);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    ASSERT (! Result.typespec().is_closure() && ! A.typespec().is_closure());
    OpImpl impl = NULL;

    // We allow two flavors: float = func (float), and triple = func (triple).
    if (Result.typespec().is_triple() && A.typespec().is_triple()) {
        impl = unary_op_noderivs<Vec3,Vec3, FUNCTION >;
    }
    else if (Result.typespec().is_float() && A.typespec().is_float()) {
        impl = unary_op_noderivs<float,float, FUNCTION >;
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        exec->error_arg_types ();
        ASSERT (0 && "Function arg type can't be handled");
    }
}

template<class FUNCTION>
DECLOP (generic_unary_function_shadeop)
{
    // 2 args, result and input.
    ASSERT (nargs == 2);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    ASSERT (! Result.typespec().is_closure() && ! A.typespec().is_closure());
    OpImpl impl = NULL;

    // We allow two flavors: float = func (float), and triple = func (triple).
    if (Result.typespec().is_triple() && A.typespec().is_triple()) {
        impl = unary_op<Vec3,Vec3, Vec3Adaptor<FUNCTION> >;
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
        exec->error_arg_types ();
        ASSERT (0 && "Function arg type can't be handled");
    }
}

// Generic template for implementing "T func(T, T)" where T can be either
// float or triple.  This expands to a function that checks the arguments
// for valid type combinations, then dispatches to a further specialized
// one for the individual types (but that doesn't do any more polymorphic
// resolution or sanity checks).
template<class FUNCTION>
DECLOP (generic_binary_function_shadeop_noderivs)
{
    // 3 args, result and two inputs.
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    ASSERT (! Result.typespec().is_closure() && ! A.typespec().is_closure() && !B.typespec().is_closure());
    OpImpl impl = NULL;

    // We allow two flavors: float = func (float, float), and triple = func (triple, triple)
    if (Result.typespec().is_triple() && A.typespec().is_triple() && B.typespec().is_triple()) {
        impl = binary_op_noderivs<Vec3,Vec3,Vec3, FUNCTION >;
    }
    else if (Result.typespec().is_float() && A.typespec().is_float() && B.typespec().is_float()) {
        impl = binary_op_noderivs<float,float,float, FUNCTION >;
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        exec->error_arg_types ();
        ASSERT (0 && "Function arg type can't be handled");
    }
}

// Generic template for implementing "T func(T, T)" where T can be either
// float or triple.  This expands to a function that checks the arguments
// for valid type combinations, then dispatches to a further specialized
// one for the individual types (but that doesn't do any more polymorphic
// resolution or sanity checks).
template<class FUNCTION>
DECLOP (generic_binary_function_shadeop)
{
    // 3 args, result and two inputs.
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    ASSERT (! Result.typespec().is_closure() && ! A.typespec().is_closure() && !B.typespec().is_closure());
    OpImpl impl = NULL;

    // We allow two flavors: float = func (float, float), and triple = func (triple, triple)
    if (Result.typespec().is_triple() && A.typespec().is_triple() && B.typespec().is_triple()) {
        impl = binary_op<Vec3,Vec3,Vec3, Vec3Adaptor<FUNCTION> >;
    }
    else if (Result.typespec().is_float() && A.typespec().is_float() && B.typespec().is_float()) {
        impl = binary_op<float,float,float, FUNCTION >;
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        exec->error_arg_types ();
        ASSERT (0 && "Function arg type can't be handled");
    }
}

// Generic template for implementing "T func(T, T, T)" where T can be either
// float or triple.  This expands to a function that checks the arguments
// for valid type combinations, then dispatches to a further specialized
// one for the individual types (but that doesn't do any more polymorphic
// resolution or sanity checks).
template<class FUNCTION>
DECLOP (generic_ternary_function_shadeop)
{
    // 3 args, result and two inputs.
    ASSERT (nargs == 4);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    Symbol &C (exec->sym (args[3]));
    ASSERT (! Result.typespec().is_closure() && ! A.typespec().is_closure() && !B.typespec().is_closure() && ! C.typespec().is_closure());
    OpImpl impl = NULL;

    // We allow two flavors: float = func (float, float, float), and triple = func (triple, triple, triple)
    if (Result.typespec().is_triple() && A.typespec().is_triple() && B.typespec().is_triple() && C.typespec().is_triple()) {
        impl = ternary_op<Vec3,Vec3,Vec3,Vec3, Vec3Adaptor<FUNCTION> >;
    }
    else if (Result.typespec().is_float() && A.typespec().is_float() && B.typespec().is_float() && C.typespec().is_float()) {
        impl = ternary_op<float,float,float,float, FUNCTION >;
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        exec->error_arg_types ();
        ASSERT (0 && "Function arg type can't be handled");
    }
}

// Generic template for implementing "T func(T, T, T)" where T can be either
// float or triple.  This expands to a function that checks the arguments
// for valid type combinations, then dispatches to a further specialized
// one for the individual types (but that doesn't do any more polymorphic
// resolution or sanity checks).
template<class FUNCTION>
DECLOP (generic_ternary_function_shadeop_noderivs)
{
    // 3 args, result and two inputs.
    ASSERT (nargs == 4);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    Symbol &C (exec->sym (args[3]));
    ASSERT (! Result.typespec().is_closure() && ! A.typespec().is_closure() && !B.typespec().is_closure() && ! C.typespec().is_closure());
    OpImpl impl = NULL;

    // We allow two flavors: float = func (float, float, float), and triple = func (triple, triple, triple)
    if (Result.typespec().is_triple() && A.typespec().is_triple() && B.typespec().is_triple() && C.typespec().is_triple()) {
        impl = ternary_op_noderivs<Vec3,Vec3,Vec3,Vec3, FUNCTION >;
    }
    else if (Result.typespec().is_float() && A.typespec().is_float() && B.typespec().is_float() && C.typespec().is_float()) {
        impl = ternary_op_noderivs<float,float,float,float, FUNCTION >;
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        exec->error_arg_types ();
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

DECLOP (OP_atan2)
{
    generic_binary_function_shadeop<ATan2> (exec, nargs, args, 
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
            impl = unary_op<Vec3,Vec3, Vec3Adaptor<Log> >;
        }
        else if (Result.typespec().is_float() && A.typespec().is_float()){
            impl = unary_op<float,float, Log>;
        }
        else {
            exec->error_arg_types ();
            ASSERT (0 && "Function arg type can't be handled");
        }
    }

    // T = log(T, float) case
    else if (nargs == 3) {
        Symbol &B (exec->sym (args[2]));
        ASSERT (! Result.typespec().is_closure() && ! A.typespec().is_closure() && ! B.typespec().is_closure());
        if (Result.typespec().is_triple() && A.typespec().is_triple() && B.typespec().is_float()) {
            impl = binary_op<Vec3,Vec3,float, Vec3Adaptor<Log> >;
        }
        else if (Result.typespec().is_float() && A.typespec().is_float() && B.typespec().is_float()){
            impl = binary_op<float,float,float, Log>;
        }
        else {
            exec->error_arg_types ();
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
    generic_unary_function_shadeop_noderivs<Logb> (exec, nargs, args, 
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

// pow() function can two forms:
//   T = pow(T, float)
//  where T is float or 3-tuple (color/vector...)
DECLOP (OP_pow)
{
    ASSERT (nargs == 3);
    OpImpl impl = NULL;
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));

    ASSERT (! Result.typespec().is_closure() && ! A.typespec().is_closure() && ! B.typespec().is_closure());
    if (Result.typespec().is_triple() && A.typespec().is_triple() && B.typespec().is_float()) {
        impl = binary_op<Vec3,Vec3,float, Vec3Adaptor<Pow> >;
    }
    else if (Result.typespec().is_float() && A.typespec().is_float() && B.typespec().is_float()){
        impl = binary_op<float,float,float, Pow>;
    }
    else {
        exec->error_arg_types ();
        ASSERT (0 && "Function arg type can't be handled");
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } 
}

DECLOP (OP_erf)
{
    unary_op<float,float,Erf> (exec, nargs, args, 
                               runflags, beginpoint, endpoint);
}

DECLOP (OP_erfc)
{
    unary_op<float,float,Erfc> (exec, nargs, args, 
                                runflags, beginpoint, endpoint);
}

// The fabs() function can be of the form:
//    T = fabs(T) where T is a Vec, Flt, or int
// NOTE: abs() is an alias for fabs()
DECLOP (OP_fabs)
{
    // 2 args, result and input.
    ASSERT (nargs == 2);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    ASSERT (! Result.typespec().is_closure() && ! A.typespec().is_closure());
    OpImpl impl = NULL;

    // We allow two flavors: float = func (float), and triple = func (triple)
    if (Result.typespec().is_triple() && A.typespec().is_triple()) {
        impl = unary_op<Vec3,Vec3, Vec3Adaptor<FAbs> >;
    }
    else if (Result.typespec().is_float() && A.typespec().is_float()) {
        impl = unary_op<float,float, FAbs >;
    }
    else if (Result.typespec().is_int() && A.typespec().is_int()) {
        impl = unary_op_noderivs<int,int, FAbs >;
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        exec->error_arg_types ();
        ASSERT (0 && "Function arg type can't be handled");
    }
}


DECLOP (OP_floor)
{
    generic_unary_function_shadeop_noderivs<Floor> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_ceil)
{
    generic_unary_function_shadeop_noderivs<Ceil> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_round)
{
    generic_unary_function_shadeop_noderivs<Round> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_trunc)
{
    generic_unary_function_shadeop_noderivs<Trunc> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_sign)
{
    generic_unary_function_shadeop_noderivs<Sign> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_sqrt)
{
    generic_unary_function_shadeop<Sqrt> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_inversesqrt)
{
    generic_unary_function_shadeop<InverseSqrt> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_isnan)
{
    unary_op_noderivs<int,float,IsNan> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_isinf)
{
    unary_op_noderivs<int,float,IsInf> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_isfinite)
{
    unary_op_noderivs<int,float,IsFinite> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_clamp)
{
    generic_ternary_function_shadeop<Clamp> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_max)
{
    generic_binary_function_shadeop<Max> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_min)
{
    generic_binary_function_shadeop<Min> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

// mix() function can two forms:
//   T = log(T, T, T)
//   T = log(T, T, float)
//  where T is float or 3-tuple (color/vector...)
DECLOP (OP_mix)
{
    ASSERT (nargs == 4);
    OpImpl impl = NULL;
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    Symbol &C (exec->sym (args[3]));

    ASSERT (! Result.typespec().is_closure() && ! A.typespec().is_closure() && ! B.typespec().is_closure() && ! C.typespec().is_closure());
    if (Result.typespec().is_triple() && A.typespec().is_triple() && B.typespec().is_triple() && C.typespec().is_triple()) {
        impl = ternary_op<Vec3,Vec3,Vec3,Vec3, Vec3Adaptor<Mix> >;
    }
    else if (Result.typespec().is_triple() && A.typespec().is_triple() && B.typespec().is_triple() && C.typespec().is_float()) {
        impl = ternary_op<Vec3,Vec3,Vec3,float, Vec3Adaptor<Mix> >;
    }
    else if (Result.typespec().is_float() && A.typespec().is_float() && B.typespec().is_float() && C.typespec().is_float()){
        impl = ternary_op<float,float,float,float, Mix>;
    }
    else {
        exec->error_arg_types ();
        ASSERT (0 && "Function arg type can't be handled");
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } 
}

DECLOP (OP_step)
{
    binary_op_noderivs<float,float,float,Step> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

// hypot() function can two forms:
//   float = hypot(float, float)
//   float = hypot(float, float, float)
DECLOP (OP_hypot)
{
    ASSERT (nargs == 3 || nargs == 4);
    OpImpl impl = NULL;
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));

    // float = hypot(float, float) case
    if (nargs == 3) {
        ASSERT (! Result.typespec().is_closure() && ! A.typespec().is_closure() && ! B.typespec().is_closure());
        if (Result.typespec().is_float() && A.typespec().is_float() && B.typespec().is_float()) {
            impl = binary_op<float,float,float, Hypot>;
        }
        else {
            exec->error_arg_types ();
            ASSERT (0 && "Function arg type can't be handled");
        }
    }

    // float = hypot(float, float, float) case
    else if (nargs == 4) {
        Symbol &C (exec->sym (args[3]));
        ASSERT (! Result.typespec().is_closure() && ! A.typespec().is_closure() && ! B.typespec().is_closure() && ! C.typespec().is_closure());
        if (Result.typespec().is_float() && A.typespec().is_float() && B.typespec().is_float() && C.typespec().is_float()) {
            impl = ternary_op<float,float,float,float, Hypot>;
        }
        else {
            exec->error_arg_types ();
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

DECLOP (OP_smoothstep)
{
    ternary_op<float,float,float,float, Smoothstep> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_reflect)
{
    binary_op_noderivs<Vec3,Vec3,Vec3, Reflect> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_refract)
{
    ternary_op_noderivs<Vec3,Vec3,Vec3,float, Refract> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

inline void
fresnel4_op_guts (Symbol &I, Symbol &N, Symbol &eta, Symbol &Kr,
                ShadingExecution *exec, 
                Runflag *runflags, int beginpoint, int endpoint)
{
    // Adjust the result's uniform/varying status
    exec->adjust_varying (Kr, I.is_varying() | N.is_varying() | eta.is_varying(),
                          eta.data() == Kr.data());

    // FIXME -- clear derivs for now, make it right later.
    if (Kr.has_derivs ())
        exec->zero_derivs (Kr);

    // Loop over points, do the operation
    VaryingRef<Vec3> inI ((Vec3 *)I.data(), I.step());
    VaryingRef<Vec3> inN ((Vec3 *)N.data(), N.step());
    VaryingRef<float> inEta ((float *)eta.data(), eta.step());
    VaryingRef<float> outKr ((float *)Kr.data(), Kr.step());
    Fresnel fresnel (exec);
    if (outKr.is_uniform()) {
        // Uniform case
        fresnel (*inI, *inN, *inEta, *outKr);
    } else if (inI.is_uniform() && inN.is_uniform() && inEta.is_uniform()) {
        // Operands are uniform but we're assigning to a varying (it can
        // happen if we're in a conditional).  Take a shortcut by doing
        // the operation only once.
        float tKr;
        fresnel (*inI, *inN, *inEta, tKr);
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                outKr[i] = tKr;
    } else {
        // Fully varying case
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                fresnel (inI[i], inN[i], inEta[i], outKr[i]);
    }
}

DECLOP (fresnel4_op)
{
    Symbol &I   (exec->sym (args[0]));
    Symbol &N   (exec->sym (args[1]));
    Symbol &eta (exec->sym (args[2]));
    Symbol &Kr  (exec->sym (args[3]));
    fresnel4_op_guts(I, N, eta, Kr, exec, runflags, beginpoint, endpoint);
}

inline void
fresnel7_op_guts (Symbol &I, Symbol &N, Symbol &eta, Symbol &Kr, Symbol &Kt, Symbol &R, Symbol &T,
                ShadingExecution *exec, 
                Runflag *runflags, int beginpoint, int endpoint)
{
    // Adjust the results' uniform/varying status
    bool varying_assig = I.is_varying() | N.is_varying() | eta.is_varying();

    exec->adjust_varying (Kr, varying_assig, eta.data() == Kr.data());
    exec->adjust_varying (Kt, varying_assig, eta.data() == Kt.data());
    exec->adjust_varying (R,  varying_assig, I.data()   == R.data() || 
                                             N.data()   == R.data());
    exec->adjust_varying (T,  varying_assig, I.data()   == T.data() || 
                                             N.data()   == T.data());

    // FIXME -- clear derivs for now, make it right later.
    if (Kr.has_derivs ())
        exec->zero_derivs (Kr);
    if (Kt.has_derivs ())
        exec->zero_derivs (Kt);
    if (R.has_derivs ())
        exec->zero_derivs (R);
    if (T.has_derivs ())
        exec->zero_derivs (T);

    // Loop over points, do the operation
    VaryingRef<Vec3>  inI ((Vec3 *)I.data(), I.step());
    VaryingRef<Vec3>  inN ((Vec3 *)N.data(), N.step());
    VaryingRef<float> inEta ((float *)eta.data(), eta.step());
    VaryingRef<float> outKr ((float *)Kr.data(),  Kr.step());
    VaryingRef<float> outKt ((float *)Kt.data(),  Kt.step());
    VaryingRef<Vec3>  outR  ((Vec3 *)R.data(),   R.step());
    VaryingRef<Vec3>  outT  ((Vec3 *)T.data(),   T.step());
    Fresnel fresnel (exec);
    if (outKr.is_uniform() && outKt.is_uniform() && outR.is_uniform() && outT.is_uniform()) {
        // Uniform case
        fresnel (*inI, *inN, *inEta, *outKr, *outKt, *outR, *outT);
    } else if (inI.is_uniform() && inN.is_uniform() && inEta.is_uniform()) {
        // Operands are uniform but we're assigning to a varying (it can
        // happen if we're in a conditional).  Take a shortcut by doing
        // the operation only once.
        float tKr, tKt;
        Vec3  tR, tT;
        fresnel (*inI, *inN, *inEta, tKr, tKt, tR, tT);
        for (int i = beginpoint;  i < endpoint;  ++i) {
            if (runflags[i]) {
                outKr[i] = tKr;
                outKt[i] = tKt;
                outR[i]  = tR;
                outT[i]  = tT;
            }
        }
    } else {
        // Fully varying case
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                fresnel (inI[i], inN[i], inEta[i], outKr[i], outKt[i], outR[i], outT[i]);
    }
}

DECLOP (fresnel7_op)
{
    Symbol &I   (exec->sym (args[0]));
    Symbol &N   (exec->sym (args[1]));
    Symbol &eta (exec->sym (args[2]));
    Symbol &Kr  (exec->sym (args[3]));
    Symbol &Kt  (exec->sym (args[4]));
    Symbol &R   (exec->sym (args[5]));
    Symbol &T   (exec->sym (args[6]));
    fresnel7_op_guts(I, N, eta, Kr, Kt, R, T, exec, runflags, beginpoint, endpoint);
}


DECLOP (OP_fresnel)
{
    ASSERT (nargs == 4 || nargs == 7);
    OpImpl impl = NULL;
    Symbol &I   (exec->sym (args[0]));
    Symbol &N   (exec->sym (args[1]));
    Symbol &eta (exec->sym (args[2]));
    Symbol &Kr  (exec->sym (args[3]));

    ASSERT (! I.typespec().is_closure()   &&
            ! N.typespec().is_closure()   &&
            ! eta.typespec().is_closure() &&
            ! Kr.typespec().is_closure());

    if (nargs == 4) {
        if (I.typespec().is_triple()  &&
            N.typespec().is_triple()  &&
            eta.typespec().is_float() &&
            Kr.typespec().is_float() ) 
        {
            impl = fresnel4_op;
        } else {
            exec->error_arg_types ();
            ASSERT (0 && "Function arg type can't be handled");
        }

    } else if (nargs == 7) {
        Symbol &Kt (exec->sym (args[4]));
        Symbol &R  (exec->sym (args[5]));
        Symbol &T  (exec->sym (args[6]));

        ASSERT (! Kt.typespec().is_closure() &&
                ! R.typespec().is_closure()  &&
                ! T.typespec().is_closure());

        if (I.typespec().is_triple()  &&
            N.typespec().is_triple()  &&
            eta.typespec().is_float() &&
            Kr.typespec().is_float()  &&
            Kt.typespec().is_float()  &&
            R.typespec().is_triple()  &&
            T.typespec().is_triple())
        {
            impl = fresnel7_op;
        } else {
            exec->error_arg_types ();
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


template <class TYPE, class FUNCTION>
DECLOP(sincos_op_guts)
{
    Symbol &A = exec->sym(args[0]);
    Symbol &ResultSin = exec->sym(args[1]);
    Symbol &ResultCos = exec->sym(args[2]);
    // Adjust the result's uniform/varying status
    exec->adjust_varying (ResultSin, A.is_varying(), A.data() == ResultSin.data());
    exec->adjust_varying (ResultCos, A.is_varying(), A.data() == ResultCos.data());

    // Loop over points, do the operation
    FUNCTION function;
    if (ResultSin.is_uniform() && ResultCos.is_uniform()) {
        // Uniform case
        function (*(TYPE *)A.data(), *((TYPE *)ResultSin.data()), *((TYPE *)ResultCos.data()));
        if (ResultSin.has_derivs()) exec->zero_derivs (ResultSin);
        if (ResultCos.has_derivs()) exec->zero_derivs (ResultCos);
    } else if (A.is_uniform()) {
        // Should have been uniform, but we are varying because of being inside
        // a conditional. Only execute function once
        TYPE rs, rc;
        function (*(TYPE *)A.data(), rs, rc);
        VaryingRef<TYPE> resultSin ((TYPE *)ResultSin.data(), ResultSin.step());
        VaryingRef<TYPE> resultCos ((TYPE *)ResultCos.data(), ResultCos.step());
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i]) {
                resultSin[i] = rs;
                resultCos[i] = rc;
            }
        if (ResultSin.has_derivs()) exec->zero_derivs (ResultSin);
        if (ResultCos.has_derivs()) exec->zero_derivs (ResultCos);
    } else {
        // Fully varying case
        if (A.has_derivs() && (ResultSin.has_derivs() || ResultCos.has_derivs())) {
            if (ResultSin.has_derivs() && ResultCos.has_derivs()) {
                VaryingRef<Dual2<TYPE> > a ((Dual2<TYPE> *)A.data(), A.step());
                VaryingRef<Dual2<TYPE> > resultSin ((Dual2<TYPE> *)ResultSin.data(), ResultSin.step());
                VaryingRef<Dual2<TYPE> > resultCos ((Dual2<TYPE> *)ResultCos.data(), ResultCos.step());
                for (int i = beginpoint;  i < endpoint;  ++i)
                    if (runflags[i])
                        function (a[i], resultSin[i], resultCos[i]);
            } else if (ResultSin.has_derivs()) {
                // only sine has derivs
                VaryingRef<Dual2<TYPE> > a ((Dual2<TYPE> *)A.data(), A.step());
                VaryingRef<Dual2<TYPE> > resultSin ((Dual2<TYPE> *)ResultSin.data(), ResultSin.step());
                VaryingRef<TYPE> resultCos ((TYPE *)ResultCos.data(), ResultCos.step());
                for (int i = beginpoint;  i < endpoint;  ++i)
                    if (runflags[i])
                        function (a[i], resultSin[i], resultCos[i]);
            } else {
                // only cosine has derivs
                VaryingRef<Dual2<TYPE> > a ((Dual2<TYPE> *)A.data(), A.step());
                VaryingRef<TYPE> resultSin ((TYPE *)ResultSin.data(), ResultSin.step());
                VaryingRef<Dual2<TYPE> > resultCos ((Dual2<TYPE> *)ResultCos.data(), ResultCos.step());
                for (int i = beginpoint;  i < endpoint;  ++i)
                    if (runflags[i])
                        function (a[i], resultSin[i], resultCos[i]);
            }
        } else {
            // A doesn't come with derivatives, or both results don't need derivatives
            VaryingRef<TYPE> a ((TYPE *)A.data(), A.step());
            VaryingRef<TYPE> resultSin ((TYPE *)ResultSin.data(), ResultSin.step());
            VaryingRef<TYPE> resultCos ((TYPE *)ResultCos.data(), ResultCos.step());
            for (int i = beginpoint;  i < endpoint;  ++i)
                if (runflags[i])
                    function (a[i], resultSin[i], resultCos[i]);
            if (ResultSin.has_derivs()) exec->zero_derivs (ResultSin);
            if (ResultCos.has_derivs()) exec->zero_derivs (ResultCos);
        }
    }
}

DECLOP (OP_sincos)
{
    ASSERT (nargs == 3);
    Symbol &A (exec->sym (args[0]));
    Symbol &ResultSin (exec->sym (args[1]));
    Symbol &ResultCos (exec->sym (args[2]));

    ASSERT (! ResultSin.typespec().is_closure() &&
            ! ResultCos.typespec().is_closure() &&
            ! A.typespec().is_closure());
    OpImpl impl = NULL;

    // We allow two flavors: float = func (float), and triple = func (triple).
    if (ResultSin.typespec().is_triple() && ResultCos.typespec().is_triple() && A.typespec().is_triple()) {
        impl = sincos_op_guts<Vec3, SinCos >;
    }
    else if (ResultSin.typespec().is_float() && ResultCos.typespec().is_float() && A.typespec().is_float()) {
        impl = sincos_op_guts<float, SinCos >;
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        exec->error_arg_types ();
        ASSERT (0 && "Function arg type can't be handled");
    }
}

}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
