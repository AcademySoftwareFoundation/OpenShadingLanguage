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
    inline void operator() (float &result, float x) { result = cosf (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) {
        result = Vec3 (cosf (x[0]), cosf (x[1]), cosf (x[2]));
    }
};

class Sin {
public:
    Sin (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = sinf (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) {
        result = Vec3 (sinf (x[0]), sinf (x[1]), sinf (x[2]));
    }
};

class Tan {
public:
    Tan (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = tanf (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) {
        result = Vec3 (tanf (x[0]), tanf (x[1]), tanf (x[2]));
    }
};

// inverse trigonometric functions

class ACos {
public:
    ACos (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = safe_acosf (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) {
        result = Vec3 (safe_acosf (x[0]), safe_acosf (x[1]), safe_acosf (x[2]));
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
    inline void operator() (float &result, float x) { result = safe_asinf (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) {
        result = Vec3 (safe_asinf (x[0]), safe_asinf (x[1]), safe_asinf (x[2]));
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
    inline void operator() (float &result, float x) { result = atanf (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) {
        result = Vec3 (atanf (x[0]), atanf (x[1]), atanf (x[2]));
    }
};

class ATan2 {
public:
    ATan2 (ShadingExecution *) { }
    inline void operator() (float &result, float y, float x) { result = atan2f (y, x); }
    inline void operator() (Vec3 &result, const Vec3 &y, const Vec3 &x) {
        result = Vec3 (atan2f (y[0], x[0]), atan2f (y[1], x[1]), atan2f (y[2], x[2]));
    }
};

// Degrees/Radians

class Degrees {
public:
    Degrees (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = x*180.0/M_PI; }
    inline void operator() (Vec3 &result, const Vec3 &x) { result = x*Float(180.0/M_PI); }
};

class Radians {
public:
    Radians (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = x*M_PI/180.0; }
    inline void operator() (Vec3 &result, Vec3 &x) { result = x*Float(M_PI/180.0); }
};


// hyperbolic functions

class Cosh {
public:
    Cosh (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = coshf (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) {
        result = Vec3 (coshf (x[0]), coshf (x[1]), coshf (x[2]));
    }
};

class Sinh {
public:
    Sinh (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = sinhf (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) {
        result = Vec3 (sinhf (x[0]), sinhf (x[1]), sinhf (x[2]));
    }
};

class Tanh {
public:
    Tanh (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = tanhf (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) {
        result = Vec3 (tanhf (x[0]), tanhf (x[1]), tanhf (x[2]));
    }
};

// logarithmic/exponential functions

class Log {
public:
    Log (ShadingExecution *exec) : m_exec(exec) { }
    inline void operator() (float &result, float x) { result = safe_log (x, M_E);}
    inline void operator() (float &result, float x, float b) { result = safe_log (x, b);  }
    inline void operator() (Vec3 &result, const Vec3 &x)    { result = safe_log (x, M_E);} 
    inline void operator() (Vec3 &result, const Vec3 &x, float b) { result = safe_log (x, b);}
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
    inline void operator() (float &result, float x) { result = safe_log2f (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) { result = safe_log2f (x); }
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
    inline void operator() (float &result, float x) { result = safe_log10f (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) { result = safe_log10f (x); }
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
    Exp (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = expf (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) {
        result = Vec3 (expf (x[0]), expf (x[1]), expf (x[2]));
    }
};

class Exp2 {
public:
    Exp2 (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = exp2f (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) {
        result = Vec3 (exp2f (x[0]), exp2f (x[1]), exp2f (x[2]));
    }
};

class Expm1 {
public:
    Expm1 (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = expm1f (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) {
        result = Vec3 (expm1f (x[0]), expm1f (x[1]), expm1f (x[2]));
    }
};

class Pow {
public:
    Pow (ShadingExecution *exec) : m_exec(exec) { }
    inline void operator() (float &result, float x, float y) { result = safe_pow (x, y); }
    inline void operator() (Vec3 &result, const Vec3 &x, float y) { result = safe_pow (x, y); }
private:
    inline float safe_pow (float x, float y) {
        if (x <= 0.0f &&  (y < 0.0f  || truncf(y) != y) ) {
            m_exec->error ("attempted to compute pow(%g, %g)", x, y);
           return  0.0f;
        } else {
            return powf (x, y);
        }
    }
    inline Vec3 safe_pow (const Vec3 &x, float y) {
        if ( (x[0] <= 0.0f || x[1] <= 0.0f || x[2] <= 0.0f) && 
              (y < 0.0f || truncf(y) != y) ) {
            m_exec->error ("attempted to compute log(%g %g %g, %g)", x[0], x[1], x[2], y);
            float x0 = (x[0] <= 0) ? 0.0f : powf (x[0], y);
            float x1 = (x[1] <= 0) ? 0.0f : powf (x[1], y);
            float x2 = (x[2] <= 0) ? 0.0f : powf (x[2], y);
            return Vec3 (x0, x1, x2);
        } else {
            return Vec3 (powf (x[0], y), powf (x[1], y), powf (x[2], y));
        }
    }
    ShadingExecution *m_exec;
};

class Erf {
public:
    Erf (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = erff(x); }
};

class Erfc {
public:
    Erfc (ShadingExecution *) { }
    inline void operator() (float &result, float x) { result = erfcf(x); }
};


// miscellaneous math ops

class FAbs {
public:
    FAbs (ShadingExecution *) { }
    inline void operator() (int &result, int x) { result = abs (x); }
    inline void operator() (float &result, float x) { result = fabsf (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) {
        result = Vec3 (fabsf (x[0]), fabsf (x[1]), fabsf (x[2]));
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
    Sqrt (ShadingExecution *exec) : m_exec(exec) { }
    inline void operator() (float &result, float x) { result = safe_sqrt (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) { result = safe_sqrt(x); }
private:
    inline float safe_sqrt (float f) {
        if (f < 0.0f) {
            m_exec->error ("attempted to compute sqrt(%g)", f);
            return 0.0f;
        } else {
            return sqrtf (f);
        }
    }
    inline Vec3 safe_sqrt (const Vec3 &x) {
        if (x[0] < 0.0f || x[1] < 0.0f || x[2] < 0.0f) {
            m_exec->error ("attempted to compute sqrt(%g %g %g)", x[0], x[1], x[2]);
            float x0 = (x[0] < 0) ? 0.0f : sqrtf (x[0]);
            float x1 = (x[1] < 0) ? 0.0f : sqrtf (x[1]);
            float x2 = (x[2] < 0) ? 0.0f : sqrtf (x[2]);
            return Vec3 (x0, x1, x2);
        } else {
            return Vec3( sqrtf (x[0]), sqrtf (x[1]), sqrtf (x[2]));
        }
    }
    ShadingExecution *m_exec;
};

class InverseSqrt {
public:
    InverseSqrt (ShadingExecution *exec) : m_exec(exec) { }
    inline void operator() (float &result, float x) { result = safe_invsqrt (x); }
    inline void operator() (Vec3 &result, const Vec3 &x) { result = safe_invsqrt(x); }
private:
    inline float safe_invsqrt (float f) {
        if (f <= 0.0f) {
            m_exec->error ("attempted to compute inversesqrt(%g)", f);
            return 0.0f;
        } else {
            return 1.0f/sqrtf (f);
        }
    }
    inline Vec3 safe_invsqrt (const Vec3 &x) {
        if (x[0] <= 0.0f || x[1] <= 0.0f || x[2] <= 0.0f) {
            m_exec->error ("attempted to compute inversesqrt(%g %g %g)", x[0], x[1], x[2]);
            float x0 = (x[0] <= 0) ? 0.0f : 1.0f/sqrtf (x[0]);
            float x1 = (x[1] <= 0) ? 0.0f : 1.0f/sqrtf (x[1]);
            float x2 = (x[2] <= 0) ? 0.0f : 1.0f/sqrtf (x[2]);
            return Vec3 (x0, x1, x2);
        } else {
            return Vec3( 1.0f/sqrtf (x[0]), 1.0f/sqrtf (x[1]), 1.0f/sqrtf (x[2]));
        }
    }
    ShadingExecution *m_exec;
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
    Clamp (ShadingExecution *) { }
    inline void operator() (float &result, float x, float minv, float maxv) {
        result = clamp(x, minv, maxv);
    }
    inline void operator() (Vec3 &result, const Vec3 &x, const Vec3 &minv, const Vec3 &maxv) {
        result = clamp(x, minv, maxv);
    }
private:
    inline float clamp(float x, float minv, float maxv) {
        if (x < minv) return minv;
        else if (x > maxv) return maxv;
        else return x;
    }
    inline Vec3 clamp(const Vec3 &x, const Vec3 &minv, const Vec3 &maxv) {
        float x0 = clamp(x[0], minv[0], maxv[0]);
        float x1 = clamp(x[1], minv[1], maxv[1]);
        float x2 = clamp(x[2], minv[2], maxv[2]);
        return Vec3 (x0, x1, x2);
    }
};

class Max {
public:
    Max (ShadingExecution *) { }
    inline void operator() (float &result, float x, float y) { result = max(x,y); }
    inline void operator() (Vec3 &result, const Vec3 &x, const Vec3 &y) { 
        result = Vec3 (max (x[0], y[0]), max (x[1], y[1]), max (x[2], y[2]));
    }
private:
    inline float max (float x, float y) { 
        if (x > y) return x;
        else return y;
    }
};

class Min {
public:
    Min (ShadingExecution *) { }
    inline void operator() (float &result, float x, float y) { result = min(x,y); }
    inline void operator() (Vec3 &result, const Vec3 &x, const Vec3 &y) { 
        result = Vec3 (min (x[0], y[0]), min (x[1], y[1]), min (x[2], y[2]));
    }
private:
    inline float min (float x, float y) { 
        if (x > y) return y;
        else return x;
    }
};

class Mix {
public:
    Mix (ShadingExecution *) { }
    inline void operator() (float &result, float x, float y, float a) {
        result = x*(1.0f-a) + y*a;
    }
    inline void operator() (Vec3 &result, const Vec3 &x, const Vec3 &y, float a) {
        result = x*(1.0f-a) + y*a;
    }
    inline void operator() (Vec3 &result, const Vec3 &x, const Vec3 &y, const Vec3 &a) { 
        Vec3 one(1.0f, 1.0f, 1.0f);
        result = x*(one-a) + y*a;
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
    Hypot (ShadingExecution *) { }
    inline void operator() (float &result, float x, float y) {
        result = sqrtf (x*x + y*y);
    }
    inline void operator() (float &result, float x, float y, float z) {
        result = sqrtf (x*x + y*y + z*z);
    }
};

class Smoothstep {
public:
    Smoothstep (ShadingExecution *) { }
    inline void operator() (float &result, float edge0, float edge1, float x) { 
        if (x < edge0) result = 0.0f;
        else if (x >= edge1) result = 1.0f;
        else {
            float t = (x - edge0)/(edge1 - edge0);
            result = (3.0f-2.0f*t)*(t*t);
        }
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
        std::cerr << "Don't know how compute " << Result.typespec().string()
                  << " = " << exec->op().opname() << "(" 
                  << A.typespec().string() << ")\n";
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
        std::cerr << "Don't know how compute " << Result.typespec().string()
                  << " = " << exec->op().opname() << "(" 
                  << A.typespec().string() << ", "
                  << B.typespec().string() << ")\n";
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
        impl = ternary_op<Vec3,Vec3,Vec3,Vec3, FUNCTION >;
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
        std::cerr << "Don't know how compute " << Result.typespec().string()
                  << " = " << exec->op().opname() << "(" 
                  << A.typespec().string() << ", "
                  << B.typespec().string() << ". "
                  << C.typespec().string() << ")\n";
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
            impl = unary_op_noderivs<Vec3,Vec3, Log>;
        }
        else if (Result.typespec().is_float() && A.typespec().is_float()){
            impl = unary_op_noderivs<float,float, Log>;
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
            impl = binary_op_noderivs<Vec3,Vec3,float, Log>;
        }
        else if (Result.typespec().is_float() && A.typespec().is_float() && B.typespec().is_float()){
            impl = binary_op_noderivs<float,float,float, Log>;
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
        impl = binary_op_noderivs<Vec3,Vec3,float, Pow>;
    }
    else if (Result.typespec().is_float() && A.typespec().is_float() && B.typespec().is_float()){
        impl = binary_op_noderivs<float,float,float, Pow>;
    }
    else {
        std::cerr << "Don't know how compute " << Result.typespec().string()
                  << " = " << exec->op().opname() << "(" 
                  << A.typespec().string() << ", "
                  << B.typespec().string() << ")\n";
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
    unary_op_noderivs<float,float,Erf> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_erfc)
{
    unary_op_noderivs<float,float,Erfc> (exec, nargs, args, 
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
        impl = unary_op_noderivs<Vec3,Vec3, FAbs >;
    }
    else if (Result.typespec().is_float() && A.typespec().is_float()) {
        impl = unary_op_noderivs<float,float, FAbs >;
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
        std::cerr << "Don't know how compute " << Result.typespec().string()
                  << " = " << exec->op().opname() << "(" 
                  << A.typespec().string() << ")\n";
        ASSERT (0 && "Function arg type can't be handled");
    }
}


DECLOP (OP_floor)
{
    generic_unary_function_shadeop<Floor> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_ceil)
{
    generic_unary_function_shadeop<Ceil> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_round)
{
    generic_unary_function_shadeop<Round> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_trunc)
{
    generic_unary_function_shadeop<Trunc> (exec, nargs, args, 
                                         runflags, beginpoint, endpoint);
}

DECLOP (OP_sign)
{
    generic_unary_function_shadeop<Sign> (exec, nargs, args, 
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
        impl = ternary_op<Vec3,Vec3,Vec3,float, Mix>;
    }
    else if (Result.typespec().is_triple() && A.typespec().is_triple() && B.typespec().is_triple() && C.typespec().is_float()) {
        impl = ternary_op<Vec3,Vec3,Vec3,float, Mix>;
    }
    else if (Result.typespec().is_float() && A.typespec().is_float() && B.typespec().is_float() && C.typespec().is_float()){
        impl = ternary_op<float,float,float,float, Mix>;
    }
    else {
        std::cerr << "Don't know how compute " << Result.typespec().string()
                  << " = " << exec->op().opname() << "(" 
                  << A.typespec().string() << ", "
                  << B.typespec().string() << ", "
                  << C.typespec().string() << ")\n";
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
            impl = binary_op_noderivs<float,float,float, Hypot>;
        }
        else {
            std::cerr << "Don't know how compute " << Result.typespec().string()
                      << " = " << exec->op().opname() << "(" 
                      << A.typespec().string() << ", "
                      << B.typespec().string() << ")\n";
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
            std::cerr << "Don't know how compute " << Result.typespec().string()
                      << " = " << exec->op().opname() << "(" 
                      << A.typespec().string() << ", "
                      << B.typespec().string() << ", "
                      << C.typespec().string() << ")\n";
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
    ternary_op<Vec3,Vec3,Vec3,float, Refract> (exec, nargs, args, 
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
            std::cerr << "Don't know how compute "
                      << "void " << exec->op().opname() << "(" 
                      << I.typespec().string()   << ", "
                      << N.typespec().string()   << ", "
                      << eta.typespec().string() << ", "
                      << Kr.typespec().string()  << ")\n";
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
            std::cerr << "Don't know how compute "
                      << "void " << exec->op().opname() << "(" 
                      << I.typespec().string()   << ", "
                      << N.typespec().string()   << ", "
                      << eta.typespec().string() << ", "
                      << Kr.typespec().string()  << ", "
                      << Kt.typespec().string()  << ", "
                      << R.typespec().string()   << ", "
                      << T.typespec().string()   << ")\n";
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

}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
