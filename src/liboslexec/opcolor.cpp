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
/// Shader interpreter implementation of color operations.
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


extern DECLOP (triple_ctr);   // definition is elsewhere


namespace {


static Color3
hsv_to_rgb (float h, float s, float v)
{
    // Reference for this technique: Foley & van Dam
    if (s < 0.0001f) {
      return Color3 (v, v, v);
    } else {
        h = 6.0f * (h - floorf(h));  // expand to [0..6)
        int hi = (int) h;
        float f = h - hi;
        float p = v * (1.0f-s);
        float q = v * (1.0f-s*f);
        float t = v * (1.0f-s*(1.0f-f));
        switch (hi) {
        case 0 : return Color3 (v, t, p);
        case 1 : return Color3 (q, v, p);
        case 2 : return Color3 (p, v, t);
        case 3 : return Color3 (p, q, v);
        case 4 : return Color3 (t, p, v);
        default: return Color3 (v, p, q);
	}
    }
}



static Color3
hsl_to_rgb (float h, float s, float l)
{
    // Easiest to convert hsl -> hsv, then hsv -> RGB (per Foley & van Dam)
    float v = (l <= 0.5) ? (l * (1.0f + s)) : (l * (1.0f - s) + s);
    if (v <= 0.0f) {
        return Color3 (0.0f, 0.0f, 0.0f);
    } else {
	float min = 2.0f * l - v;
	s = (v - min) / v;
	return hsv_to_rgb (h, s, v);
    }
}



static Color3
YIQ_to_rgb (float Y, float I, float Q)
{
    return Color3 (Y + 0.9557f * I + 0.6199f * Q,
                   Y - 0.2716f * I - 0.6469f * Q,
                   Y - 1.1082f * I + 1.7051f * Q);
}



static Color3
xyz_to_rgb (float x, float y, float z)
{
    return Color3 ( 3.240479f * x + -1.537150f * y + -0.498535f * z,
                   -0.969256f * x +  1.875991f * y +  0.041556f * z,
                    0.055648f * x + -0.204043f * y +  1.057311f * z);
}



static Color3
to_rgb (ustring fromspace, float a, float b, float c, ShadingExecution *exec)
{
    if (fromspace == Strings::RGB || fromspace == Strings::rgb)
        return Color3 (a, b, c);
    if (fromspace == Strings::hsv)
        return hsv_to_rgb (a, b, c);
    if (fromspace == Strings::hsl)
        return hsl_to_rgb (a, b, c);
    if (fromspace == Strings::YIQ)
        return YIQ_to_rgb (a, b, c);
    if (fromspace == Strings::xyz)
        return xyz_to_rgb (a, b, c);

    exec->error ("Unknown color space \"%s\"", fromspace.c_str());
    return Color3 (a, b, c);
}



class Luminance {
public:
    Luminance (ShadingExecution *) { }
    void operator() (float &result, const Color3 &c) {
       result = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    }
};



/// Implementation of the constructor "color (string, float, float, float)".
///
DECLOP (color_ctr_transform)
{
    Symbol &Result (exec->sym (args[0]));
    Symbol &Space (exec->sym (args[1]));
    Symbol &X (exec->sym (args[2]));
    Symbol &Y (exec->sym (args[3]));
    Symbol &Z (exec->sym (args[4]));

    // Adjust the result's uniform/varying status
    bool vary = (Space.is_varying() |
                 X.is_varying() | Y.is_varying() | Z.is_varying() );
    exec->adjust_varying (Result, vary, false /* can't alias */);

    // FIXME -- clear derivs for now, make it right later.
    if (Result.has_derivs ())
        exec->zero_derivs (Result);

    VaryingRef<Color3> result ((Color3 *)Result.data(), Result.step());
    VaryingRef<ustring> space ((ustring *)Space.data(), Space.step());
    VaryingRef<float> x ((float *)X.data(), X.step());
    VaryingRef<float> y ((float *)Y.data(), Y.step());
    VaryingRef<float> z ((float *)Z.data(), Z.step());

    Matrix44 M;
    if (result.is_uniform()) {
        // Everything is uniform
        *result = to_rgb (*space, *x, *y, *z, exec);
    } else if (! vary) {
        // Result is varying, but everything else is uniform
        Color3 r = to_rgb (*space, *x, *y, *z, exec);
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = r;
    } else {
        // Fully varying case
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = to_rgb (space[i], x[i], y[i], z[i], exec);
    }
}


};  // End anonymous namespace


DECLOP (OP_color)
{
    DASSERT (nargs == 4 || nargs == 5);
    Symbol &Result (exec->sym (args[0]));
    bool using_space = (nargs == 5);
    Symbol &Space (exec->sym (args[1]));
    Symbol &X (exec->sym (args[1+using_space]));
    Symbol &Y (exec->sym (args[2+using_space]));
    Symbol &Z (exec->sym (args[3+using_space]));
    DASSERT (! Result.typespec().is_closure() && 
            ! X.typespec().is_closure() && ! Y.typespec().is_closure() &&
            ! Z.typespec().is_closure() && ! Space.typespec().is_closure());
    
    // We allow two flavors: point = point(float,float,float) and
    // point = point(string,float,float,float)
    if (Result.typespec().is_triple() && X.typespec().is_float() &&
          Y.typespec().is_float() && Z.typespec().is_float() &&
          (using_space == false || Space.typespec().is_string())) {
        OpImpl impl = NULL;
        if (using_space)
            impl = color_ctr_transform;  // special case: colors different
        else
            impl = triple_ctr;
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



DECLOP (OP_luminance)
{
    DASSERT (nargs == 2);
    Symbol &Result (exec->sym (args[0]));
    Symbol &C (exec->sym (args[1]));
    DASSERT (! Result.typespec().is_closure() && ! C.typespec().is_closure());
    DASSERT (Result.typespec().is_float() && C.typespec().is_triple());

    unary_op_guts_noderivs<Float, Color3, Luminance> (Result, C, exec, runflags,
                                                      beginpoint, endpoint);
}




}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
