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
/// Shader interpreter implementation of color operations.
///
/////////////////////////////////////////////////////////////////////////


#include <iostream>

#include "oslexec_pvt.h"
#include "oslops.h"
#include "dual.h"

#include "OpenImageIO/varyingref.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {

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

};  // End anonymous namespace



Color3
ShadingSystemImpl::to_rgb (ustring fromspace, float a, float b, float c)
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
    error ("Unknown color space \"%s\"", fromspace.c_str());
    return Color3 (a, b, c);
}

}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
