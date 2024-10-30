// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once
#define COLOR2_H


// color2 is a single color channel + alpha
struct color2
{
    float r;
    float a;
};


//
// For color2, define math operators to match color
//

color2 __operator__neg__(color2 a)
{
    return color2(-a.r, -a.a);
}

color2 __operator__add__(color2 a, color2 b)
{
    return color2(a.r + b.r, a.a + b.a);
}

color2 __operator__add__(color2 a, int b)
{
    return a + color2(b, b);
}

color2 __operator__add__(color2 a, float b)
{
    return a + color2(b, b);
}

color2 __operator__add__(int a, color2 b)
{
    return color2(a, a) + b;
}

color2 __operator__add__(float a, color2 b)
{
    return color2(a, a) + b;
}

color2 __operator__sub__(color2 a, color2 b)
{
    return color2(a.r - b.r, a.a - b.a);
}

color2 __operator__sub__(color2 a, int b)
{
    return a - color2(b, b);
}

color2 __operator__sub__(color2 a, float b)
{
    return a - color2(b, b);
}

color2 __operator__sub__(int a, color2 b)
{
    return color2(a,a) - b;
}

color2 __operator__sub__(float a, color2 b)
{
    return color2(a,a) - b;
}

color2 __operator__mul__(color2 a, color2 b)
{
    return color2(a.r * b.r, a.a * b.a);
}

color2 __operator__mul__(color2 a, int b)
{
    return a * color2(b, b);
}

color2 __operator__mul__(color2 a, float b)
{
    return a * color2(b, b);
}

color2 __operator__mul__(int a, color2 b)
{
    return b * color2(a, a);
}

color2 __operator__mul__(float a, color2 b)
{
    return b * color2(a, a);
}

color2 __operator__div__(color2 a, color2 b)
{
    return color2(a.r / b.r, a.a / b.a);
}

color2 __operator__div__(color2 a, int b)
{
    float b_inv = 1.0 / float(b);
    return a * color2(b_inv, b_inv);
}

color2 __operator__div__(color2 a, float b)
{
    float b_inv = 1.0 / b;
    return a * color2(b_inv, b_inv);
}

color2 __operator__div__(int a, color2 b)
{
    return color2(a, a) / b;
}

color2 __operator__div__(float a, color2 b)
{
    return color2(a, a) / b;
}

int __operator__eq__(color2 a, color2 b)
{
    return (a.r == b.r) && (a.a == b.a);
}

int __operator__neq__(color2 a, color2 b)
{
    return (a.r != b.r) || (a.a != b.a);
}



//
// For color2, define most of the stdosl functions to match color
//

color2 abs(color2 a)
{
    return color2(abs(a.r), abs(a.a));
}

color2 ceil(color2 a)
{
    return color2(ceil(a.r), ceil(a.a));
}

color2 floor(color2 a)
{
    return color2(floor(a.r), floor(a.a));
}

color2 sqrt(color2 a)
{
    return color2(sqrt(a.r), sqrt(a.a));
}

color2 exp(color2 a)
{
    return color2(exp(a.r), exp(a.a));
}

color2 log(color2 a)
{
    return color2(log(a.r), log(a.a));
}

color2 log2(color2 a)
{
    return color2(log2(a.r), log2(a.a));
}

color2 mix(color2 a, color2 b, float x )
{
    return color2(mix(a.r, b.r, x),
                  mix(a.a, b.a, x));
}

color2 smoothstep(color2 edge0, color2 edge1, color2 c)
{
    return color2(smoothstep(edge0.r, edge1.r, c.r),
                  smoothstep(edge0.a, edge1.a, c.a));
}

color2 smoothstep(float edge0, float edge1, color2 c)
{
    return smoothstep(color2(edge0, edge0), color2(edge1, edge1), c);
}

color2 clamp(color2 c, color2 minval, color2 maxval)
{
    return color2(clamp(c.r, minval.r, maxval.r),
                  clamp(c.a, minval.a, maxval.a));
}

color2 clamp(color2 c, float minval, float maxval)
{
    return clamp(c, color2(minval, minval), color2(maxval, maxval));
}

color2 max(color2 a, color2 b)
{
    return color2(max(a.r, b.r),
                  max(a.a, b.a));
}

color2 max(color2 a, float b)
{
    return color2(max(a.r, b),
                  max(a.a, b));
}

color2 min(color2 a, color2 b)
{
    return color2(min(a.r, b.r),
                  min(a.a, b.a));
}

color2 min(color2 a, float b)
{
    return color2(min(a.r, b),
                  min(a.a, b));
}

color2 fmod(color2 a, color2 b)
{
    return color2(fmod(a.r, b.r),
                  fmod(a.a, b.a));
}

color2 fmod(color2 a, int b)
{
    return fmod(a, color2(b, b));
}

color2 fmod(color2 a, float b)
{
    return fmod(a, color2(b, b));
}

color2 pow(color2 base, color2 power)
{
    return color2(pow(base.r, power.r), pow(base.a, power.a));
}

color2 pow(color2 base, float power)
{
    return pow(base, color2(power, power));
}

color2 sign(color2 a)
{
    return color2(sign(a.r),
                  sign(a.a));
}

color2 sin(color2 a)
{
    return color2(sin(a.r),
                  sin(a.a));
}

color2 cos(color2 a)
{
    return color2(cos(a.r),
                  cos(a.a));
}

color2 tan(color2 a)
{
    return color2(tan(a.r),
                  tan(a.a));
}

color2 asin(color2 a)
{
    return color2(asin(a.r),
                  asin(a.a));
}

color2 acos(color2 a)
{
    return color2(acos(a.r),
                  acos(a.a));
}

color2 atan2(color2 a, float f)
{
    return color2(atan2(a.r, f),
                  atan2(a.a, f));
}


color2 atan2(color2 a, color2 b)
{
    return color2(atan2(a.r, b.r),
                  atan2(a.a, b.a));
}
