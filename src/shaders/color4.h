// Open Shading Language : Copyright (c) 2009-2017 Sony Pictures Imageworks Inc., et al.
// https://github.com/imageworks/OpenShadingLanguage/blob/master/LICENSE

#pragma once
#define COLOR4_H


// color4 is a color + alpha
struct color4
{
    color rgb;
    float a;
};



//
// For color4, define math operators to match color
//

color4 __operator__neg__(color4 a)
{
    return color4(-a.rgb, -a.a);
}

color4 __operator__add__(color4 a, color4 b)
{
    return color4(a.rgb + b.rgb, a.a + b.a);
}

color4 __operator__add__(color4 a, int b)
{
    return a + color4(color(b), b);
}

color4 __operator__add__(color4 a, float b)
{
    return a + color4(color(b), b);
}

color4 __operator__add__(int a, color4 b)
{
    return color4(color(a), a) + b;
}

color4 __operator__add__(float a, color4 b)
{
    return color4(color(a), a) + b;
}

color4 __operator__sub__(color4 a, color4 b)
{
    return color4(a.rgb - b.rgb, a.a - b.a);
}

color4 __operator__sub__(color4 a, int b)
{
    return a - color4(color(b), b);
}

color4 __operator__sub__(color4 a, float b)
{
    return a - color4(color(b), b);
}

color4 __operator__sub__(int a, color4 b)
{
    return color4(color(a), a) - b;
}

color4 __operator__sub__(float a, color4 b)
{
    return color4(color(a), a) - b;
}

color4 __operator__mul__(color4 a, color4 b)
{
    return color4(a.rgb * b.rgb, a.a * b.a);
}

color4 __operator__mul__(color4 a, int b)
{
    return a * color4(color(b), b);
}

color4 __operator__mul__(color4 a, float b)
{
    return a * color4(color(b), b);
}

color4 __operator__mul__(int a, color4 b)
{
    return color4(color(a), a) * b;
}

color4 __operator__mul__(float a, color4 b)
{
    return color4(color(a), a) * b;
}

color4 __operator__div__(color4 a, color4 b)
{
    return color4(a.rgb / b.rgb, a.a / b.a);
}

color4 __operator__div__(color4 a, int b)
{
    float b_inv = 1/b;
    return a * color4(color(b_inv), b_inv);
}

color4 __operator__div__(color4 a, float b)
{
    float b_inv = 1/b;
    return a * color4(color(b_inv), b_inv);
}

color4 __operator_div__(int a, color4 b)
{
    return color4(color(a), a) / b;
}

color4 __operator__div__(float a, color4 b)
{
    return color4(color(a), a) / b;
}

int __operator__eq__(color4 a, color4 b)
{
    return (a.rgb == b.rgb) && (a.a == b.a);
}

int __operator__ne__(color4 a, color4 b)
{
    return (a.rgb != b.rgb) || (a.a != b.a);
}



//
// For color4, define most of the stdosl functions to match color
//

color4 abs(color4 a)
{
    return color4(abs(a.rgb), abs(a.a));
}

color4 floor(color4 a)
{
    return color4(floor(a.rgb), floor(a.a));
}

color4 mix(color4 a, color4 b, float x )
{
    return color4(mix(a.rgb, b.rgb, x),
                  mix(a.a, b.a, x));
}

float dot(color4 a, color b)
{
    return dot(a.rgb, b);
}

color4 smoothstep(color4 edge0, color4 edge1, color4 c)
{
    return color4(smoothstep(edge0.rgb, edge1.rgb, c.rgb),
                  smoothstep(edge0.a, edge1.a, c.a));
}

color4 smoothstep(float edge0, float edge1, color4 c)
{
    return smoothstep(color4(color(edge0), edge0), color4(color(edge1), edge1), c);
}

color4 clamp(color4 c, color4 minval, color4 maxval)
{
    return color4(clamp(c.rgb, minval.rgb, maxval.rgb),
                  clamp(c.a, minval.a, maxval.a));
}

color4 clamp(color4 c, float minval, float maxval)
{
    return clamp(c, color4(color(minval), minval), color4(color(maxval), maxval));
}

color4 max(color4 a, color4 b)
{
    return color4(max(a.rgb, b.rgb),
                  max(a.a, b.a));
}

color4 max(color4 a, float b)
{
    return color4(max(a.rgb, b),
                  max(a.a, b));
}

color4 min(color4 a, color4 b)
{
    return color4(min(a.rgb, b.rgb),
                  min(a.a, b.a));
}

color4 min(color4 a, float b)
{
    return color4(min(a.rgb, b),
                  min(a.a, b));
}

color4 fmod(color4 a, color4 b)
{
    return color4(fmod(a.rgb, b.rgb),
                  fmod(a.a, b.a));
}

color4 fmod(color4 a, int b)
{
    return fmod(a, color4(color(b), b));
}

color4 fmod(color4 a, float b)
{
    return fmod(a, color4(color(b), b));
}

color4 pow(color4 base, color4 power)
{
    return color4(pow(base.rgb, power.rgb), pow(base.a, power.a));
}

color4 pow(color4 base, float power)
{
    return pow(base, color4(color(power), power));
}
