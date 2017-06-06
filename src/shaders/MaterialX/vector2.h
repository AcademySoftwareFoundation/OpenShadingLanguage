// Open Shading Language : Copyright (c) 2009-2017 Sony Pictures Imageworks Inc., et al.
// https://github.com/imageworks/OpenShadingLanguage/blob/master/LICENSE
// 
// MaterialX specification (c) 2017 Lucasfilm Ltd. 
// http://www.materialx.org/

#pragma once
#include "funcs.h"

struct vector2
{
    float x;
    float y;
};

vector2 __operator__neg__(vector2 a)
{
    return vector2(-a.x, -a.y);
}

vector2 __operator__add__(vector2 a, vector2 b)
{
    return vector2(a.x + b.x, a.y + b.y);
}

vector2 __operator__add__(vector2 a, int b)
{
    return a + vector2(b, b);
}

vector2 __operator__add__(vector2 a, float b)
{
    return a + vector2(b, b);
}

vector2 __operator__sub__(vector2 a, vector2 b)
{
    return vector2(a.x - b.x, a.y - b.y);
}

vector2 __operator__sub__(vector2 a, int b)
{
    return a - vector2(b, b);
}

vector2 __operator__sub__(vector2 a, float b)
{
    return a - vector2(b, b);
}

vector2 __operator__mul__(vector2 a, vector2 b)
{
    return vector2(a.x * b.x, a.y * b.y);
}

vector2 __operator__mul__(vector2 a, int b)
{
    return a * vector2(b, b);
}

vector2 __operator__mul__(vector2 a, float b)
{
    return a * vector2(b, b);
}

vector2 __operator__div__(vector2 a, vector2 b)
{
    return vector2(a.x / b.x, a.y / b.y);
}

vector2 __operator__div__(vector2 a, int b)
{
    float b_inv = 1/b;
    return a * vector2(b_inv, b_inv);
}

vector2 __operator__div__(vector2 a, float b)
{
    float b_inv = 1/b;
    return a * vector2(b_inv, b_inv);
}

int __operator__eq__(vector2 a, vector2 b)
{
    return (a.x == b.x) && (a.y == b.y);
}

int __operator__ne__(vector2 a, vector2 b)
{
    return (a.x != b.x) || (a.y != b.y);
}


vector2 abs(vector2 a)
{
    return vector2 (abs(a.x),
                    abs(a.y)
                    );
}

vector2 floor(vector2 a)
{
    return vector2 (floor(a.x),
                    floor(a.y)
                    );
}

vector2 mix(vector2 a, vector2 b, float x )
{
    return vector2 (mix(a.x, b.x, x),
                    mix(a.y, b.y, x)
                    );
}

float dot(vector2 a, vector2 b)
{
    return (a.x * b.x + a.y * b.y);
}

vector2 smoothstep(vector2 low, vector2 high, vector2 in)
{
    return vector2 (smoothstep(low.x, high.x, in.x),
                    smoothstep(low.y, high.y, in.y)
                    );
}

vector2 smoothstep(float low, float high, vector2 in)
{
    return vector2 (smoothstep(low, high, in.x),
                    smoothstep(low, high, in.y)
                    );
}

vector2 remap(vector2 in, vector2 inLow, vector2 inHigh, vector2 outLow, vector2 outHigh, int doClamp)
{
    //remap in from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range

    return vector2 (remap(in.x, inLow.x, inHigh.x, outLow.x, outHigh.x, doClamp),
                    remap(in.y, inLow.y, inHigh.y, outLow.y, outHigh.y, doClamp)
                    );
}

vector2 remap(vector2 in, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
    //remap in from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range

    return vector2 (remap(in.x, inLow, inHigh, outLow, outHigh, doClamp),
                    remap(in.y, inLow, inHigh, outLow, outHigh, doClamp)
                    );
}

vector2 fgamma(vector2 in, vector2 g)
{
    return vector2 (fgamma(in.x, g.x),
                    fgamma(in.y, g.y)
                    );
}

vector2 fgamma(vector2 in, float g)
{
    return vector2 (fgamma(in.x, g),
                    fgamma(in.y, g)
                    );
}

vector2 invert(vector2 in, vector2 amount)
{
    return vector2 (amount.x - in.x,
                    amount.y - in.y
                    );
}

vector2 invert(vector2 in, float amount)
{
    return vector2 (amount - in.x,
                    amount - in.y
                    );
}

vector2 invert(vector2 in)
{
    return invert(in, 1.0);
    
}

vector2 clamp(vector2 in, vector2 low, vector2 high)
{
    return vector2 (clamp(in.x, low.x, high.x),
                    clamp(in.y, low.y, high.y)
                    );
}

vector2 clamp(vector2 in, float low, float high)
{
    return vector2 (clamp(in.x, low, high),
                    clamp(in.y, low, high)
                    );
}

vector2 contrast(vector2 in, vector2 amount, vector2 pivot)
{
    return vector2 (contrast(in.x, amount.x, pivot.x),
                    contrast(in.y, amount.y, pivot.y)
                    );
}

vector2 contrast(vector2 in, float amount, float pivot)
{
    return vector2 (contrast(in.x, amount, pivot),
                    contrast(in.y, amount, pivot)
                    );
}

vector2 exponent(vector2 in, vector2 amount)
{
    return vector2 (exponent(in.x, amount.x),
                    exponent(in.y, amount.y)
                    );
}

vector2 exponent(vector2 in, float amount)
{
    return vector2 (exponent(in.x, amount),
                    exponent(in.y, amount)
                    );
}

vector2 max(vector2 in, vector2 amount)
{
    return vector2 (max(in.x, amount.x),
                    max(in.y, amount.y)
                    );
}

vector2 max(vector2 in, float amount)
{
    return vector2 (max(in.x, amount),
                    max(in.y, amount)
                    );
}

vector2 normalize(vector2 in)
{
    vector v = normalize(vector(in.x, in.y, 0));
    return vector2 (v[0], v[1]);
}

vector2 vscale(vector2 in, vector2 amount, vector2 center)
{
    return vector2 ((in.x - center.x)/amount.x + center.x,
                    (in.y - center.y)/amount.y + center.y
                    );
}


vector2 min(vector2 in, vector2 amount)
{
    return vector2 (min(in.x, amount.x),
                    min(in.y, amount.y)
                    );
}

vector2 min(vector2 in, float amount)
{
    return vector2 (min(in.x, amount),
                    min(in.y, amount)
                    );
}

vector2 fmod(vector2 in, vector2 amount)
{
    return vector2 (fmod(in.x, amount.x),
                    fmod(in.y, amount.y)
                    );
}

vector2 fmod(vector2 in, float amount)
{
    return vector2 (fmod(in.x, amount),
                    fmod(in.y, amount)
                    );
}

float mag(vector2 in){
    return length(vector(in.x, in.y, 0));
}

vector2 difference(vector2 fg, vector2 bg)
{
    return vector2 (difference(fg.x, bg.x),
                    difference(fg.y, bg.y)
                    );
}

vector2 rotate2d(vector2 in, float amount, vector2 center)
{
    vector2 out;
    out.x = out.x - center.x;
    out.y = out.y - center.y;
    out.x = in.x*cos(amount) - in.y*sin(amount);
    out.y = in.y*cos(amount) + in.x*sin(amount);
    out.x = out.x+center.x;
    out.y = out.y+center.y;
    return out;
}

