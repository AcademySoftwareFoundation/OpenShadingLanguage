// Open Shading Language : Copyright (c) 2009-2017 Sony Pictures Imageworks Inc., et al.
// https://github.com/imageworks/OpenShadingLanguage/blob/master/LICENSE
// 
// MaterialX specification (c) 2017 Lucasfilm Ltd. 
// http://www.materialx.org/

#pragma once
#include "funcs.h"

struct vector4
{
    float x;
    float y;
    float z;
    float w;
};

vector4 abs(vector4 in)
{
    return vector4 (abs(in.x),
                    abs(in.y),
                    abs(in.z),
                    abs(in.w)
                    );
}

vector4 floor(vector4 in)
{
    return vector4 (floor(in.x),
                    floor(in.y),
                    floor(in.z),
                    floor(in.w)
                    );
}

vector4 mix(vector4 value1, vector4 value2, float x )
{
    return vector4 (mix( value1.x, value2.x, x),
                    mix( value1.y, value2.y, x),
                    mix( value1.z, value2.z, x),
                    mix( value1.w, value2.w, x)
                    );
}

vector vec4ToVec3(vector4 v)
{
    float s = 1/v.w;
    return vector(v.x * s,
                  v.y * s, 
                  v.z * s);
}

float dot(vector4 in1, vector4 in2)
{
    return (in1.x*in2.x + in1.y*in2.y + in1.z*in2.z + in1.w*in2.w);
}

vector4 add(vector4 in, vector4 amount)
{
    return vector4 (in.x + amount.x,
                    in.y + amount.y,
                    in.z + amount.z,
                    in.w + amount.w
                    );
}

vector4 add(vector4 in, float amount)
{
    return vector4 (in.x + amount,
                    in.y + amount,
                    in.z + amount,
                    in.w + amount
                    );
}

vector4 subtract(vector4 in, vector4 amount)
{
    return vector4 (in.x - amount.x,
                    in.y - amount.y,
                    in.z - amount.z,
                    in.w - amount.w
                    );
}

vector4 subtract(vector4 in, float amount)
{
    return vector4 (in.x - amount,
                    in.y - amount,
                    in.z - amount,
                    in.w - amount
                    );
}

vector4 smoothstep(vector4 low, vector4 high, vector4 in)
{
    return vector4 (smoothstep(low.x, high.x, in.x),
                    smoothstep(low.y, high.y, in.y),
                    smoothstep(low.z, high.z, in.z),
                    smoothstep(low.w, high.w, in.w)
                    );
}

vector4 smoothstep(float low, float high, vector4 in)
{
    return vector4 (smoothstep(low, high, in.x),
                    smoothstep(low, high, in.y),
                    smoothstep(low, high, in.z),
                    smoothstep(low, high, in.w)
                    );
}

vector4 remap(vector4 in, vector4 inLow, vector4 inHigh, vector4 outLow, vector4 outHigh, int doClamp)
{
    //remap in from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range

    return vector4 (remap(in.x, inLow.x, inHigh.x, outLow.x, outHigh.x, doClamp),
                    remap(in.y, inLow.y, inHigh.y, outLow.y, outHigh.y, doClamp),
                    remap(in.z, inLow.z, inHigh.z, outLow.z, outHigh.z, doClamp),
                    remap(in.w, inLow.w, inHigh.w, outLow.w, outHigh.w, doClamp)
                    );
}

vector4 remap(vector4 in, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
    //remap in from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range

    return vector4 (remap(in.x, inLow, inHigh, outLow, outHigh, doClamp),
                    remap(in.y, inLow, inHigh, outLow, outHigh, doClamp),
                    remap(in.z, inLow, inHigh, outLow, outHigh, doClamp),
                    remap(in.w, inLow, inHigh, outLow, outHigh, doClamp)
                    );
}

vector4 fgamma(vector4 in, vector4 g)
{
    return vector4 (fgamma(in.x, g.x),
                    fgamma(in.y, g.y),
                    fgamma(in.z, g.z),
                    fgamma(in.w, g.w)
                    );
}

vector4 fgamma(vector4 in, float g)
{
    return vector4 (fgamma(in.x, g),
                    fgamma(in.y, g),
                    fgamma(in.z, g),
                    fgamma(in.w, g)
                    );
}

vector4 invert(vector4 in, vector4 amount)
{
    return vector4 (amount.x - in.x,
                    amount.y - in.y,
                    amount.z - in.z,
                    amount.w - in.w
                    );
}

vector4 invert(vector4 in, float amount)
{
    return vector4 (amount - in.x,
                    amount - in.y,
                    amount - in.z,
                    amount - in.w
                    );
}

vector4 invert(vector4 in)
{
    return invert(in, 1.0);
}

vector4 clamp(vector4 in, vector4 low, vector4 high)
{
    return vector4 (clamp(in.x, low.x, high.x),
                    clamp(in.y, low.y, high.y),
                    clamp(in.z, low.z, high.z),
                    clamp(in.w, low.w, high.w)
                    );
}

vector4 clamp(vector4 in, float low, float high)
{
    return vector4 (clamp(in.x, low, high),
                    clamp(in.y, low, high),
                    clamp(in.z, low, high),
                    clamp(in.w, low, high)
                    );
}

vector4 contrast(vector4 in, vector4 amount, vector4 pivot)
{
    return vector4 (contrast(in.x, amount.x, pivot.x),
                    contrast(in.y, amount.y, pivot.y),
                    contrast(in.z, amount.z, pivot.z),
                    contrast(in.w, amount.w, pivot.w)
                    );
}

vector4 contrast(vector4 in, float amount, float pivot)
{
    return vector4 (contrast(in.x, amount, pivot),
                    contrast(in.y, amount, pivot),
                    contrast(in.z, amount, pivot),
                    contrast(in.w, amount, pivot)
                    );
}

vector4 divide(vector4 in1, vector4 amount)
{
    return vector4 (in1.x / amount.x,
                    in1.y / amount.y,
                    in1.z / amount.z,
                    in1.w / amount.w
                    );
}

vector4 divide(vector4 in1, float amount)
{
    return vector4 (in1.x / amount,
                    in1.y / amount,
                    in1.z / amount,
                    in1.w / amount
                    );
}

vector4 exponent(vector4 in, vector4 amount)
{
    return vector4 (exponent(in.x, amount.x),
                    exponent(in.y, amount.y),
                    exponent(in.z, amount.z),
                    exponent(in.w, amount.w)
                    );
}

vector4 exponent(vector4 in, float amount)
{
    return vector4 (exponent(in.x, amount),
                    exponent(in.y, amount),
                    exponent(in.z, amount),
                    exponent(in.w, amount)
                    );
}


vector4 max(vector4 in, vector4 amount)
{
    return vector4 (max(in.x, amount.x),
                    max(in.y, amount.y),
                    max(in.z, amount.z),
                    max(in.w, amount.w)
                    );
}

vector4 max(vector4 in, float amount)
{
    return vector4 (max(in.x, amount),
                    max(in.y, amount),
                    max(in.z, amount),
                    max(in.w, amount)
                    );
}

vector4 normalize(vector4 in)
{
    vector v = normalize(vec4ToVec3(in));
    return vector4 (v[0], v[1], v[2], 1.0 );
}

vector4 multiply(vector4 in, vector4 amount)
{
    return vector4 (in.x * amount.x,
                    in.y * amount.y,
                    in.z * amount.z,
                    in.w * amount.w
                    );
}

vector4 multiply(vector4 in, float amount)
{
    return vector4 (in.x * amount,
                    in.y * amount,
                    in.z * amount,
                    in.w * amount
                    );
}

vector4 min(vector4 in, vector4 amount)
{
    return vector4 (min(in.x, amount.x),
                    min(in.y, amount.y),
                    min(in.z, amount.z),
                    min(in.w, amount.w)
                    );
}

vector4 min(vector4 in, float amount)
{
    return vector4 (min(in.x, amount),
                    min(in.y, amount),
                    min(in.z, amount),
                    min(in.w, amount)
                    );
}

vector4 fmod(vector4 in, vector4 amount)
{
    return vector4 (fmod(in.x, amount.x),
                    fmod(in.y, amount.y),
                    fmod(in.z, amount.z),
                    fmod(in.w, amount.w)
                    );
}

vector4 fmod(vector4 in, float amount)
{
    return vector4 (fmod(in.x, amount),
                    fmod(in.y, amount),
                    fmod(in.z, amount),
                    fmod(in.w, amount)
                    );
}

float mag(vector4 in)
{
    return length(vector(in.x/in.w, in.y/in.w, in.z/in.w));
}

vector4 difference(vector4 fg, vector4 bg)
{
    return vector4 (difference(fg.x, bg.x),
                    difference(fg.y, bg.y),
                    difference(fg.z, bg.z),
                    difference(fg.w, bg.w)
                    );
}

