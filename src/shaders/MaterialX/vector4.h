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

vector4 make_vector4(float x, float y, float z, float w)
{
    vector4 out;
    out.x = x;
    out.y = y;
    out.z = z;
    out.w = w;
    return out;
}

vector4 make_vector4(float x)
{
    vector4 out;
    out.x = x;
    out.y = x;
    out.z = x;
    out.w = x;
    return out;
}

vector4 abs(vector4 in)
{
    vector4 out;
    out.x = abs(in.x);
    out.y = abs(in.y);
    out.z = abs(in.z);
    out.w = abs(in.w);
    return out;
}

vector4 floor(vector4 in)
{
    vector4 out;
    out.x = floor(in.x);
    out.y = floor(in.y);
    out.z = floor(in.z);
    out.w = floor(in.w);
    return out;
}

vector4 mix(vector4 value1, vector4 value2, float x )
{
    vector4 out;
    
    out.x = mix( value1.x, value2.x, x);
    out.y = mix( value1.y, value2.y, x);
    out.z = mix( value1.z, value2.z, x);
    out.w = mix( value1.w, value2.w, x);

    return out;
}

vector vec4ToVec3(vector4 v)
{
    float s = 1/v.w;
    return vector(v.x * s,
                  v.y * s, 
                  v.z * s);
}

float dot(vector4 a, vector4 b)
{
    return (a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w);
}

vector4 add(vector4 in, vector4 amount)
{
    vector4 out = in;
    out.x = in.x + amount.x;
    out.y = in.y + amount.y;
    out.z = in.z + amount.z;
    out.w = in.w + amount.w;
    return out;
}

vector4 add(vector4 in, float amount)
{
    vector4 out = in;
    out.x = in.x + amount;
    out.y = in.y + amount;
    out.z = in.z + amount;
    out.w = in.w + amount;
    return out;
}

vector4 subtract(vector4 in, vector4 amount)
{
    vector4 out = in;
    out.x = in.x - amount.x;
    out.y = in.y - amount.y;
    out.z = in.z - amount.z;
    out.w = in.w - amount.w;
    return out;
}

vector4 subtract(vector4 in, float amount)
{
    vector4 out = in;
    out.x = in.x - amount;
    out.y = in.y - amount;
    out.z = in.z - amount;
    out.w = in.w - amount;
    return out;
}

vector4 smoothstep(vector4 low, vector4 high, vector4 in)
{
    vector4 out = in;
    out.x = smoothstep(low.x, high.x, in.x);
    out.y = smoothstep(low.y, high.y, in.y);
    out.z = smoothstep(low.z, high.z, in.z);
    out.w = smoothstep(low.w, high.w, in.w);
    return out;
}

vector4 smoothstep(float low, float high, vector4 in)
{
    vector4 out = in;
    out.x = smoothstep(low, high, in.x);
    out.y = smoothstep(low, high, in.y);
    out.z = smoothstep(low, high, in.z);
    out.w = smoothstep(low, high, in.w);
    return out;
}

vector4 remap(vector4 in, vector4 inLow, vector4 inHigh, vector4 outLow, vector4 outHigh, int doClamp)
{
    //remap in from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range

    vector4 out = in;
    out.x = remap(in.x, inLow.x, inHigh.x, outLow.x, outHigh.x, doClamp);
    out.y = remap(in.y, inLow.y, inHigh.y, outLow.y, outHigh.y, doClamp);
    out.z = remap(in.z, inLow.z, inHigh.z, outLow.z, outHigh.z, doClamp);
    out.w = remap(in.w, inLow.w, inHigh.w, outLow.w, outHigh.w, doClamp);
    return out;
}

vector4 remap(vector4 in, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
    //remap in from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range

    vector4 out = in;
    out.x = remap(in.x, inLow, inHigh, outLow, outHigh, doClamp);
    out.y = remap(in.y, inLow, inHigh, outLow, outHigh, doClamp);
    out.z = remap(in.z, inLow, inHigh, outLow, outHigh, doClamp);
    out.w = remap(in.w, inLow, inHigh, outLow, outHigh, doClamp);

    return out;
}

vector4 fgamma(vector4 in, vector4 g)
{
    vector4 out = in;

    out.x = fgamma(in.x, g.x);
    out.y = fgamma(in.y, g.y);
    out.z = fgamma(in.z, g.z);
    out.w = fgamma(in.w, g.w);
      return out;
}

vector4 fgamma(vector4 in, float g)
{
    vector4 out = in;

    out.x = fgamma(in.x, g);
    out.y = fgamma(in.y, g);
    out.z = fgamma(in.z, g);
    out.w = fgamma(in.w, g);
      return out;
}

vector4 invert(vector4 in, vector4 amount)
{
    vector4 out = in;
    out.x = amount.x - in.x;
    out.y = amount.y - in.y;
    out.z = amount.z - in.z;
    out.w = amount.w - in.w;

    return out;
}

vector4 invert(vector4 in, float amount)
{
    vector4 out = in;
    out.x = amount - in.x;
    out.y = amount - in.y;
    out.z = amount - in.z;
    out.w = amount - in.w;

    return out;
}

vector4 invert(vector4 in)
{
    vector4 out = invert(in, 1.0);
    return out;
}

vector4 clamp(vector4 in, vector4 low, vector4 high)
{
    vector4 out = in;
    out.x = clamp(in.x, low.x, high.x);
    out.y = clamp(in.y, low.y, high.y);
    out.z = clamp(in.z, low.z, high.z);
    out.w = clamp(in.w, low.w, high.w);
    return out;
}

vector4 clamp(vector4 in, float low, float high)
{
    vector4 out = in;
    out.x = clamp(in.x, low, high);
    out.y = clamp(in.y, low, high);
    out.z = clamp(in.z, low, high);
    out.w = clamp(in.w, low, high);
    return out;
}

vector4 contrast(vector4 in, vector4 amount, vector4 pivot)
{
    vector4 out = in;
    out.x = contrast(in.x, amount.x, pivot.x);
    out.y = contrast(in.y, amount.y, pivot.y);
    out.z = contrast(in.z, amount.z, pivot.z);
    out.w = contrast(in.w, amount.w, pivot.w);
    return out;
}

vector4 contrast(vector4 in, float amount, float pivot)
{
    vector4 out = in;
    out.x = contrast(in.x, amount, pivot);
    out.y = contrast(in.y, amount, pivot);
    out.z = contrast(in.z, amount, pivot);
    out.w = contrast(in.w, amount, pivot);
    return out;
}

vector4 divide(vector4 in1, vector4 amount)
{
    vector4 out = in1;
    out.x = in1.x / amount.x;
    out.y = in1.y / amount.y;
    out.z = in1.z / amount.z;
    out.w = in1.w / amount.w;
    return out;
}

vector4 divide(vector4 in1, float amount)
{
    vector4 out = in1;
    out.x = in1.x / amount;
    out.y = in1.y / amount;
    out.z = in1.z / amount;
    out.w = in1.w / amount;
    return out;
}

vector4 exponent(vector4 in, vector4 amount)
{
    vector4 out = in;
    out.x = exponent(in.x, amount.x);
    out.y = exponent(in.y, amount.y);
    out.z = exponent(in.z, amount.z);
    out.w = exponent(in.w, amount.w);
    return out;
}

vector4 exponent(vector4 in, float amount)
{
    vector4 out = in;
    out.x = exponent(in.x, amount);
    out.y = exponent(in.y, amount);
    out.z = exponent(in.z, amount);
    out.w = exponent(in.w, amount);
    return out;
}


vector4 max(vector4 in, vector4 amount)
{
    vector4 out = in;
    out.x = max(in.x, amount.x);
    out.y = max(in.y, amount.y);
    out.z = max(in.z, amount.z);
    out.w = max(in.w, amount.w);
    return out;
}

vector4 max(vector4 in, float amount)
{
    vector4 out = in;
    out.x = max(in.x, amount);
    out.y = max(in.y, amount);
    out.z = max(in.z, amount);
    out.w = max(in.w, amount);
    return out;
}

vector4 normalize(vector4 in)
{
    vector v = normalize(vec4ToVec3(in));
    vector4 out;
    out.x = v[0];
    out.y = v[1];
    out.z = v[2];
    out.w = 1.0;
    return out;
}

vector4 multiply(vector4 in1, vector4 amount)
{
    vector4 out = in1;
    out.x = in1.x * amount.x;
    out.y = in1.y * amount.y;
    out.z = in1.z * amount.z;
    out.w = in1.w * amount.w;
    return out;
}

vector4 multiply(vector4 in1, float amount)
{
    vector4 out = in1;
    out.x = in1.x * amount;
    out.y = in1.y * amount;
    out.z = in1.z * amount;
    out.w = in1.w * amount;
    return out;
}

vector4 min(vector4 in, vector4 amount)
{
    vector4 out = in;
    out.x = min(in.x, amount.x);
    out.y = min(in.y, amount.y);
    out.z = min(in.z, amount.z);
    out.w = min(in.w, amount.w);

    return out;
}

vector4 min(vector4 in, float amount)
{
    vector4 out = in;
    out.x = min(in.x, amount);
    out.y = min(in.y, amount);
    out.z = min(in.z, amount);
    out.w = min(in.w, amount);

    return out;
}

vector4 fmod(vector4 in, vector4 amount)
{
    vector4 out = in;
    out.x = fmod(in.x, amount.x);
    out.y = fmod(in.y, amount.y);
    out.z = fmod(in.z, amount.z);
    out.w = fmod(in.w, amount.w);
    return out;
}

vector4 fmod(vector4 in, float amount)
{
    vector4 out = in;
    out.x = fmod(in.x, amount);
    out.y = fmod(in.y, amount);
    out.z = fmod(in.z, amount);
    out.w = fmod(in.w, amount);
    return out;
}

float mag(vector4 in)
{
    return length(vector(in.x/in.w, in.y/in.w, in.z/in.w));
}

vector4 difference(vector4 fg, vector4 bg){
    vector4 out = { difference(fg.x, bg.x),
                 difference(fg.y, bg.y),
                 difference(fg.z, bg.z),
                 difference(fg.w, bg.w)
               };

    return out;
}

