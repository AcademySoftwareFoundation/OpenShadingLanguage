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

vector2 make_vector2(float x, float y)
{
    vector2 out;
    out.x = x;
    out.y = y;
    return out;
}

vector2 abs(vector2 in)
{
    vector2 out;
    out.x = abs(in.x);
    out.y = abs(in.y);
    return out;
}

vector2 floor(vector2 in)
{
    vector2 out;
    out.x = floor(in.x);
    out.y = floor(in.y);
    return out;
}

vector2 mix(vector2 value1, vector2 value2, float x )
{
    vector2 out;
    
    out.x = mix(value1.x, value2.x, x);
    out.y = mix(value1.y, value2.y, x);
    return out;
}

float dot(vector2 a, vector2 b)
{
    return (a.x*b.x + a.y*b.y);
}

vector2 add(vector2 in, vector2 amount)
{
    vector2 out = in;
    out.x = in.x + amount.x;
    out.y = in.y + amount.y;
    return out;
}

vector2 add(vector2 in, float amount)
{
    vector2 out = in;
    out.x = in.x + amount;
    out.y = in.y + amount;
    return out;
}

vector2 subtract(vector2 in, vector2 amount)
{
    vector2 out = in;
    out.x = in.x - amount.x;
    out.y = in.y - amount.y;
    return out;
}

vector2 subtract(vector2 in, float amount)
{
    vector2 out = in;
    out.x = in.x - amount;
    out.y = in.y - amount;
    return out;
}

vector2 smoothstep(vector2 low, vector2 high, vector2 in)
{
    vector2 out = in;
    out.x = smoothstep(low.x, high.x, in.x);
    out.y = smoothstep(low.y, high.y, in.y);
    return out;
}

vector2 smoothstep(float low, float high, vector2 in)
{
    vector2 out = in;
    out.x = smoothstep(low, high, in.x);
    out.y = smoothstep(low, high, in.y);

    return out;
}

vector2 remap(vector2 in, vector2 inLow, vector2 inHigh, vector2 outLow, vector2 outHigh, int doClamp)
{
      //remap in from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range

    vector2 out = in;
    out.x = remap(in.x, inLow.x, inHigh.x, outLow.x, outHigh.x, doClamp);
    out.y = remap(in.y, inLow.y, inHigh.y, outLow.y, outHigh.y, doClamp);

    return out;
}

vector2 remap(vector2 in, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
      //remap in from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range

    vector2 out = in;
    out.x = remap(in.x, inLow, inHigh, outLow, outHigh, doClamp);
    out.y = remap(in.y, inLow, inHigh, outLow, outHigh, doClamp);

    return out;
}

vector2 fgamma(vector2 in, vector2 g)
{
    vector2 out = in;

    out.x = fgamma(in.x, g.x);
    out.y = fgamma(in.y, g.y);

    return out;
}

vector2 fgamma(vector2 in, float g)
{
    vector2 out = in;

    out.x = fgamma(in.x, g);
    out.y = fgamma(in.y, g);

    return out;
}

vector2 invert(vector2 in, vector2 amount)
{
    vector2 out = in;
    out.x = amount.x - in.x;
    out.y = amount.y - in.y;
    return out;
}

vector2 invert(vector2 in, float amount)
{
    vector2 out = in;
    out.x = amount - in.x;
    out.y = amount - in.y;
    return out;
}

vector2 invert(vector2 in)
{
    vector2 out = invert(in, 1.0);
    return out;
}

vector2 clamp(vector2 in, vector2 low, vector2 high)
{
    vector2 out = in;
    out.x = clamp(in.x, low.x, high.x);
    out.y = clamp(in.y, low.y, high.y);

    return out;
}

vector2 clamp(vector2 in, float low, float high)
{
    vector2 out = in;
    out.x = clamp(in.x, low, high);
    out.y = clamp(in.y, low, high);

    return out;
}

vector2 contrast(vector2 in, vector2 amount, vector2 pivot)
{
    vector2 out = in;
    out.x = contrast(in.x, amount.x, pivot.x);
    out.y = contrast(in.y, amount.y, pivot.y);

    return out;
}

vector2 contrast(vector2 in, float amount, float pivot)
{
    vector2 out = in;
    out.x = contrast(in.x, amount, pivot);
    out.y = contrast(in.y, amount, pivot);

    return out;
}

vector2 divide(vector2 in1, vector2 amount)
{
    vector2 out = in1;
    out.x = in1.x / amount.x;
    out.y = in1.y / amount.y;

    return out;
}

vector2 divide(vector2 in1, float amount){
    vector2 out = in1;
    out.x = in1.x / amount;
    out.y = in1.y / amount;

    return out;
}

vector2 exponent(vector2 in, vector2 amount)
{
    vector2 out = in;
    out.x = exponent(in.x, amount.x);
    out.y = exponent(in.y, amount.y);
    return out;
}

vector2 exponent(vector2 in, float amount)
{
    vector2 out = in;
    out.x = exponent(in.x, amount);
    out.y = exponent(in.y, amount);
    return out;
}

vector2 max(vector2 in, vector2 amount)
{
    vector2 out = in;
    out.x = max(in.x, amount.x);
    out.y = max(in.y, amount.y);

    return out;
}

vector2 max(vector2 in, float amount)
{
    vector2 out = in;
    out.x = max(in.x, amount);
    out.y = max(in.y, amount);

    return out;
}

vector2 normalize(vector2 in)
{
    vector v = normalize(vector(in.x, in.y, 0));
    vector2 out;
    out.x = v[0];
    out.y = v[1];
    return out;
}

vector2 vscale(vector2 in, vector2 amount, vector2 center)
{
    vector2 out;
    out.x = (in.x - center.x)/amount.x + center.x;
    out.y = (in.y - center.y)/amount.y + center.y;
    return out;
}

vector2 multiply(vector2 in1, vector2 amount)
{
    vector2 out = in1;
    out.x = in1.x * amount.x;
    out.y = in1.y * amount.y;

    return out;
}

vector2 multiply(vector2 in1, float amount)
{
    vector2 out = in1;
    out.x = in1.x * amount;
    out.y = in1.y * amount;

    return out;
}

vector2 min(vector2 in, vector2 amount)
{
    vector2 out = in;
    out.x = min(in.x, amount.x);
    out.y = min(in.y, amount.y);

    return out;
}

vector2 min(vector2 in, float amount)
{
    vector2 out = in;
    out.x = min(in.x, amount);
    out.y = min(in.y, amount);

    return out;
}

vector2 fmod(vector2 in, vector2 amount)
{
    vector2 out = in;
    out.x = fmod(in.x, amount.x);
    out.y = fmod(in.y, amount.y);

    return out;
}

vector2 fmod(vector2 in, float amount)
{
    vector2 out = in;
    out.x = fmod(in.x, amount);
    out.y = fmod(in.y, amount);

    return out;
}

float mag(vector2 in){
    return length(vector(in.x, in.y, 0));
}

vector2 difference(vector2 fg, vector2 bg)
{
    vector2 out = { difference(fg.x, bg.x),
                    difference(fg.y, bg.y)
                  };

    return out;
}

vector2 rotate2d(vector2 in, float amount, vector2 center)
{
    vector2 out = in;    
    out.x = out.x - center.x;
    out.y = out.y - center.y;
    out.x = in.x*cos(amount) - in.y*sin(amount);
    out.y = in.y*cos(amount) + in.x*sin(amount);
    out.x = out.x+center.x;
    out.y = out.y+center.y;
    
    return out;
    
}

