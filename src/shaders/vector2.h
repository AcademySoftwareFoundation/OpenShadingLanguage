// Open Shading Language : Copyright (c) 2009-2017 Sony Pictures Imageworks Inc., et al.
// https://github.com/imageworks/OpenShadingLanguage/blob/master/LICENSE

#pragma once
#define VECTOR2_H


// vector2 is a 2D vector
struct vector2
{
    float x;
    float y;
};



//
// For vector2, define math operators to match vector
//

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

vector2 __operator__add__(int a, vector2 b)
{
    return vector2(a, a) + b;
}

vector2 __operator__add__(float a, vector2 b)
{
    return vector2(a, a) + b;
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

vector2 __operator__sub__(int a, vector2 b)
{
    return vector2(a, a) - b;
}

vector2 __operator__sub__(float a, vector2 b)
{
    return vector2(a, a) - b;
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

vector2 __operator__mul__(int a, vector2 b)
{
    return b * vector2(a, a);
}

vector2 __operator__mul__(float a, vector2 b)
{
    return b * vector2(a, a);
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

vector2 __operator__div__(int a, vector2 b)
{
    return vector2(a, a) / b;
}

vector2 __operator__div__(float a, vector2 b)
{
    return vector2(a, a) / b;
}

int __operator__eq__(vector2 a, vector2 b)
{
    return (a.x == b.x) && (a.y == b.y);
}

int __operator__ne__(vector2 a, vector2 b)
{
    return (a.x != b.x) || (a.y != b.y);
}




//
// For vector2, define most of the stdosl functions to match vector
//

vector2 abs(vector2 a)
{
    return vector2 (abs(a.x), abs(a.y));
}

vector2 floor(vector2 a)
{
    return vector2 (floor(a.x), floor(a.y));
}

vector2 mix(vector2 a, vector2 b, float x )
{
    return vector2 (mix(a.x, b.x, x), mix(a.y, b.y, x));
}

float dot(vector2 a, vector2 b)
{
    return (a.x * b.x + a.y * b.y);
}

float length (vector2 a)
{
    return hypot (a.x, a.y);
}

vector2 smoothstep(vector2 low, vector2 high, vector2 in)
{
    return vector2 (smoothstep(low.x, high.x, in.x),
                    smoothstep(low.y, high.y, in.y));
}

vector2 smoothstep(float low, float high, vector2 in)
{
    return vector2 (smoothstep(low, high, in.x),
                    smoothstep(low, high, in.y));
}

vector2 clamp(vector2 in, vector2 low, vector2 high)
{
    return vector2 (clamp(in.x, low.x, high.x),
                    clamp(in.y, low.y, high.y));
}

vector2 clamp(vector2 in, float low, float high)
{
    return clamp(in, vector2(low, low), vector2(high, high));
}

vector2 max(vector2 a, vector2 b)
{
    return vector2 (max(a.x, b.x),
                    max(a.y, b.y));
}

vector2 max(vector2 a, float b)
{
    return max(a, vector2(b, b));
}

vector2 normalize(vector2 a)
{
    return a / length(a);
}

vector2 min(vector2 a, vector2 b)
{
    return vector2 (min(a.x, a.x),
                    min(b.y, b.y));
}

vector2 min(vector2 a, float b)
{
    return min(a, vector2(b, b));
}

vector2 fmod(vector2 a, vector2 b)
{
    return vector2 (fmod(a.x, b.x),
                    fmod(a.y, b.y));
}

vector2 fmod(vector2 a, float b)
{
    return fmod(a, vector2(b, b));
}

vector2 rotate2d(vector2 in, float amount, vector2 center)
{
    vector2 out = in - center;
    float sine, cosine;
    sincos(amount, sine, cosine);
    out.x = in.x * cosine - in.y * sine;
    out.y = in.y * cosine + in.x * sine;
    out = out + center;
    return out;
}

vector2 pow (vector2 in, vector2 amount)
{
    return vector2 (pow(in.x, amount.x), pow(in.y, amount.y));
}

vector2 pow (vector2 in, float amount)
{
    return pow(in, vector2(amount, amount));
}
