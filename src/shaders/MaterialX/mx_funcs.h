// Open Shading Language : Copyright (c) 2009-2017 Sony Pictures Imageworks Inc., et al.
// https://github.com/imageworks/OpenShadingLanguage/blob/master/LICENSE
// 
// MaterialX specification (c) 2017 Lucasfilm Ltd. 
// http://www.materialx.org/

#pragma once
#include "color2.h"
#include "color4.h"
#include "vector2.h"
#include "vector4.h"
#include "mx_types.h"

///////////////////////////////////////////////////////////////////////////
// This file contains lots of functions helpful in the implementation of
// the MaterialX nodes.
///////////////////////////////////////////////////////////////////////////



// remap `in` from [inLow, inHigh] to [outLow, outHigh], optionally clamping
// to the new range.
//
float remap(float in, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
      float x = (in - inLow)/(inHigh-inLow);
      if (doClamp == 1) {
           x = clamp(x, 0, 1);
      }
      return outLow + (outHigh - outLow) * x;
}

color remap(color in, color inLow, color inHigh, color outLow, color outHigh, int doClamp)
{
      color x = (in - inLow) / (inHigh - inLow);
      if (doClamp == 1) {
           x = clamp(x, 0, 1);
      }
      return outLow + (outHigh - outLow) * x;
}

color remap(color in, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
      color x = (in - inLow) / (inHigh - inLow);
      if (doClamp == 1) {
           x = clamp(x, 0, 1);
      }
      return outLow + (outHigh - outLow) * x;
}

color2 remap(color2 c, color2 inLow, color2 inHigh, color2 outLow, color2 outHigh, int doClamp)
{
      return color2(remap(c.r, inLow.r, inHigh.r, outLow.r, outHigh.r, doClamp),
                    remap(c.a, inLow.a, inHigh.a, outLow.a, outHigh.a, doClamp));
}

color2 remap(color2 c, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
    return remap(c, color2(inLow, inLow), color2(inHigh, inHigh), color2(outLow, outLow), color2(outHigh, outHigh), doClamp);
}

color4 remap(color4 c, color4 inLow, color4 inHigh, color4 outLow, color4 outHigh, int doClamp)
{
      return color4(remap(c.rgb, inLow.rgb, inHigh.rgb, outLow.rgb, outHigh.rgb, doClamp),
                    remap(c.a, inLow.a, inHigh.a, outLow.a, outHigh.a, doClamp));
}

color4 remap(color4 c, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
    color4 c4_inLow = color4(color(inLow), inLow);
    color4 c4_inHigh = color4(color(inHigh), inHigh);
    color4 c4_outLow = color4(color(outLow), outLow);
    color4 c4_outHigh = color4(color(outHigh), outHigh);
    return remap(c, c4_inLow, c4_inHigh, c4_outLow, c4_outHigh, doClamp);
}

vector2 remap(vector2 in, vector2 inLow, vector2 inHigh, vector2 outLow, vector2 outHigh, int doClamp)
{
    return vector2 (remap(in.x, inLow.x, inHigh.x, outLow.x, outHigh.x, doClamp),
                    remap(in.y, inLow.y, inHigh.y, outLow.y, outHigh.y, doClamp));
}

vector2 remap(vector2 in, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
    return vector2 (remap(in.x, inLow, inHigh, outLow, outHigh, doClamp),
                    remap(in.y, inLow, inHigh, outLow, outHigh, doClamp));
}

vector4 remap(vector4 in, vector4 inLow, vector4 inHigh, vector4 outLow, vector4 outHigh, int doClamp)
{
    return vector4 (remap(in.x, inLow.x, inHigh.x, outLow.x, outHigh.x, doClamp),
                    remap(in.y, inLow.y, inHigh.y, outLow.y, outHigh.y, doClamp),
                    remap(in.z, inLow.z, inHigh.z, outLow.z, outHigh.z, doClamp),
                    remap(in.w, inLow.w, inHigh.w, outLow.w, outHigh.w, doClamp));
}

vector4 remap(vector4 in, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
    return vector4 (remap(in.x, inLow, inHigh, outLow, outHigh, doClamp),
                    remap(in.y, inLow, inHigh, outLow, outHigh, doClamp),
                    remap(in.z, inLow, inHigh, outLow, outHigh, doClamp),
                    remap(in.w, inLow, inHigh, outLow, outHigh, doClamp));
}



float fgamma(float in, float g)
{
    return sign(in) * pow(abs(in), g);
}

color fgamma(color in, color g)
{
    return sign(in) * pow(abs(in), g);
}

color fgamma(color in, float g)
{
    return sign(in) * pow(abs(in), g);
}

color2 fgamma(color2 c, color2 a)
{
    return color2(fgamma(c.r, a.r), fgamma(c.a, a.a));
}

color2 fgamma(color2 c, float a)
{
    return fgamma(c, color2(a, a));
}

color4 fgamma(color4 a, color4 b)
{
    return color4(fgamma(a.rgb, b.rgb), fgamma(a.a, b.a));
}

color4 fgamma(color4 a, float b)
{
    return fgamma(a, color4(color(b), b));
}

vector2 fgamma(vector2 in, vector2 g)
{
    return vector2 (fgamma(in.x, g.x), fgamma(in.y, g.y));
}

vector2 fgamma(vector2 in, float g)
{
    return vector2 (fgamma(in.x, g), fgamma(in.y, g));
}

vector4 fgamma(vector4 in, vector4 g)
{
    return vector4 (fgamma(in.x, g.x),
                    fgamma(in.y, g.y),
                    fgamma(in.z, g.z),
                    fgamma(in.w, g.w));
}

vector4 fgamma(vector4 in, float g)
{
    return vector4 (fgamma(in.x, g),
                    fgamma(in.y, g),
                    fgamma(in.z, g),
                    fgamma(in.w, g));
}



//
// contrast scales the input around a central `pivot` value.
//
float contrast(float in, float amount, float pivot)
{
    float out = in - pivot;
    out *= amount;
    out += pivot;
    return out;
}

color contrast(color in, color amount, color pivot)
{
    color out = in - pivot;
    out *= amount;
    out += pivot;
    return out;
}

color contrast(color in, float amount, float pivot)
{
    color out = in - pivot;
    out *= amount;
    out += pivot;
    return out;
}

color2 contrast(color2 c, color2 amount, color2 pivot)
{
    return color2(contrast(c.r, amount.r, pivot.r),
                  contrast(c.a, amount.a, pivot.a));
}

color2 contrast(color2 c, float amount, float pivot)
{
    return contrast(c, color2(amount, amount), color2(pivot, pivot));
}

color4 contrast(color4 c, color4 amount, color4 pivot)
{
    return color4(contrast(c.rgb, amount.rgb, pivot.rgb),
                  contrast(c.a, amount.a, pivot.a));
}

color4 contrast(color4 c, float amount, float pivot)
{
    return contrast(c, color4(color(amount), amount), color4(color(pivot), pivot));
}

vector2 contrast(vector2 in, vector2 amount, vector2 pivot)
{
    return vector2 (contrast(in.x, amount.x, pivot.x),
                    contrast(in.y, amount.y, pivot.y));
}

vector2 contrast(vector2 in, float amount, float pivot)
{
    return contrast(in, vector2(amount, amount), vector2(pivot, pivot));
}

vector4 contrast(vector4 in, vector4 amount, vector4 pivot)
{
    return vector4 (contrast(in.x, amount.x, pivot.x),
                    contrast(in.y, amount.y, pivot.y),
                    contrast(in.z, amount.z, pivot.z),
                    contrast(in.w, amount.w, pivot.w));
}

vector4 contrast(vector4 in, float amount, float pivot)
{
    return vector4 (contrast(in.x, amount, pivot),
                    contrast(in.y, amount, pivot),
                    contrast(in.z, amount, pivot),
                    contrast(in.w, amount, pivot));
}


//
// fractional Brownian motion
//
float fBm( point position, int octaves, float lacunarity, float diminish, string noisetype)
{
    float out = 0;
    float amp = 1.0;
    point p = position;

    for (int i = 0;  i < octaves;  i += 1) {
        out += amp * noise(noisetype, p);
        amp *= diminish;
        p *= lacunarity;
    }
    return out;
}

color fBm( point position, int octaves, float lacunarity, float diminish, string noisetype)
{

    color out = 0;
    float amp = 1.0;
    point p = position;

    for (int i = 0;  i < octaves;  i += 1) {
        out += amp * (color)noise(noisetype, p);
        amp *= diminish;
        p *= lacunarity;
    }
    return out;
}
