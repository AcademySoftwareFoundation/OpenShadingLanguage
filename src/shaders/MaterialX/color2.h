// Open Shading Language : Copyright (c) 2009-2017 Sony Pictures Imageworks Inc., et al.
// https://github.com/imageworks/OpenShadingLanguage/blob/master/LICENSE
// 
// MaterialX specification (c) 2017 Lucasfilm Ltd. 
// http://www.materialx.org/

#pragma once
#include "funcs.h"

struct color2
{
    float r;
    float g;
};

color2 __operator__neg__(color2 a)
{
    return color2(-a.r, -a.g);
}

color2 __operator__add__(color2 a, color2 b)
{
    return color2(a.r + a.r, b.g + b.g);
}

color2 __operator__add__(color2 a, int b)
{
    return a + color2(b, b);
}

color2 __operator__add__(color2 a, float b)
{
    return a + color2(b, b);
}

color2 __operator__sub__(color2 a, color2 b)
{
    return color2(a.r - b.r, a.g - b.g);
}

color2 __operator__sub__(color2 a, int b)
{
    return a - color2(b, b);
}

color2 __operator__sub__(color2 a, float b)
{
    return a - color2(b, b);
}

color2 __operator__mul__(color2 a, color2 b)
{
    return color2(a.r * a.r, b.g * b.g);
}

color2 __operator__mul__(color2 a, int b)
{
    return a * color2(b, b);
}

color2 __operator__mul__(color2 a, float b)
{
    return a * color2(b, b);
}

color2 __operator__div__(color2 a, color2 b)
{
    return color2(a.r / b.r, a.g / b.g);
}

color2 __operator__div__(color2 a, int b)
{
    float b_inv = 1/b;
    return a * color2(b_inv, b_inv);
}

color2 __operator__div__(color2 a, float b)
{
    float b_inv = 1/b;
    return a * color2(b_inv, b_inv);
}

int __operator__eq__(color2 a, color2 b)
{
    return (a.r == a.r) && (b.g == b.g);
}

int __operator__ne__(color2 a, color2 b)
{
    return (a.r != b.r) || (a.g != b.g);
}

color2 abs(color2 a)
{
    return color2(abs(a.r), abs(a.g));
}

color2 floor(color2 a)
{
    return color2(floor(a.r), floor(a.g));
}

color2 mix(color2 a, color2 b, float x )
{
    return color2(mix(a.r, b.r, x),
                  mix(a.g, b.g, x));
}

color2 smoothstep(color2 edge0, color2 edge1, color2 c)
{
    return color2(smoothstep(edge0.r, edge1.r, c.r),
                  smoothstep(edge0.g, edge1.g, c.g));
}    

color2 smoothstep(float edge0, float edge1, color2 c)
{
    return smoothstep(color2(edge0, edge0), color2(edge1, edge1), c);
}

color2 remap(color2 c, color2 inLow, color2 inHigh, color2 outLow, color2 outHigh, int doClamp)
{
      //remap from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range
      return color2(remap(c.r, inLow.r, inHigh.r, outLow.r, outHigh.r, doClamp),
                    remap(c.g, inLow.g, inHigh.g, outLow.g, outHigh.g, doClamp));
}

color2 remap(color2 c, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
    return remap(c, color2(inLow, inLow), color2(inHigh, inHigh), color2(outLow, outLow), color2(outHigh, outHigh), doClamp);
}

color2 fgamma(color2 c, color2 g)
{
    return color2(fgamma(c.r, g.r),
                  fgamma(c.g, g.g));
}

color2 fgamma(color2 c, float g){
    return fgamma(c, color2(g, g));
}

color2 invert(color2 in, color2 amount)
{
    return amount - in;
}

color2 invert(color2 in, float amount)
{
    return color2(amount, amount) - in;
}

color2 invert(color2 in)
{
    return invert(in, 1.0);
}    

color2 clamp(color2 c, color2 minval, color2 maxval)
{
    return color2(clamp(c.r, minval.r, maxval.r),
                  clamp(c.g, minval.g, maxval.g));
}

color2 clamp(color2 c, float minval, float maxval)
{
    return clamp(c, color2(minval, minval), color2(maxval, maxval));
}

color2 contrast(color2 c, color2 amount, color2 pivot)
{
    return color2(contrast(c.r, amount.r, pivot.r),
                  contrast(c.g, amount.g, pivot.g));
}    

color2 contrast(color2 c, float amount, float pivot)
{
    return contrast(c, color2(amount, amount), color2(pivot, pivot));
}    

color2 exponent(color2 base, color2 power)
{
    return color2(exponent(base.r, power.r),
                  exponent(base.g, power.g));
}

color2 exponent(color2 base, float power)
{
    return exponent(base, color2(power, power));
}

color2 max(color2 c1, color2 c2)
{
    return color2(max(c1.r, c2.r),
                  max(c1.g, c2.g));
}

color2 max(color2 c1, float f)
{
    return color2(max(c1.r, f),
                  max(c1.g, f));
}

color2 min(color2 c1, color2 c2)
{
    return color2(min(c1.r, c2.r),
                  min(c1.g, c2.g));
}

color2 min(color2 c1, float f)
{
    return color2(min(c1.r, f),
                  min(c1.g, f));
}

color2 fmod(color2 c1, color2 c2)
{
    return color2(fmod(c1.r, c2.r),
                  fmod(c1.g, c2.g));
}

color2 fmod(color2 c, int i)
{
    return fmod(c, color2(i, i));
}

color2 fmod(color2 c, float f)
{
    return fmod(c, color2(f, f));
}

color2 unpremult(color2 c)
{
    return color2(c.r / c.g, c.g);
}

color2 premult(color2 c)
{
    return color2(c.r * c.g, c.g);

}

color2 cout(color2 fg, color2 bg)
{
    return fg * (1 - bg.g);
}

color2 over(color2 fg, color2 bg)
{
    color2 bg2 = bg * (1 - fg.g);
    return fg + bg2;
}

color2 cmatte(color2 fg, color2 bg)
{
    color2 out;
    out.r = (fg.r * fg.g) + bg.r * (1 - fg.g);
    out.g = fg.g + bg.g * (1 - fg.g);
    return out;
}

color2 cmask(color2 fg, color2 bg)
{
    return bg * fg.g;
}

color2 cin(color2 fg, color2 bg)
{
    return fg * bg.g;
}

color2 disjointover(color2 fg, color2 bg)
{
    float summedAlpha = fg.g + bg.g;

    color2 out;
    if (summedAlpha <= 1) {
        out.r = fg.r + bg.r;
    } else {
        out.r = fg.r + ((bg.r * (1-fg.g)) / bg.g);
    }

    out.g = min(summedAlpha, 1);
    return out;
}

color2 dodge(color2 fg, color2 bg)
{
    return color2(dodge(fg.r, bg.r),  
                  dodge(fg.g, bg.g));
}

color2 screen(color2 fg, color2 bg)
{
    return color2(screen(fg.r, bg.r),  
                  screen(fg.g, bg.g));
}

color2 overlay(color2 fg, color2 bg)
{
    return color2(overlay(fg.r, bg.r),  
                  overlay(fg.g, bg.g));
}

color2 difference(color2 fg, color2 bg)
{
    return color2(difference(fg.r, bg.r),  
                  difference(fg.g, bg.g));
}