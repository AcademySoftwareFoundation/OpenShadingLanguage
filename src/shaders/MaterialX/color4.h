// Open Shading Language : Copyright (c) 2009-2017 Sony Pictures Imageworks Inc., et al.
// https://github.com/imageworks/OpenShadingLanguage/blob/master/LICENSE
// 
// MaterialX specification (c) 2017 Lucasfilm Ltd. 
// http://www.materialx.org/

#pragma once
#include "funcs.h"

struct color4
{
    color rgb;
    float a;
};

color4 __operator__neg__(color4 c)
{
    return color4(-c.rgb, -c.a);
}

color4 __operator__add__(color4 c1, color4 c2)
{
    return color4(c1.rgb + c2.rgb, c1.a + c2.a);
}

color4 __operator__add__(color4 c, int i)
{
    return c + color4(color(i), i);
}

color4 __operator__add__(color4 c, float f)
{
    return c + color4(color(f), f);
}

color4 __operator__sub__(color4 c1, color4 c2)
{
    return color4(c1.rgb - c2.rgb, c1.a - c2.a);
}

color4 __operator__sub__(color4 c, int i)
{
    return c - color4(color(i), i);
}

color4 __operator__sub__(color4 c, float f)
{
    return c - color4(color(f), f);
}

color4 __operator__mul__(color4 c1, color4 c2)
{
    return color4(c1.rgb * c2.rgb, c1.a * c2.a);
}

color4 __operator__mul__(color4 c, int i)
{
    return c * color4(color(i), i);
}

color4 __operator__mul__(color4 c, float f)
{
    return c * color4(color(f), f);
}

color4 __operator__div__(color4 c1, color4 c2)
{
    return color4(c1.rgb / c2.rgb, c1.a / c2.a);
}

color4 __operator__div__(color4 c, int i)
{
    float i_inv = 1/i;
    return c * color4(color(i_inv), i_inv);
}

color4 __operator__div__(color4 c, float f)
{
    float f_inv = 1/f;
    return c * color4(color(f_inv), f_inv);
}

int __operator__eq__(color4 c1, color4 c2)
{
    return (c1.rgb == c2.rgb) && (c1.a == c2.a);
}

int __operator__ne__(color4 c1, color4 c2)
{
    return (c1.rgb != c2.rgb) || (c1.a != c2.a);
}

color4 abs(color4 c)
{
    return color4(abs(c.rgb), abs(c.a));
}

color4 floor(color4 c)
{
    return color4(floor(c.rgb), floor(c.a));
}

color4 mix(color4 c1, color4 c2, float x )
{
    return color4(mix(c1.rgb, c2.rgb, x),
                  mix(c1.a, c2.a, x));
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

color4 remap(color4 c, color4 inLow, color4 inHigh, color4 outLow, color4 outHigh, int doClamp)
{
      //remap from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range
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

color4 fgamma(color4 c, color4 g)
{
    return color4(fgamma(c.rgb, g.rgb),
                  fgamma(c.a, g.a));
}

color4 fgamma(color4 c, float g){
    return fgamma(c, color4(color(g), g));
}

color4 invert(color4 in, color4 amount)
{
    return amount - in;
}

color4 invert(color4 in, float amount)
{
    return color4(color(amount), amount) - in;
}

color4 invert(color4 in)
{
    return invert(in, 1.0);
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

//QUESTION: should this operate on the alpha as well?
color4 contrast(color4 c, color4 amount, color4 pivot)
{
    return color4(contrast(c.rgb, amount.rgb, pivot.rgb),
                  contrast(c.a, amount.a, pivot.a));
}    

color4 contrast(color4 c, float amount, float pivot)
{
    return contrast(c, color4(color(amount), amount), color4(color(pivot), pivot));
}    

color4 exponent(color4 base, color4 power)
{
    return color4(exponent(base.rgb, power.rgb),
                  exponent(base.a, power.a));
}

color4 exponent(color4 base, float power){
    return exponent(base, color4(color(power), power));
}

color4 max(color4 c1, color4 c2)
{
    return color4(max(c1.rgb, c2.rgb),
                  max(c1.a, c2.a));
}

color4 max(color4 c1, float f)
{
    return color4(max(c1.rgb, f),
                  max(c1.a, f));
}

color4 min(color4 c1, color4 c2)
{
    return color4(min(c1.rgb, c2.rgb),
                  min(c1.a, c2.a));
}

color4 min(color4 c1, float f)
{
    return color4(min(c1.rgb, f),
                  min(c1.a, f));
}

color4 fmod(color4 c1, color4 c2)
{
    return color4(fmod(c1.rgb, c2.rgb),
                  fmod(c1.a, c2.a));
}

color4 fmod(color4 c, int i)
{
    return fmod(c, color4(color(i), i));
}

color4 fmod(color4 c, float f)
{
    return fmod(c, color4(color(f), f));
}

color4 unpremult(color4 c)
{
    return color4(c.rgb / c.a, c.a);
}

color4 premult(color4 c)
{
    return color4(c.rgb * c.a, c.a);

}

color4 cluminance( color4 in, color lumacoeffs)
{
    float l =  cluminance(in.rgb, lumacoeffs);
    return color4(color(l), in.a);
}

color4 saturate(color4 in, float amount){
    color hsv3 = transformc("rgb","hsv", in.rgb);
    hsv3[1] *= amount;
    color4 out = {transformc("hsv","rgb", hsv3), in.a};
    return out;
}

color4 hueshift(color4 in, float amount){
    color hsv3 = transformc("rgb","hsv", in.rgb);
    hsv3[0] += amount;
    hsv3[0] = fmod(hsv3[0], 1.0);
    color4 out = {transformc("hsv","rgb", hsv3), in.a};
    return out;
}

color4 over(color4 fg, color4 bg){
    color4 over;
    float x = 1 - fg.a;
    over.rgb = fg.rgb + bg.rgb * (1 - fg.a);

    over.a = fg.a + bg.a * (1 - fg.a);
    return over;
}

color4 cout(color4 fg, color4 bg){
    color4 out;
    out.rgb = fg.rgb * (1 - bg.a);

    out.a = fg.a * (1 - bg.a);
    return out;
}

color4 cmatte(color4 fg, color4 bg){
    color4 out;
    out.rgb = (fg.rgb * fg.a) + bg.rgb * (1 - fg.a);

    out.a = fg.a + bg.a * (1 - fg.a);
    return out;
}

color4 cmask(color4 fg, color4 bg){
    color4 out;
    out.rgb = bg.rgb * fg.a;

    out.a = bg.a * fg.a;
    return out;
}

color4 cin(color4 fg, color4 bg){
    color4 out;
    out.rgb = fg.rgb * bg.a;

    out.a = fg.a * bg.a;
    return out;
}

color4 disjointover(color4 fg, color4 bg){
    float summedAlpha = fg.a + bg.a;

    color4 out;
    if (summedAlpha <= 1){
        out.rgb = fg.rgb + bg.rgb;
    }else{
        float x = (1-fg.a) / bg.a;
        out.rgb = fg.rgb + bg.rgb * x;
        
    }

    out.a = min(summedAlpha, 1);
    return out;
}

color4 dodge(color4 fg, color4 bg){
    color4 out = { dodge(fg.rgb, bg.rgb),  

                 dodge(fg.a, bg.a)
               };

    return out;
}

color4 screen(color4 fg, color4 bg){
    color4 out = { screen(fg.rgb, bg.rgb),   
                 screen(fg.a, bg.a)
               };

    return out;
}

color4 overlay(color4 fg, color4 bg){
    color4 out = { overlay(fg.rgb, bg.rgb),  
                 overlay(fg.a, bg.a)
               };

    return out;
}

color4 difference(color4 fg, color4 bg){
    color4 out = { difference(fg.rgb, bg.rgb),  
                 difference(fg.a, bg.a)
               };

    return out;
}