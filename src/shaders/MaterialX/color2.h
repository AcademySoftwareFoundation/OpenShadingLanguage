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

color2 make_color2(float r, float g)
{
    color2 out;
    out.r = r;
    out.g = g;
    return out;
}

color2 make_color2(float r)
{
    color2 out;
    out.r = r;
    out.g = r;
    return out;

}

color2 make_color2()
{
    color2 out;
    out.r = 0;
    out.g = 0;
    return out;

}

color2 abs(color2 in){
    color2 out;
    out.r = abs(in.r);
    out.g = abs(in.g);
    return out;
}

color2 floor(color2 in){
    color2 out;
    out.r = floor(in.r);
    out.g = floor(in.g);
    return out;
}

color2 mix(color2 value1, color2 value2, float x ){
    color2 out;
    
    out.r = mix(value1.r, value2.r, x);
    out.g = mix(value1.g, value2.g, x);

    return out;
}

color2 add(color2 in, color2 amount){
    color2 out = in;
    out.r = in.r + amount.r;
    out.g = in.g + amount.g;
    return out;
}

color2 add(color2 in, float amount){
    color2 out = in;
    out.r = in.r + amount;
    out.g = in.g + amount;
    return out;
}

color2 subtract(color2 in, color2 amount){
    color2 out = in;
    out.r = in.r - amount.r;
    out.g = in.g - amount.g;
    return out;
}    

color2 subtract(color2 in, float amount){
    color2 out = in;
    out.r = in.r - amount;
    out.g = in.g - amount;
    return out;
}    

color2 smoothstep(color2 low, color2 high, color2 in){
    color2 out = in;
    out.r = smoothstep(low.r, high.r, in.r);
    out.g = smoothstep(low.g, high.g, in.g);
    return out;
}    

color2 smoothstep(float low, float high, color2 in){
    color2 out = in;
    out.r = smoothstep(low, high, in.r);
    out.g = smoothstep(low, high, in.g);
    return out;
}

color2 remap(color2 in, color2 inLow, color2 inHigh, color2 outLow, color2 outHigh, int doClamp)
{
      //remap in from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range

      color2 out = in;
      out.r = remap(in.r, inLow.r, inHigh.r, outLow.r, outHigh.r, doClamp);
      out.g = remap(in.g, inLow.g, inHigh.g, outLow.g, outHigh.g, doClamp);
      return out;
}

color2 remap(color2 in, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
      //remap in from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range

      color2 out = in;
      out.r = remap(in.r, inLow, inHigh, outLow, outHigh, doClamp);
      out.g = remap(in.g, inLow, inHigh, outLow, outHigh, doClamp);
      return out;
}

color2 fgamma(color2 in, color2 g){
    color2 out = in;

    out.r = fgamma(in.r, g.r);
    out.g = fgamma(in.g, g.g);
      return out;
}

color2 fgamma(color2 in, float g){
    color2 out = in;

    out.r = fgamma(in.r, g);
    out.g = fgamma(in.g, g);
      return out;
}

color2 invert(color2 in, color2 amount){
    color2 out = in;
    out.r = amount.r - in.r;
    out.g = amount.g - in.g;
    return out;
}

color2 invert(color2 in, float amount){
    color2 out = in;
    out.r = amount - in.r;
    out.g = amount - in.g;
    return out;
}

color2 invert(color2 in){
    color2 out = invert(in, 1.0);
    return out;
}

color2 clamp(color2 in, color2 low, color2 high){
    color2 out = in;
    out.r = clamp(in.r, low.r, high.r);
    out.g = clamp(in.g, low.g, high.g);
    return out;
}

color2 clamp(color2 in, float low, float high){
    color2 out = in;
    out.r = clamp(in.r, low, high);
    out.g = clamp(in.g, low, high);
    return out;
}

color2 contrast(color2 in, color2 amount, color2 pivot){
    color2 out = in;
    out.r = contrast(in.r, amount.r, pivot.r);
    out.g = contrast(in.g, amount.g, pivot.g);
    return out;
}    

color2 contrast(color2 in, float amount, float pivot){
    color2 out = in;
    out.r = contrast(in.r, amount, pivot);
    out.g = contrast(in.g, amount, pivot);
    return out;
}    

color2 divide(color2 in1, color2 amount){
    color2 out = in1;
    out.r = in1.r / amount.r;
    out.g = in1.g / amount.g;
    return out;
}

color2 divide(color2 in1, float amount){
    color2 out = in1;
    out.r = in1.r / amount;
    out.g = in1.g / amount;
    return out;
}

color2 exponent(color2 in, color2 amount){
    color2 out = in;
    out.r = exponent(in.r, amount.r);
    out.g = exponent(in.g, amount.g);
    return out;
}

color2 exponent(color2 in, float amount){
    color2 out = in;
    out.r = exponent(in.r, amount);
    out.g = exponent(in.g, amount);
    return out;
}

color2 max(color2 in, color2 amount){
    color2 out = in;
    out.r = max(in.r, amount.r);
    out.g = max(in.g, amount.g);
    return out;
}

color2 max(color2 in, float amount){
    color2 out = in;
    out.r = max(in.r, amount);
    out.g = max(in.g, amount);
    return out;
}

color2 multiply(color2 in1, color2 amount){
    color2 out = in1;
    out.r = in1.r * amount.r;
    out.g = in1.g * amount.g;
    return out;
}

color2 multiply(color2 in1, float amount){
    color2 out = in1;
    out.r = in1.r * amount;
    out.g = in1.g * amount;
    return out;
}

color2 min(color2 in, color2 amount){
    color2 out = in;
    out.r = min(in.r, amount.r);
    out.g = min(in.g, amount.g);
    return out;
}

color2 min(color2 in, float amount){
    color2 out = in;
    out.r = min(in.r, amount);
    out.g = min(in.g, amount);
    return out;
}

color2 fmod(color2 in, color2 amount){
    color2 out = in;
    out.r = fmod(in.r, amount.r);
    out.g = fmod(in.g, amount.g);
    return out;
}

color2 fmod(color2 in, float amount){
    color2 out = in;
    out.r = fmod(in.r, amount);
    out.g = fmod(in.g, amount);
    return out;
}

color2 unpremult(color2 in){
    color2 out = in;
    out.r = in.r / in.g;
    return out;
}

color2 premult(color2 in){
    color2 out = in;
    out.r = in.r * in.g;
    return out;
}

color2 cout(color2 fg, color2 bg){
    color2 out;
    out.r = fg.r * (1 - bg.g);
    out.g = fg.g * (1 - bg.g);
    return out;
}

color2 over(color2 fg, color2 bg){
    color2 over;
    over.r = fg.r + bg.r * (1 - fg.g);
    over.g = fg.g + bg.g * (1 - fg.g);
    return over;
}

color2 cmatte(color2 fg, color2 bg){
    color2 out;
    out.r = (fg.r * fg.g) + bg.r * (1 - fg.g);
    out.g = fg.g + bg.g * (1 - fg.g);
    return out;
}

color2 cmask(color2 fg, color2 bg){
    color2 out;
    out.r = bg.r * fg.g;
    out.g = bg.g * fg.g;
    return out;
}

color2 cin(color2 fg, color2 bg){
    color2 out;
    out.r = fg.r * bg.g;
    out.g = fg.g * bg.g;
    return out;
}

color2 disjointover(color2 fg, color2 bg){
    float summedAlpha = fg.g + bg.g;

    color2 out;
    if (summedAlpha <= 1){
        out.r = fg.r + bg.r;
    }else{
        out.r = fg.r + ((bg.r * (1-fg.g)) / bg.g);
    }

    out.g = min(summedAlpha, 1);
    return out;
}

color2 dodge(color2 fg, color2 bg){
    color2 out = { dodge(fg.r, bg.r),  
                 dodge(fg.g, bg.g)
               };

    return out;
}

color2 screen(color2 fg, color2 bg){
    color2 out = { screen(fg.r, bg.r),  
                 screen(fg.g, bg.g)
               };

    return out;
}

color2 overlay(color2 fg, color2 bg){
    color2 out = { overlay(fg.r, bg.r),  
                 overlay(fg.g, bg.g)
                };

    return out;
}

color2 difference(color2 fg, color2 bg){
    color2 out = { difference(fg.r, bg.r),  
                 difference(fg.g, bg.g)
               };
    return out;
}
