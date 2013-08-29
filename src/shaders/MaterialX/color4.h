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

color4 make_color4(float r, float g, float b, float a)
{
    color4 out;
    out.rgb = color(r,g,b);
    out.a = a;
    return out;
}

color4 make_color4(float r)
{
    color4 out;
    out.rgb = color(r,r,r);
    out.a = r;
    return out;
}

color4 make_color4(color rgb, float a)
{
    color4 out;
    out.rgb = rgb;
    out.a = a;
    return out;
}

color4 make_color4(color rgb)
{
    color4 out;
    out.rgb = rgb;
    out.a = 1;
    return out;
}

color4 abs(color4 in){
    color4 out;
    out.rgb = abs(in.rgb);
    out.a = abs(in.a);
    return out;
}

color4 floor(color4 in){
    color4 out;
    out.rgb = floor(in.rgb);
    out.a = floor(in.a);
    return out;
}

color4 mix(color4 value1, color4 value2, float x ){
    color4 out;
    
    out.rgb = mix(value1.rgb, value2.rgb, x);
    out.a = mix(value1.a, value2.a, x);

    return out;
}

color4 add(color4 in, color4 amount){
    color4 out = in;
    out.rgb = in.rgb + amount.rgb;
    out.a = in.a + amount.a;
    return out;
}

color4 add(color4 in, float amount){
    color4 out = in;
    out.rgb = in.rgb + amount;
    out.a = in.a + amount;
    return out;
}

color4 luminance(color4 in){
    color4 out = {color(luminance(in.rgb)), in.a};
    return out;
}

color4 subtract(color4 in, color4 amount){
    color4 out = in;
    out.rgb = in.rgb - amount.rgb;
    out.a = in.a - amount.a;
    return out;
}

color4 subtract(color4 in, float amount){
    color4 out = in;
    out.rgb = in.rgb - amount;
    out.a = in.a - amount;
    return out;
}

color4 smoothstep(color4 low, color4 high, color4 in){
    color4 out = in;
    out.rgb = smoothstep(low.rgb, high.rgb, in.rgb);
    out.a = smoothstep(low.a, high.a, in.a);
    return out;
}

color4 smoothstep(float low, float high, color4 in){
    color4 out = in;
    out.rgb = smoothstep(low, high, in.rgb);
    out.a = smoothstep(low, high, in.a);
    return out;
}

color4 remap(color4 in, color4 inLow, color4 inHigh, color4 outLow, color4 outHigh, int doClamp)
{
      //remap in from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range

      color4 out = in;
      out.rgb = remap(in.rgb, inLow.rgb, inHigh.rgb, outLow.rgb, outHigh.rgb, doClamp);
      out.a = remap(in.a, inLow.a, inHigh.a, outLow.a, outHigh.a, doClamp);
      return out;
}

color4 remap(color4 in, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
      //remap in from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range

      color4 out = in;
      out.rgb = remap(in.rgb, inLow, inHigh, outLow, outHigh, doClamp);
      out.a = remap(in.a, inLow, inHigh, outLow, outHigh, doClamp);
      return out;
}

color4 fgamma(color4 in, color4 g){
    color4 out = in;

    out.rgb = fgamma(in.rgb, g.rgb);
    out.a = fgamma(in.a, g.a);
      return out;
}

color4 fgamma(color4 in, float g){
    color4 out = in;

    out.rgb = fgamma(in.rgb, g);
    out.a = fgamma(in.a, g);
      return out;
}

color4 invert(color4 in, color4 amount){
    color4 out = in;
    out.rgb = amount.rgb - in.rgb;
    out.a = amount.a - in.a;
    return out;
}

color4 invert(color4 in, float amount){
    color4 out = in;
    out.rgb = color(amount) - in.rgb;
    out.a = amount - in.a;
    return out;
}

color4 invert(color4 in){
    color4 out = invert(in, 1.0);
    return out;
}

color4 clamp(color4 in, color4 low, color4 high){
    color4 out = in;
    out.rgb = clamp(in.rgb, low.rgb, high.rgb);
    out.a = clamp(in.a, low.a, high.a);
    return out;
}

color4 clamp(color4 in, float low, float high){
    color4 out = in;
    out.rgb = clamp(in.rgb, low, high);
    out.a = clamp(in.a, low, high);
    return out;
}

color4 contrast(color4 in, color4 amount, color4 pivot){
    color4 out = in;
    out.rgb = contrast(in.rgb, amount.rgb, pivot.rgb);
    out.a = contrast(in.a, amount.a, pivot.a);
    return out;
}

color4 contrast(color4 in, float amount, float pivot){
    color4 out = in;
    out.rgb = contrast(in.rgb, amount, pivot);
    out.a = contrast(in.a, amount, pivot);
    return out;
}

color4 divide(color4 in1, color4 amount){
    color4 out = in1;
    out.rgb = in1.rgb / amount.rgb;
    out.a = in1.a / amount.a;
    return out;
}

color4 divide(color4 in1, float amount){
    color4 out = in1;
    out.rgb = in1.rgb / amount;
    out.a = in1.a / amount;
    return out;
}

color4 exponent(color4 in, color4 amount){
    color4 out = in;
    out.rgb = exponent(in.rgb, amount.rgb);
    out.a = exponent(in.a, amount.a);
    return out;
}


color4 exponent(color4 in, float amount){
    color4 out = in;
    out.rgb = exponent(in.rgb, amount);
    out.a = exponent(in.a, amount);
    return out;
}

color4 max(color4 in, color4 amount){
    color4 out = in;
    out.rgb = max(in.rgb, amount.rgb);
    out.a = max(in.a, amount.a);
    return out;
}

color4 max(color4 in, float amount){
    color4 out = in;
    out.rgb = max(in.rgb, amount);
    out.a = max(in.a, amount);
    return out;
}

color4 multiply(color4 in1, color4 amount){
    color4 out = in1;
    out.rgb = in1.rgb * amount.rgb;
    out.a = in1.a * amount.a;
    return out;
}

color4 multiply(color4 in1, float amount){
    color4 out = in1;
    out.rgb = in1.rgb * amount;
    out.a = in1.a * amount;
    return out;
}

color4 min(color4 in, color4 amount){
    color4 out = in;
    out.rgb = min(in.rgb, amount.rgb);
    out.a = min(in.a, amount.a);
    return out;
}

color4 min(color4 in, float amount){
    color4 out = in;
    out.rgb = min(in.rgb, amount);
    out.a = min(in.a, amount);
    return out;
}

color4 fmod(color4 in, color4 amount){
    color4 out = in;
    out.rgb = fmod(in.rgb, amount.rgb);
    out.a = fmod(in.a, amount.a);
    return out;
}

color4 fmod(color4 in, float amount){
    color4 out = in;
    out.rgb = fmod(in.rgb, amount);
    out.a = fmod(in.a, amount);
    return out;
}

color4 unpremult(color4 in){
   color4 out = in;
   out.rgb = unpremult(in.rgb, in.a);
   return out;
}

color4 premult(color4 in){
   color4 out = in;
   out.rgb = premult(in.rgb, in.a);
   return out;
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

