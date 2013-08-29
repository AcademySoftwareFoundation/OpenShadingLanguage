// Open Shading Language : Copyright (c) 2009-2017 Sony Pictures Imageworks Inc., et al.
// https://github.com/imageworks/OpenShadingLanguage/blob/master/LICENSE
// 
// MaterialX specification (c) 2017 Lucasfilm Ltd. 
// http://www.materialx.org/

#pragma once
#include "mx_types.h"

#if defined(FLOAT) || defined(COLOR) || defined(VECTOR)
TYPE add(TYPE in, TYPE amount){ return in + amount; }
TYPE add(TYPE in, float amount){ return in + amount; }
TYPE subtract(TYPE in, TYPE amount){ return in - amount; }
TYPE subtract(TYPE in, float amount){ return in - amount; }
TYPE invert(TYPE in, TYPE amount){ return amount - in; }
TYPE invert(TYPE in){ return 1 - in; }
TYPE divide(TYPE in1, TYPE amount){ return in1 / amount; }
TYPE divide(TYPE in1, float amount){ return in1 / amount; }
TYPE max(TYPE in, float amount){ return max(in, amount); }
TYPE min(TYPE in, float amount){ return min(in, amount); }
TYPE multiply(TYPE in1, TYPE amount){ return in1 * amount; }
TYPE multiply(TYPE in1, float amount){ return in1 * amount; }
TYPE fmod(TYPE in, TYPE amount){return fmod(in, amount); }
TYPE fmod(TYPE in, float amount){return fmod(in, amount); }

#endif

color smoothstep(color low, color high, color in){
    color out = in;
    out[0] = smoothstep(low[0], high[0], in[0]);
    out[1] = smoothstep(low[1], high[1], in[1]);
    out[2] = smoothstep(low[2], high[2], in[2]);
    return out;
}

color smoothstep(float low, float high, color in){
    color out = in;
    out[0] = smoothstep(low, high, in[0]);
    out[1] = smoothstep(low, high, in[1]);
    out[2] = smoothstep(low, high, in[2]);
    return out;
}

float remap(float in, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
      //remap in from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range
      float x = (in - inLow)/(inHigh-inLow);

      if (doClamp == 1)
           x = clamp(x, outLow, outHigh);

      return outLow + (outHigh-outLow) * x;
}

color remap(color in, color inLow, color inHigh, color outLow, color outHigh, int doClamp)
{
      //remap in from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range
      color x = (in - inLow)/(inHigh-inLow);

      if (doClamp == 1)
           x = clamp(x, outLow, outHigh);

      return outLow + (outHigh-outLow) * x;
}

color remap(color in, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
      //remap in from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range
      color x = (in - inLow)/(inHigh-inLow);

      if (doClamp == 1)
           x = clamp(x, outLow, outHigh);

      return outLow + (outHigh-outLow) * x;
}

float fgamma(float in, float g){ return sign(in) * pow(abs(in), g);}
color fgamma(color in, color g){ return sign(in) * pow(abs(in), g);}
color fgamma(color in, float g){ return sign(in) * pow(abs(in), g);}

color premult(color in, float alpha) { return in * alpha; }
color unpremult(color in, float alpha) { return in * (1/alpha); }

float contrast(float in, float amount, float pivot){
    float out = in - pivot;
    out *= amount;
    out += pivot;
    return out;
}

color contrast(color in, color amount, color pivot){
    color out = in - pivot;
    out *= amount;
    out += pivot;
    return out;
}

color contrast(color in, float amount, float pivot){
    color out = in - pivot;
    out *= amount;
    out += pivot;
    return out;
}
float exponent(float in, float amount){ return (in > 0) ? pow(in, amount) : in;}
color exponent(color in, color amount){ return (luminance(in) > 0) ? pow(in, amount) : in; }
color exponent(color in, float amount){ return (luminance(in) > 0) ? pow(in, amount) : in; }

vector vscale(vector in, vector amount, vector center){ return ((in - center)/amount); }
float mag(vector in){    return length(in); }

float fBm( point position, int octaves, float lacunarity, float diminish, string noisetype){

    float out = 0;
    float amp = 1.0;
    float i;
    point p = position;
    
    for (i = 0;  i < octaves;  i += 1) {
        out += amp * noise(noisetype, p);
        amp *= diminish;  
        p *= lacunarity;
    }
    return out;
}

color fBm( point position, int octaves, float lacunarity, float diminish, string noisetype){

    color out = 0;
    float amp = 1.0;
    float i;
    point p = position;
    
    for (i = 0;  i < octaves;  i += 1) {
        out += amp * (color)noise(noisetype, p);
        amp *= diminish;  
        p *= lacunarity;
    }
    return out;
}

color hueshift(color in, float amount){
    color hsv3 = transformc("rgb","hsv", in);
    hsv3[0] += amount;
    hsv3[0] = fmod(hsv3[0], 1.0);
    color out = transformc("hsv","rgb", hsv3);
    return out;
}

color saturate(color in, float amount){
    color hsv3 = transformc("rgb","hsv", in);
    hsv3[1] *= amount;
    color out =  transformc("hsv","rgb", hsv3);
    return out;
}

float dodge(float fg, float bg){ return bg / (1-fg); }
color dodge(color fg, color bg){ return bg / (1-fg); }
float screen(float fg, float bg){ return 1 - (1 - fg) * (1 - bg);}
color screen(color fg, color bg){ return 1 - (1 - fg) * (1 - bg);}
float difference(float fg, float bg){ return abs(fg - bg);}
color difference(color fg, color bg){ return abs(fg - bg);}

float overlay(float fg, float bg){
    if (fg < 0.5){
      return 2 * fg *bg;
    }
    return 1 - (1 - fg) * (1 - bg);
}

color overlay(color fg, color bg){
    if (luminance(fg) < 0.5){
      return 2 * fg * bg;
    }
    return color(1) - (color(1) - fg) * (color(1) - bg);
}

closure color add(closure color in1, closure color in2){
    return in1 + in2;
}

closure color mix(closure color in1, closure color in2, float mask){
    return (in1 * mask) +  (in2 * (1.0-mask));
}

