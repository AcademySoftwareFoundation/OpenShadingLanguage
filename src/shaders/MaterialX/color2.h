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

color2 abs(color2 in)
{
    return color2 (abs(in.r),
                    abs(in.g)
                    );
}

color2 floor(color2 in)
{
    return color2 (floor(in.r),
                    floor(in.g)
                    );
}

color2 mix(color2 in1, color2 in2, float x )
{
    return color2 (mix(in1.r, in2.r, x),
                    mix(in1.g, in2.g, x)
                    );
}

color2 add(color2 in, color2 amount)
{
    return color2 (in.r + amount.r,
                    in.g + amount.g
                    );
}

color2 add(color2 in, float amount)
{
    return color2 (in.r + amount,
                    in.g + amount
                    );
}

color2 subtract(color2 in, color2 amount)
{
    return color2 (in.r - amount.r,
                    in.g - amount.g
                    );
}    

color2 subtract(color2 in, float amount)
{
    return color2 (in.r - amount,
                    in.g - amount
                    );
}

color2 smoothstep(color2 low, color2 high, color2 in)
{
    return color2 (smoothstep(low.r, high.r, in.r),
                    smoothstep(low.g, high.g, in.g)
                    );
}

color2 smoothstep(float low, float high, color2 in)
{
    return color2 (smoothstep(low, high, in.r),
                    smoothstep(low, high, in.g)
                    );
}

color2 remap(color2 in, color2 inLow, color2 inHigh, color2 outLow, color2 outHigh, int doClamp)
{
    //remap in from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range
    return color2 (remap(in.r, inLow.r, inHigh.r, outLow.r, outHigh.r, doClamp),
                    remap(in.g, inLow.g, inHigh.g, outLow.g, outHigh.g, doClamp)
                    );
}

color2 remap(color2 in, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
    //remap in from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range
    return color2 (remap(in.r, inLow, inHigh, outLow, outHigh, doClamp),
                    remap(in.g, inLow, inHigh, outLow, outHigh, doClamp)
                    );
}

color2 fgamma(color2 in, color2 g){
    return color2 (fgamma(in.r, g.r),
                    fgamma(in.g, g.g)
                    );
}

color2 fgamma(color2 in, float g)
{
    return color2 (fgamma(in.r, g),
                    fgamma(in.g, g)
                    );
}

color2 invert(color2 in, color2 amount)
{
    return color2 (amount.r - in.r,
                    amount.g - in.g
                    );

}

color2 invert(color2 in, float amount)
{
    return color2 (amount - in.r,
                    amount - in.g
                    );
}

color2 invert(color2 in)
{
    return invert(in, 1.0);
}

color2 clamp(color2 in, color2 low, color2 high)
{
    return color2 (clamp(in.r, low.r, high.r),
                    clamp(in.g, low.g, high.g)
                    );
}

color2 clamp(color2 in, float low, float high)
{
    return color2 (clamp(in.r, low, high),
                    clamp(in.g, low, high)
                    );
}

color2 contrast(color2 in, color2 amount, color2 pivot)
{
    return color2 (contrast(in.r, amount.r, pivot.r),
                    contrast(in.g, amount.g, pivot.g)
                    );
}

color2 contrast(color2 in, float amount, float pivot)
{
    return color2 (contrast(in.r, amount, pivot),
                    contrast(in.g, amount, pivot)
                    );
}

color2 divide(color2 in, color2 amount)
{
    return color2 (in.r / amount.r,
                    in.g / amount.g
                    );
}

color2 divide(color2 in, float amount)
{
    return color2 (in.r / amount,
                    in.g / amount
                    );
}

color2 exponent(color2 in, color2 amount)
{
    return color2 (exponent(in.r, amount.r),
                    exponent(in.g, amount.g)
                    );

}

color2 exponent(color2 in, float amount)
{
    return color2 (exponent(in.r, amount),
                    exponent(in.g, amount)
                    );
}

color2 max(color2 in, color2 amount)
{
    return color2 (max(in.r, amount.r),
                    max(in.g, amount.g)
                    );
}

color2 max(color2 in, float amount)
{
    return color2 (max(in.r, amount),
                    max(in.g, amount)
                    );
}

color2 multiply(color2 in, color2 amount)
{
    return color2 (in.r * amount.r,
                    in.g * amount.g
                    );
}

color2 multiply(color2 in, float amount)
{
    return color2 (in.r * amount,
                    in.g * amount
                    );
}

color2 min(color2 in, color2 amount)
{
    return color2 (min(in.r, amount.r),
                    min(in.g, amount.g)
                    );
}

color2 min(color2 in, float amount)
{
    return color2 (min(in.r, amount),
                    min(in.g, amount)
                    );
}

color2 fmod(color2 in, color2 amount)
{
    return color2 (fmod(in.r, amount.r),
                    fmod(in.g, amount.g)
                    );
}

color2 fmod(color2 in, float amount)
{
    return color2 (fmod(in.r, amount),
                    fmod(in.g, amount)
                    );
}

color2 unpremult(color2 in)
{
    return color2 (in.r / in.g, in.g);
}

color2 premult(color2 in)
{
    return color2 (in.r * in.g, in.g);
}

color2 cout(color2 fg, color2 bg)
{
    return color2 (fg.r * (1 - bg.g),
                   fg.g * (1 - bg.g)
                   );
}

color2 over(color2 fg, color2 bg)
{
    return color2 (fg.r + bg.r * (1 - fg.g),
                    fg.g + bg.g * (1 - fg.g)
                    );
}

color2 cmatte(color2 fg, color2 bg)
{
    return color2 ((fg.r * fg.g) + bg.r * (1 - fg.g),
                    fg.g + bg.g * (1 - fg.g)
                    );
}

color2 cmask(color2 fg, color2 bg)
{
    return color2 (bg.r * fg.g,
                    bg.g * fg.g
                    );
}

color2 cin(color2 fg, color2 bg)
{
    return color2 (fg.r * bg.g,
                    fg.g * bg.g
                    );
}

color2 disjointover(color2 fg, color2 bg)
{
    float summedAlpha = fg.g + bg.g;
    color2 out;
    if (summedAlpha <= 1){
        out.r = fg.r + bg.r;
    } else {
        out.r = fg.r + ((bg.r * (1-fg.g)) / bg.g);
    }
    out.g = min(summedAlpha, 1);
    return out;
}

color2 dodge(color2 fg, color2 bg)
{
    return color2 (dodge(fg.r, bg.r),
                   dodge(fg.g, bg.g)
                   );
}

color2 screen(color2 fg, color2 bg)
{
    return color2 (screen(fg.r, bg.r),
                   screen(fg.g, bg.g)
                   );
}

color2 overlay(color2 fg, color2 bg)
{
    return color2 (overlay(fg.r, bg.r),
                   overlay(fg.g, bg.g)
                   );
}

color2 difference(color2 fg, color2 bg)
{
    return color2 (difference(fg.r, bg.r),
                   difference(fg.g, bg.g)
                   );
}

