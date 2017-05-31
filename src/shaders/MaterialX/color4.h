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

color4 abs(color4 in)
{
    return color4 (abs(in.rgb),
                   abs(in.a)
                   );
}

color4 floor(color4 in)
{
    return color4 (floor(in.rgb),
                   floor(in.a)
                   );
}

color4 mix(color4 in1, color4 in2, float x )
{
    return color4 (mix(in1.rgb, in2.rgb, x),
                   mix(in1.a, in2.a, x)
                   );
}

color4 add(color4 in, color4 amount)
{
    return color4 (in.rgb + amount.rgb,
                   in.a + amount.a
                   );
}

color4 add(color4 in, float amount)
{
    return color4 (in.rgb + amount,
                   in.a + amount
                   );
}

color4 cluminance(color4 in, color lumacoeffs)
{
    return color4 ( color(cluminance(in.rgb, lumacoeffs)), in.a);
}

color4 subtract(color4 in, color4 amount)
{
    return color4 (in.rgb - amount.rgb,
                   in.a - amount.a
                   );
}

color4 subtract(color4 in, float amount)
{
    return color4 (in.rgb - amount,
                   in.a - amount
                   );
}

color4 smoothstep(color4 low, color4 high, color4 in)
{
    return color4 (smoothstep(low.rgb, high.rgb, in.rgb),
                   smoothstep(low.a, high.a, in.a)
                   );
}

color4 smoothstep(float low, float high, color4 in)
{
    return color4 (smoothstep(low, high, in.rgb),
                   smoothstep(low, high, in.a)
                   );
}

color4 remap(color4 in, color4 inLow, color4 inHigh, color4 outLow, color4 outHigh, int doClamp)
{
      //remap in from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range

      return color4 (remap(in.rgb, inLow.rgb, inHigh.rgb, outLow.rgb, outHigh.rgb, doClamp),
                     remap(in.a, inLow.a, inHigh.a, outLow.a, outHigh.a, doClamp)
                     );
}

color4 remap(color4 in, float inLow, float inHigh, float outLow, float outHigh, int doClamp)
{
      //remap in from [inLow, inHigh] to [outLow, outHigh], optionally clamping to the new range
      return color4 (remap(in.rgb, inLow, inHigh, outLow, outHigh, doClamp),
                     remap(in.a, inLow, inHigh, outLow, outHigh, doClamp)
                     );
}

color4 fgamma(color4 in, color4 g)
{
    return color4 (fgamma(in.rgb, g.rgb),
                   fgamma(in.a, g.a)
                   );
}

color4 fgamma(color4 in, float g)
{
    return color4 (fgamma(in.rgb, g),
                   fgamma(in.a, g)
                   );
}

color4 invert(color4 in, color4 amount)
{
    return color4 (amount.rgb - in.rgb,
                   amount.a - in.a
                   );
}

color4 invert(color4 in, float amount)
{
    return color4 (color(amount) - in.rgb,
                   amount - in.a
                   );
}

color4 invert(color4 in)
{
    return invert(in, 1.0);
}

color4 clamp(color4 in, color4 low, color4 high)
{
    return color4 (clamp(in.rgb, low.rgb, high.rgb),
                   clamp(in.a, low.a, high.a)
                   );
}

color4 clamp(color4 in, float low, float high)
{
    return color4 (clamp(in.rgb, low, high),
                   clamp(in.a, low, high)
                   );
}

color4 contrast(color4 in, color4 amount, color4 pivot)
{
    return color4 (contrast(in.rgb, amount.rgb, pivot.rgb),
                   contrast(in.a, amount.a, pivot.a)
                   );
}

color4 contrast(color4 in, float amount, float pivot)
{
    return color4 (contrast(in.rgb, amount, pivot),
                   contrast(in.a, amount, pivot)
                   );
}

color4 divide(color4 in, color4 amount)
{
    return color4 (in.rgb / amount.rgb,
                   in.a / amount.a
                   );
}

color4 divide(color4 in, float amount)
{
    return color4 (in.rgb / amount,
                   in.a / amount
                   );
}

color4 exponent(color4 in, color4 amount)
{
    return color4 (exponent(in.rgb, amount.rgb),
                   exponent(in.a, amount.a)
                   );
}


color4 exponent(color4 in, float amount)
{
    return color4 (exponent(in.rgb, amount),
                   exponent(in.a, amount)
                   );
}

color4 max(color4 in, color4 amount)
{
    return color4 (max(in.rgb, amount.rgb),
                   max(in.a, amount.a)
                   );
}

color4 max(color4 in, float amount)
{
    return color4 (max(in.rgb, amount),
                   max(in.a, amount)
                   );
}

color4 multiply(color4 in, color4 amount)
{
    return color4 (in.rgb * amount.rgb,
                   in.a * amount.a
                   );
}

color4 multiply(color4 in, float amount)
{
    return color4 (in.rgb * amount,
                   in.a * amount
                   );
}

color4 min(color4 in, color4 amount)
{
    return color4 (min(in.rgb, amount.rgb),
                    min(in.a, amount.a)
                    );
}

color4 min(color4 in, float amount)
{
    return color4 (min(in.rgb, amount),
                    min(in.a, amount)
                    );
}

color4 fmod(color4 in, color4 amount)
{
    return color4 (fmod(in.rgb, amount.rgb),
                    fmod(in.a, amount.a)
                    );
}

color4 fmod(color4 in, float amount)
{
    return color4 (fmod(in.rgb, amount),
                    fmod(in.a, amount)
                    );
}

color4 unpremult(color4 in)
{
   return color4 (unpremult(in.rgb, in.a),
                  in.a
                  );
}

color4 premult(color4 in)
{
   return color4 (premult(in.rgb, in.a),
                  in.a
                  );
}

color4 saturate(color4 in, float amount)
{
    color hsv3 = transformc("rgb","hsv", in.rgb);
    hsv3[1] *= amount;
    color4 out = {transformc("hsv","rgb", hsv3), in.a};
    return out;
}

color4 hueshift(color4 in, float amount)
{
    color hsv3 = transformc("rgb","hsv", in.rgb);
    hsv3[0] += amount;
    hsv3[0] = fmod(hsv3[0], 1.0);
    return color4 (transformc("hsv","rgb", hsv3), in.a);
}

color4 over(color4 fg, color4 bg)
{
    color4 out;
    float x = 1 - fg.a;
    out.rgb = fg.rgb * fg.a + bg.rgb * x;
    out.a = fg.a + bg.a * x;
    return out;
}

color4 cout(color4 fg, color4 bg)
{
    return color4 (fg.rgb * (1 - bg.a),
                   fg.a * (1 - bg.a)
                   );
}

color4 cmatte(color4 fg, color4 bg)
{
    return color4 ((fg.rgb * fg.a) + bg.rgb * (1 - fg.a),
                    fg.a + bg.a * (1 - fg.a)
                    );
}

color4 cmask(color4 fg, color4 bg)
{
    return color4 (bg.rgb * fg.a,
                    bg.a * fg.a
                    );
}

color4 cin(color4 fg, color4 bg)
{
    return color4 (fg.rgb * bg.a,
                    fg.a * bg.a
                    );
}

color4 disjointover(color4 fg, color4 bg)
{
    float summedAlpha = fg.a + bg.a;

    color4 out;
    if (summedAlpha <= 1){
        out.rgb = fg.rgb + bg.rgb;
    } else {
        float x = (1-fg.a) / bg.a;
        out.rgb = fg.rgb + bg.rgb * x;
    }
    out.a = min(summedAlpha, 1);
    return out;
}

color4 dodge(color4 fg, color4 bg)
{
    return color4 (dodge(fg.rgb, bg.rgb),
                   dodge(fg.a, bg.a)
                  );
}

color4 screen(color4 fg, color4 bg)
{
    return color4 (screen(fg.rgb, bg.rgb),
                   screen(fg.a, bg.a)
                  );
}

color4 overlay(color4 fg, color4 bg)
{
    return color4 (overlay(fg.rgb, bg.rgb),
                   overlay(fg.a, bg.a)
                  );
}

color4 difference(color4 fg, color4 bg)
{
    return color4 (difference(fg.rgb, bg.rgb),
                   difference(fg.a, bg.a)
                   );
}


