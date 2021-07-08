// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#ifndef __OSL_XMACRO_OPNAME
#    error must define __OSL_XMACRO_OPNAME to name of unary operation before including this header
#endif

#ifndef __OSL_XMACRO_OP
#   define __OSL_XMACRO_OP __OSL_XMACRO_OPNAME
#endif

#ifndef __OSL_XMACRO_IN_TRANSFORM
#    define __OSL_XMACRO_IN_TRANSFORM(...) __VA_ARGS__
#endif

#ifdef __OSL_XMACRO_UNIFORM_IN
#   define __OSL_XMACRO_IN_TRANSFORM_FLOAT(...) __OSL_XMACRO_IN_TRANSFORM(1.0/(2*raytype("camera")))
#else
#   define __OSL_XMACRO_IN_TRANSFORM_FLOAT(...) __OSL_XMACRO_IN_TRANSFORM(__VA_ARGS__)
#endif

#ifndef __OSL_XMACRO_STRIPE_TRANSFORM
#    define __OSL_XMACRO_STRIPE_TRANSFORM(...) (__VA_ARGS__)*0.5
#endif

#ifndef __OSL_XMACRO_OUT_TRANSFORM
#    define __OSL_XMACRO_OUT_TRANSFORM(...) __VA_ARGS__
#endif

#define __OSL_CONCAT_INDIRECT(A, B) A##B
#define __OSL_CONCAT(A, B)          __OSL_CONCAT_INDIRECT(A, B)

shader __OSL_CONCAT(test_, __OSL_XMACRO_OPNAME)(
    int numStripes = 0, int derivX = 0, int derivY = 0, float derivShift = 0,
    float derivScale = 1, output float out_float = 1)
{
    float float_in = __OSL_XMACRO_IN_TRANSFORM_FLOAT((P[0] + P[1])*0.5);

    // Exercise the op unmasked
    float float_val  = __OSL_XMACRO_OP(float_in);

    if ((numStripes != 0) && (int(P[0]*P[0]*P[1]*2*numStripes)%2)==0)
    {
        // Exercise the op masked
        float_val  = __OSL_XMACRO_OP(__OSL_XMACRO_STRIPE_TRANSFORM(float_in));
    }

    if (derivX) {
        float_val  = Dx(float_val);
    }
    if (derivY) {
        float_val  = Dy(float_val);
    }
    if (derivX || derivY) {
        if (derivScale != 1) {
            float_val *= derivScale;
        }
        if (derivShift != 0) {
            float_val += derivShift;
        }
    }

    out_float  = __OSL_XMACRO_OUT_TRANSFORM(float_val);
}
