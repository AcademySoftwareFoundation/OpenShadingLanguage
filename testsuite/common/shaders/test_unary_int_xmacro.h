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
#   define __OSL_XMACRO_IN_TRANSFORM_INT(...) __OSL_XMACRO_IN_TRANSFORM(1.0/(2*raytype("camera")))
#else
#   define __OSL_XMACRO_IN_TRANSFORM_INT(...) __OSL_XMACRO_IN_TRANSFORM(__VA_ARGS__)
#endif

#ifndef __OSL_XMACRO_OUT_TRANSFORM
#    define __OSL_XMACRO_OUT_TRANSFORM(out) out
#endif

#ifndef __OSL_XMACRO_STRIPE_TRANSFORM
#    define __OSL_XMACRO_STRIPE_TRANSFORM(...) (__VA_ARGS__)/2
#endif

#ifndef __OSL_CONCAT
#    define __OSL_CONCAT_INDIRECT(A, B) A##B
#    define __OSL_CONCAT(A, B)          __OSL_CONCAT_INDIRECT(A, B)
#    define __OSL_CONCAT3(A, B, C)      __OSL_CONCAT(__OSL_CONCAT(A, B), C)
#endif

shader __OSL_CONCAT3(test_, __OSL_XMACRO_OPNAME, _int)(int numStripes     = 0,
                                                       output int out_int = 1, )
{
    int int_in = int(__OSL_XMACRO_IN_TRANSFORM_INT(((P[0] + P[1]) * 0.5)));

    // Exercise the op unmasked
    int int_val = __OSL_XMACRO_OP(int_in);

    // Exercise the op masked
    if ((numStripes != 0) && (int(P[0]*P[0]*P[1]*2*numStripes)%2 == 0))
    {
        int_val = __OSL_XMACRO_OP(int(__OSL_XMACRO_STRIPE_TRANSFORM(int_in)));
    }

    out_int = __OSL_XMACRO_OUT_TRANSFORM(int_val);
}

