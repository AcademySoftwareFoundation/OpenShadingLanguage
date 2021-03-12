// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#ifndef __OSL_XMACRO_OPNAME
#    error must define __OSL_XMACRO_OPNAME to name of unary operation before including this header
#endif

#ifndef __OSL_XMACRO_VAL_TRANSFORM
#    define __OSL_XMACRO_VAL_TRANSFORM(val) val
#endif

#ifndef __OSL_XMACRO_OUT_TRANSFORM
#    define __OSL_XMACRO_OUT_TRANSFORM(out) out
#endif

#ifndef __OSL_CONCAT
#    define __OSL_CONCAT_INDIRECT(A, B) A##B
#    define __OSL_CONCAT(A, B)          __OSL_CONCAT_INDIRECT(A, B)
#    define __OSL_CONCAT3(A, B, C)      __OSL_CONCAT(__OSL_CONCAT(A, B), C)
#endif

shader __OSL_CONCAT3(test_, __OSL_XMACRO_OPNAME, _int)(int numStripes     = 0,
                                                       output int out_int = 1, )
{
    int int_val = int(__OSL_XMACRO_VAL_TRANSFORM(((P[0] + P[1]) * 0.5)));

    // After "if" is supported in batching, uncomment conditional
    // if ((numStripes == 0) || ((numStripes != 0) && (int(P[0]*2*numStripes)%2)))
    {
        int_val = __OSL_XMACRO_OPNAME(int_val);
    }

    out_int = __OSL_XMACRO_OUT_TRANSFORM(int_val);
}

#undef __OSL_XMACRO_OPNAME
#undef __OSL_XMACRO_VAL_TRANSFORM
#undef __OSL_XMACRO_OUT_TRANSFORM
