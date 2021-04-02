// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#ifndef __OSL_XMACRO_OPNAME
#    error must define __OSL_XMACRO_OPNAME to name of unary operation before including this header
#endif

#ifndef __OSL_XMACRO_OPERATOR
#    error must define __OSL_XMACRO_OPERATOR to operator to be tested  before including this header
#endif

#ifndef __OSL_XMACRO_VAL_TRANSFORM
#    define __OSL_XMACRO_VAL_TRANSFORM(val) val
#endif

#ifndef __OSL_XMACRO_VAL2_TRANSFORM
#    define __OSL_XMACRO_VAL2_TRANSFORM(val) val
#endif

#ifndef __OSL_XMACRO_OUT_TRANSFORM
#    define __OSL_XMACRO_OUT_TRANSFORM(out) out
#endif

#define __OSL_CONCAT_INDIRECT(A, B) A##B
#define __OSL_CONCAT(A, B)          __OSL_CONCAT_INDIRECT(A, B)

shader __OSL_CONCAT(test_, __OSL_XMACRO_OPNAME)(
    int numStripes = 0, int derivX = 0, int derivY = 0, float derivShift = 0,
    float derivScale = 1, output int out_int = 1, output float out_float = 1,
    output color out_color = 1, output point out_point = 1,
    output vector out_vector = 1, output normal out_normal = 1)
{
    float x_comp      = __OSL_XMACRO_VAL_TRANSFORM(P[0]);
    float y_comp      = __OSL_XMACRO_VAL_TRANSFORM(P[1]);
    float z_comp      = __OSL_XMACRO_VAL_TRANSFORM(P[2]);
    float float_val   = (x_comp + y_comp) * 0.5;
    color color_val   = color(x_comp, y_comp, z_comp);
    point point_val   = point(x_comp, y_comp, z_comp);
    vector vector_val = vector(x_comp, y_comp, z_comp);
    normal normal_val = normal(x_comp, y_comp, z_comp);

    float x_comp2      = __OSL_XMACRO_VAL2_TRANSFORM(P[0]);
    float y_comp2      = __OSL_XMACRO_VAL2_TRANSFORM(P[1]);
    float z_comp2      = __OSL_XMACRO_VAL2_TRANSFORM(P[2]);
    float float_val2   = (x_comp2 + y_comp2) * 0.5;
    color color_val2   = color(x_comp2, y_comp2, z_comp2);
    point point_val2   = point(x_comp2, y_comp2, z_comp2);
    vector vector_val2 = vector(x_comp2, y_comp2, z_comp2);
    normal normal_val2 = normal(x_comp2, y_comp2, z_comp2);

    // After "if" is supported in batching, uncomment conditional
    // if ((numStripes == 0) || ((numStripes != 0) && (int(P[0]*2*numStripes)%2)))
    {
        float_val  = float_val __OSL_XMACRO_OPERATOR float_val2;
        color_val  = color_val __OSL_XMACRO_OPERATOR color_val2;
        point_val  = point_val __OSL_XMACRO_OPERATOR point_val2;
        vector_val = vector_val __OSL_XMACRO_OPERATOR vector_val2;
        normal_val = normal_val __OSL_XMACRO_OPERATOR normal_val2;
    }

    if (derivX) {
        float_val  = Dx(float_val);
        color_val  = Dx(color_val);
        point_val  = Dx(point_val);
        vector_val = Dx(vector_val);
        normal_val = Dx(normal_val);
    }
    if (derivY) {
        float_val  = Dy(float_val);
        color_val  = Dy(color_val);
        point_val  = Dy(point_val);
        vector_val = Dy(vector_val);
        normal_val = Dy(normal_val);
    }
    if (derivX || derivY) {
        if (derivScale != 1) {
            float_val *= derivScale;
            color_val *= derivScale;
            point_val *= derivScale;
            vector_val *= derivScale;
            normal_val *= derivScale;
        }
        if (derivShift != 0) {
            float_val += derivShift;
            color_val += derivShift;
            point_val += derivShift;
            vector_val += derivShift;
            normal_val += derivShift;
        }
    }

    out_float  = __OSL_XMACRO_OUT_TRANSFORM(float_val);
    out_color  = __OSL_XMACRO_OUT_TRANSFORM(color_val);
    out_point  = __OSL_XMACRO_OUT_TRANSFORM(point_val);
    out_vector = __OSL_XMACRO_OUT_TRANSFORM(vector_val);
    out_normal = __OSL_XMACRO_OUT_TRANSFORM(normal_val);
}

#undef __OSL_XMACRO_OPNAME
#undef __OSL_XMACRO_VAL_TRANSFORM
