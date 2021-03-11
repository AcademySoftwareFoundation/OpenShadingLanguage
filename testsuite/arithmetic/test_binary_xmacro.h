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

#define __OSL_CONCAT_INDIRECT(A, B) A##B
#define __OSL_CONCAT(A, B)          __OSL_CONCAT_INDIRECT(A, B)

shader __OSL_CONCAT(test_, __OSL_XMACRO_OPNAME)(
    int numStripes = 0, int derivX = 0, int derivY = 0, float derivShift = 0,
    float derivScale = 1, output float out_float = 1,
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

    float float_valB   = 1.0 - float_val;
    color color_valB   = color(1) - color_val;
    point point_valB   = point(1) - point_val;
    vector vector_valB = vector(1) - vector_val;
    normal normal_valB = normal(1) - normal_val;

    // After "if" is supported in batching, uncomment conditional
    // if ((numStripes == 0) || ((numStripes != 0) && (int(P[0]*2*numStripes)%2)))
    {
        float_val  = __OSL_XMACRO_OPNAME(float_val, float_valB);
        color_val  = __OSL_XMACRO_OPNAME(color_val, color_valB);
        point_val  = __OSL_XMACRO_OPNAME(point_val, point_valB);
        vector_val = __OSL_XMACRO_OPNAME(vector_val, vector_valB);
        normal_val = __OSL_XMACRO_OPNAME(normal_val, normal_valB);
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
