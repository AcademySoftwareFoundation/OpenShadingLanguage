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
#   define __OSL_XMACRO_IN_TRANSFORM_X(...) __OSL_XMACRO_IN_TRANSFORM(1.0/(2*raytype("camera")))
#   define __OSL_XMACRO_IN_TRANSFORM_Y(...) __OSL_XMACRO_IN_TRANSFORM(1.0/(3*raytype("camera")))
#   define __OSL_XMACRO_IN_TRANSFORM_Z(...) __OSL_XMACRO_IN_TRANSFORM(1.0/(4*raytype("camera")))
#else
#   define __OSL_XMACRO_IN_TRANSFORM_X(...) __OSL_XMACRO_IN_TRANSFORM(__VA_ARGS__)
#   define __OSL_XMACRO_IN_TRANSFORM_Y(...) __OSL_XMACRO_IN_TRANSFORM(__VA_ARGS__)
#   define __OSL_XMACRO_IN_TRANSFORM_Z(...) __OSL_XMACRO_IN_TRANSFORM(__VA_ARGS__)
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
    int numStripes = 0, int derivSinX = 0, int derivSinY = 0,
    int derivCosX = 0, int derivCosY = 0,
    output float out_float = 1,
    output color out_color = 1, output point out_point = 1,
    output vector out_vector = 1, output normal out_normal = 1)
{
    float x_comp      = __OSL_XMACRO_IN_TRANSFORM_X(P[0]);
    float y_comp      = __OSL_XMACRO_IN_TRANSFORM_Y(P[1]);
    float z_comp      = __OSL_XMACRO_IN_TRANSFORM_Z(P[2]);

    float float_in   = (x_comp + y_comp) * 0.5;
    color color_in   = color(x_comp, y_comp, z_comp);
    point point_in   = point(x_comp, y_comp, z_comp);
    vector vector_in = vector(x_comp, y_comp, z_comp);
    normal normal_in = normal(x_comp, y_comp, z_comp);

    float float_sin = 0;
    color color_sin = color(0);
    point point_sin = point(0);
    vector vector_sin = vector(0);
    normal normal_sin = normal(0);

    float float_cos = 1;
    color color_cos = color(1);
    point point_cos = point(1);
    vector vector_cos = vector(1);
    normal normal_cos = normal(1);

    // Exercise the op unmasked
    sincos(float_in, float_sin, float_cos);
    sincos(color_in, color_sin, color_cos);
    sincos(point_in, point_sin, point_cos);
    sincos(vector_in, vector_sin, vector_cos);
    sincos(normal_in, normal_sin, normal_cos);

    // Exercise the op masked
    if ((numStripes != 0) && (int(P[0]*P[0]*P[1]*2*numStripes)%2)==0)
    {
        sincos(__OSL_XMACRO_STRIPE_TRANSFORM(float_in), float_sin, float_cos);
        sincos(__OSL_XMACRO_STRIPE_TRANSFORM(color_in), color_sin, color_cos);
        sincos(__OSL_XMACRO_STRIPE_TRANSFORM(point_in), point_sin, point_cos);
        sincos(__OSL_XMACRO_STRIPE_TRANSFORM(vector_in), vector_sin, vector_cos);
        sincos(__OSL_XMACRO_STRIPE_TRANSFORM(normal_in), normal_sin, normal_cos);
    }

    if (derivSinX && derivSinY) {
        error("only set 1 at a time, derivSinX or derivSinY");
    } else {
        if (derivSinX) {
            float_sin  = Dx(float_sin);
            color_sin  = Dx(color_sin);
            point_sin  = Dx(point_sin);
            vector_sin = Dx(vector_sin);
            normal_sin = Dx(normal_sin);
        }
        if (derivSinY) {
            float_sin  = Dy(float_sin);
            color_sin  = Dy(color_sin);
            point_sin  = Dy(point_sin);
            vector_sin = Dy(vector_sin);
            normal_sin = Dy(normal_sin);
        }
    }

    if (derivCosX && derivCosY) {
        error("only set 1 at a time, derivSinX or derivSinY");
    } else {
        if (derivCosX) {
            float_cos  = Dx(float_cos);
            color_cos  = Dx(color_cos);
            point_cos  = Dx(point_cos);
            vector_cos = Dx(vector_cos);
            normal_cos = Dx(normal_cos);
        }
        if (derivCosY) {
            float_cos  = Dy(float_cos);
            color_cos  = Dy(color_cos);
            point_cos  = Dy(point_cos);
            vector_cos = Dy(vector_cos);
            normal_cos = Dy(normal_cos);
        }
    }

    out_float  = __OSL_XMACRO_OUT_TRANSFORM((float_sin + float_cos)*0.5);
    out_color  = __OSL_XMACRO_OUT_TRANSFORM(color(color_sin[0], color_cos[1], (color_sin[1] + color_cos[0] *0.5)));
    out_point  = __OSL_XMACRO_OUT_TRANSFORM((point_sin + point_cos)*0.5);
    out_vector = __OSL_XMACRO_OUT_TRANSFORM((vector_sin + vector_cos)*0.5);
    out_normal = __OSL_XMACRO_OUT_TRANSFORM((normal_sin + normal_cos)*0.5);
}

