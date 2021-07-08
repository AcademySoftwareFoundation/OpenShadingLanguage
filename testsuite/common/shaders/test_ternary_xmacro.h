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

#ifdef __OSL_XMACRO_CONSTANT_IN
#   define __OSL_XMACRO_IN_TRANSFORM_X(...) __OSL_XMACRO_IN_TRANSFORM(0.5)
#   define __OSL_XMACRO_IN_TRANSFORM_Y(...) __OSL_XMACRO_IN_TRANSFORM(0.75)
#   define __OSL_XMACRO_IN_TRANSFORM_Z(...) __OSL_XMACRO_IN_TRANSFORM(0.25)
#elif defined(__OSL_XMACRO_UNIFORM_IN)
#   define __OSL_XMACRO_IN_TRANSFORM_X(...) __OSL_XMACRO_IN_TRANSFORM(1.0/(2*raytype("camera")))
#   define __OSL_XMACRO_IN_TRANSFORM_Y(...) __OSL_XMACRO_IN_TRANSFORM(1.0/(3*raytype("camera")))
#   define __OSL_XMACRO_IN_TRANSFORM_Z(...) __OSL_XMACRO_IN_TRANSFORM(1.0/(4*raytype("camera")))
#else
#   define __OSL_XMACRO_IN_TRANSFORM_X(...) __OSL_XMACRO_IN_TRANSFORM(__VA_ARGS__)
#   define __OSL_XMACRO_IN_TRANSFORM_Y(...) __OSL_XMACRO_IN_TRANSFORM(__VA_ARGS__)
#   define __OSL_XMACRO_IN_TRANSFORM_Z(...) __OSL_XMACRO_IN_TRANSFORM(__VA_ARGS__)
#endif

#ifndef __OSL_XMACRO_IN_TRANSFORM2
#    define __OSL_XMACRO_IN_TRANSFORM2(...) __VA_ARGS__
#endif

#ifdef __OSL_XMACRO_CONSTANT_IN2
#   define __OSL_XMACRO_IN_TRANSFORM_X2(...) __OSL_XMACRO_IN_TRANSFORM2(0.75)
#   define __OSL_XMACRO_IN_TRANSFORM_Y2(...) __OSL_XMACRO_IN_TRANSFORM2(0.95)
#   define __OSL_XMACRO_IN_TRANSFORM_Z2(...) __OSL_XMACRO_IN_TRANSFORM2(0.45)
#elif defined(__OSL_XMACRO_UNIFORM_IN2)
#   define __OSL_XMACRO_IN_TRANSFORM_X2(...) __OSL_XMACRO_IN_TRANSFORM2(1.5/(2*raytype("camera")))
#   define __OSL_XMACRO_IN_TRANSFORM_Y2(...) __OSL_XMACRO_IN_TRANSFORM2(1.5/(3*raytype("camera")))
#   define __OSL_XMACRO_IN_TRANSFORM_Z2(...) __OSL_XMACRO_IN_TRANSFORM2(1.5/(4*raytype("camera")))
#else
#   define __OSL_XMACRO_IN_TRANSFORM_X2(...) __OSL_XMACRO_IN_TRANSFORM2(__VA_ARGS__)
#   define __OSL_XMACRO_IN_TRANSFORM_Y2(...) __OSL_XMACRO_IN_TRANSFORM2(__VA_ARGS__)
#   define __OSL_XMACRO_IN_TRANSFORM_Z2(...) __OSL_XMACRO_IN_TRANSFORM2(__VA_ARGS__)
#endif

#ifndef __OSL_XMACRO_IN_TRANSFORM3
#    define __OSL_XMACRO_IN_TRANSFORM3(...) __VA_ARGS__
#endif

#ifdef __OSL_XMACRO_CONSTANT_IN3
#   define __OSL_XMACRO_IN_TRANSFORM_X3(...) __OSL_XMACRO_IN_TRANSFORM3(0.65)
#   define __OSL_XMACRO_IN_TRANSFORM_Y3(...) __OSL_XMACRO_IN_TRANSFORM3(0.85)
#   define __OSL_XMACRO_IN_TRANSFORM_Z3(...) __OSL_XMACRO_IN_TRANSFORM3(0.35)
#elif defined(__OSL_XMACRO_UNIFORM_IN3)
#   define __OSL_XMACRO_IN_TRANSFORM_X3(...) __OSL_XMACRO_IN_TRANSFORM3(1.25/(2*raytype("camera")))
#   define __OSL_XMACRO_IN_TRANSFORM_Y3(...) __OSL_XMACRO_IN_TRANSFORM3(1.25/(3*raytype("camera")))
#   define __OSL_XMACRO_IN_TRANSFORM_Z3(...) __OSL_XMACRO_IN_TRANSFORM3(1.25/(4*raytype("camera")))
#else
#   define __OSL_XMACRO_IN_TRANSFORM_X3(...) __OSL_XMACRO_IN_TRANSFORM3(__VA_ARGS__)
#   define __OSL_XMACRO_IN_TRANSFORM_Y3(...) __OSL_XMACRO_IN_TRANSFORM3(__VA_ARGS__)
#   define __OSL_XMACRO_IN_TRANSFORM_Z3(...) __OSL_XMACRO_IN_TRANSFORM3(__VA_ARGS__)
#endif

#ifndef __OSL_XMACRO_STRIPE_TRANSFORM
#    define __OSL_XMACRO_STRIPE_TRANSFORM(...) (__VA_ARGS__)*0.5
#endif

#ifndef __OSL_XMACRO_STRIPE_TRANSFORM2
#    define __OSL_XMACRO_STRIPE_TRANSFORM2(...) (__VA_ARGS__)*2
#endif

#ifndef __OSL_XMACRO_STRIPE_TRANSFORM3
#    define __OSL_XMACRO_STRIPE_TRANSFORM3(...) (__VA_ARGS__)*0.75
#endif


#ifndef __OSL_XMACRO_OUT_TRANSFORM
#    define __OSL_XMACRO_OUT_TRANSFORM(...) __VA_ARGS__
#endif

#define __OSL_CONCAT_INDIRECT(A, B) A##B
#define __OSL_CONCAT(A, B)          __OSL_CONCAT_INDIRECT(A, B)

shader __OSL_CONCAT(test_, __OSL_XMACRO_OPNAME)(
    int numStripes = 0, int derivX = 0, int derivY = 0, float derivShift = 0,
    float derivScale = 1, output float out_float = 1,
    output color out_color = 1, output point out_point = 1,
    output vector out_vector = 1, output normal out_normal = 1)
{
    float x_comp      = __OSL_XMACRO_IN_TRANSFORM_X(P[0]);
    float y_comp      = __OSL_XMACRO_IN_TRANSFORM_Y(P[1]);
    float z_comp      = __OSL_XMACRO_IN_TRANSFORM_Z(P[2]);

    float x_comp2     = __OSL_XMACRO_IN_TRANSFORM_X2(max(0,1 - P[1]));
    float y_comp2     = __OSL_XMACRO_IN_TRANSFORM_Y2(max(0,1 - P[0]));
    float z_comp2     = __OSL_XMACRO_IN_TRANSFORM_Z2(max(0,1 - P[2])/2);

    float x_comp3     = __OSL_XMACRO_IN_TRANSFORM_X3(max(0,(1 + P[0] - P[1])*0.5));
    float y_comp3     = __OSL_XMACRO_IN_TRANSFORM_Y3(max(0,(1 + P[1] - P[0])*0.5));
    float z_comp3     = __OSL_XMACRO_IN_TRANSFORM_Z3(max(0,(P[0] + P[1] + P[2]))/3);

    float float_in   = (x_comp + y_comp) * 0.5;
    color color_in   = color(x_comp, y_comp, z_comp);
    point point_in   = point(x_comp, y_comp, z_comp);
    vector vector_in = vector(x_comp, y_comp, z_comp);
    normal normal_in = normal(x_comp, y_comp, z_comp);

    float float_in2   = (x_comp2 + y_comp2) * 0.5;
    color color_in2   = color(x_comp2, y_comp2, z_comp2);
    point point_in2   = point(x_comp2, y_comp2, z_comp2);
    vector vector_in2 = vector(x_comp2, y_comp2, z_comp2);
    normal normal_in2 = normal(x_comp2, y_comp2, z_comp2);

    float float_in3   = (x_comp3 + y_comp3) * 0.5;
    color color_in3   = color(x_comp3, y_comp3, z_comp3);
    point point_in3   = point(x_comp3, y_comp3, z_comp3);
    vector vector_in3 = vector(x_comp3, y_comp3, z_comp3);
    normal normal_in3 = normal(x_comp3, y_comp3, z_comp3);

    // Exercise the op unmasked
    float float_val  = __OSL_XMACRO_OP(float_in, float_in2, float_in3);
    color color_val  = __OSL_XMACRO_OP(color_in, color_in2, color_in3);
    point point_val  = __OSL_XMACRO_OP(point_in, point_in2, point_in3);
    vector vector_val = __OSL_XMACRO_OP(vector_in, vector_in2, vector_in3);
    normal normal_val = __OSL_XMACRO_OP(normal_in, normal_in2, normal_in3);

    // Exercise the op masked
    if ((numStripes != 0) && (int(P[0]*P[0]*P[1]*2*numStripes)%2)==0)
    {
        float_val  = __OSL_XMACRO_OP(__OSL_XMACRO_STRIPE_TRANSFORM(float_in), __OSL_XMACRO_STRIPE_TRANSFORM2(float_in2), __OSL_XMACRO_STRIPE_TRANSFORM3(float_in3));
        color_val  = __OSL_XMACRO_OP(__OSL_XMACRO_STRIPE_TRANSFORM(color_in), __OSL_XMACRO_STRIPE_TRANSFORM2(color_in2), __OSL_XMACRO_STRIPE_TRANSFORM3(color_in3));
        point_val  = __OSL_XMACRO_OP(__OSL_XMACRO_STRIPE_TRANSFORM(point_in), __OSL_XMACRO_STRIPE_TRANSFORM2(point_in2), __OSL_XMACRO_STRIPE_TRANSFORM3(point_in3));
        vector_val = __OSL_XMACRO_OP(__OSL_XMACRO_STRIPE_TRANSFORM(vector_in), __OSL_XMACRO_STRIPE_TRANSFORM2(vector_in2), __OSL_XMACRO_STRIPE_TRANSFORM3(vector_in3));
        normal_val = __OSL_XMACRO_OP(__OSL_XMACRO_STRIPE_TRANSFORM(normal_in), __OSL_XMACRO_STRIPE_TRANSFORM2(normal_in2), __OSL_XMACRO_STRIPE_TRANSFORM3(normal_in3));
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
