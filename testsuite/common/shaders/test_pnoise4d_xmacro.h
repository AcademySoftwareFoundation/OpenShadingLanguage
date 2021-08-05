// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#ifndef __OSL_XMACRO_SUFFIX
#    error must define __OSL_XMACRO_SUFFIX to create a unique testname before including this header
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
#elif defined(__OSL_XMACRO_UNIFORM_IN2)
#   define __OSL_XMACRO_IN_TRANSFORM_X2(...) __OSL_XMACRO_IN_TRANSFORM2(1.5/(2*raytype("camera")))
#   define __OSL_XMACRO_IN_TRANSFORM_Y2(...) __OSL_XMACRO_IN_TRANSFORM2(1.5/(3*raytype("camera")))
#else
#   define __OSL_XMACRO_IN_TRANSFORM_X2(...) __OSL_XMACRO_IN_TRANSFORM2(__VA_ARGS__)
#   define __OSL_XMACRO_IN_TRANSFORM_Y2(...) __OSL_XMACRO_IN_TRANSFORM2(__VA_ARGS__)
#endif

#ifndef __OSL_XMACRO_PERIOD_TRANSFORM
#    define __OSL_XMACRO_PERIOD_TRANSFORM(...) int(64*(__VA_ARGS__))
#endif

#ifdef __OSL_XMACRO_CONSTANT_PERIOD
#   define __OSL_XMACRO_PERIOD_TRANSFORM_X(...) __OSL_XMACRO_PERIOD_TRANSFORM(0.5)
#   define __OSL_XMACRO_PERIOD_TRANSFORM_Y(...) __OSL_XMACRO_PERIOD_TRANSFORM(0.75)
#   define __OSL_XMACRO_PERIOD_TRANSFORM_Z(...) __OSL_XMACRO_PERIOD_TRANSFORM(0.25)
#elif defined(__OSL_XMACRO_UNIFORM_PERIOD)
#   define __OSL_XMACRO_PERIOD_TRANSFORM_X(...) __OSL_XMACRO_PERIOD_TRANSFORM(1.0/(2*raytype("camera")))
#   define __OSL_XMACRO_PERIOD_TRANSFORM_Y(...) __OSL_XMACRO_PERIOD_TRANSFORM(1.0/(3*raytype("camera")))
#   define __OSL_XMACRO_PERIOD_TRANSFORM_Z(...) __OSL_XMACRO_PERIOD_TRANSFORM(1.0/(4*raytype("camera")))
#else
#   define __OSL_XMACRO_PERIOD_TRANSFORM_X(...) __OSL_XMACRO_PERIOD_TRANSFORM(__VA_ARGS__)
#   define __OSL_XMACRO_PERIOD_TRANSFORM_Y(...) __OSL_XMACRO_PERIOD_TRANSFORM(__VA_ARGS__)
#   define __OSL_XMACRO_PERIOD_TRANSFORM_Z(...) __OSL_XMACRO_PERIOD_TRANSFORM(__VA_ARGS__)
#endif

#ifndef __OSL_XMACRO_PERIOD_TRANSFORM2
#    define __OSL_XMACRO_PERIOD_TRANSFORM2(...) (64 - int(64*(__VA_ARGS__)))
#endif

#ifdef __OSL_XMACRO_CONSTANT_PERIOD2
#   define __OSL_XMACRO_PERIOD_TRANSFORM_X2(...) __OSL_XMACRO_PERIOD_TRANSFORM2(0.75)
#elif defined(__OSL_XMACRO_UNIFORM_PERIOD)
#   define __OSL_XMACRO_PERIOD_TRANSFORM_X2(...) __OSL_XMACRO_PERIOD_TRANSFORM2(1.5/(2*raytype("camera")))
#else
#   define __OSL_XMACRO_PERIOD_TRANSFORM_X2(...) __OSL_XMACRO_PERIOD_TRANSFORM2(__VA_ARGS__)
#endif

#ifndef __OSL_XMACRO_STRIPE_TRANSFORM
#    define __OSL_XMACRO_STRIPE_TRANSFORM(...) (__VA_ARGS__)*0.5
#endif

#ifndef __OSL_XMACRO_STRIPE_TRANSFORM2
#    define __OSL_XMACRO_STRIPE_TRANSFORM2(...) (__VA_ARGS__)*2
#endif

#ifndef __OSL_XMACRO_OUT_TRANSFORM
#    define __OSL_XMACRO_OUT_TRANSFORM(...) __VA_ARGS__
#endif

#define __OSL_CONCAT_INDIRECT(A, B) A##B
#define __OSL_CONCAT(A, B)          __OSL_CONCAT_INDIRECT(A, B)

void do_pnoise4d(
    string noisetype,
    int numStripes, int derivs, output float out_float,
    output color out_color)
{
    float x_comp      = __OSL_XMACRO_IN_TRANSFORM_X(P[0]);
    float y_comp      = __OSL_XMACRO_IN_TRANSFORM_Y(P[1]);
    float z_comp      = __OSL_XMACRO_IN_TRANSFORM_Z((P[0]+P[1])*0.5);

    float x_comp2     = __OSL_XMACRO_IN_TRANSFORM_X2(max(0,1 - P[1]));
    float y_comp2     = __OSL_XMACRO_IN_TRANSFORM_Y2(max(0,1 - P[0]));

    point point_in   = point(x_comp, y_comp, z_comp);
    float float_in2   = (x_comp2 + y_comp2) * 0.5;

    float px_comp      = __OSL_XMACRO_PERIOD_TRANSFORM_X(P[0]);
    float py_comp      = __OSL_XMACRO_PERIOD_TRANSFORM_Y(P[1]);
    float pz_comp      = __OSL_XMACRO_PERIOD_TRANSFORM_Z((P[0]+P[1])*0.5);

    point point_period   = point(px_comp, py_comp, pz_comp);
    float float_period2 = __OSL_XMACRO_PERIOD_TRANSFORM_X2(v);

    // Exercise the op unmasked
    float float_val  = pnoise(noisetype, point_in, float_in2, point_period, float_period2);
    color color_val  = pnoise(noisetype, point_in, float_in2, point_period, float_period2);

    // Exercise the op masked
    if ((numStripes != 0) && (int(P[0]*P[0]*P[1]*(2*numStripes))%2==0))
    {
        float_val  = pnoise(noisetype, __OSL_XMACRO_STRIPE_TRANSFORM(point_in), __OSL_XMACRO_STRIPE_TRANSFORM2(float_in2), point_period, float_period2);
        color_val  = pnoise(noisetype, __OSL_XMACRO_STRIPE_TRANSFORM(point_in), __OSL_XMACRO_STRIPE_TRANSFORM2(float_in2), point_period, float_period2);
    }

    if (derivs) {
        if ((v < 0.24) || (v > 0.51 && v < 0.74)) {
            float_val  = Dx(float_val);
            color_val  = Dx(color_val);
        } else if ((v > 0.26 && v < 0.49) || (v > 0.76)) {
            float_val  = Dy(float_val);
            color_val  = Dy(color_val);
        }
    }

    if (noisetype == "perlin" ||
        noisetype == "gabor")
    {
        // Bring results in [-1,1] back into [0-1]
        float_val = (float_val + 1)*0.5;
        color_val = (color_val + 1)*0.5;
    }

    out_float  = __OSL_XMACRO_OUT_TRANSFORM(float_val);
    out_color  = __OSL_XMACRO_OUT_TRANSFORM(color_val);
}

shader __OSL_CONCAT(test_pnoise4d_, __OSL_XMACRO_SUFFIX)(
    int numStripes = 0, int derivs = 0, output float out_float = 1,
    output color out_color = 1)
{
    if (v < 0.49) {
        // float noise in 1,2,3,4 dimensions
        if (u < 0.24)
            do_pnoise4d("perlin", numStripes, derivs, out_float, out_color);
        else if (u > 0.26 && u < 0.49)
            do_pnoise4d("snoise", numStripes, derivs, out_float, out_color);
        else if (u > 0.51 && u < 0.74)
            do_pnoise4d("uperlin", numStripes, derivs, out_float, out_color);
        else if (u > 0.76)
            do_pnoise4d("noise", numStripes, derivs, out_float, out_color);
    } else if (v > 0.51) {
        // color noise in 1,2,3,4 dimensions
        if (u < 0.24)
            do_pnoise4d("cell", numStripes, derivs, out_float, out_color);
        else if (u > 0.26 && u < 0.49)
            do_pnoise4d("hash", numStripes, derivs, out_float, out_color);
        else if (u > 0.51 && u < 0.74)
            do_pnoise4d("gabor", numStripes, derivs, out_float, out_color);
    }
}
