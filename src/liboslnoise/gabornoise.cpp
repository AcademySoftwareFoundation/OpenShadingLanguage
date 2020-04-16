// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#include <OSL/sfmath.h>

#include "gabornoise.h"

OSL_NAMESPACE_ENTER

namespace pvt {

// Use some helpers to avoid aliasing issues when mixing data member
// access vs. reinterpret casted array based access (non-conforming C++).
using sfm::Matrix33;
using sfm::make_matrix33_cols;

struct GaborParams {
    Vec3 omega;
    int anisotropic;
    bool do_filter;
    float a;
    float weight;
    Vec3 N;
    Matrix22 filter;
    Matrix33 local;
    float det_filter;
    float bandwidth;
    bool periodic;
    Vec3 period;
    float lambda;
    float sqrt_lambda_inv;
    float radius, radius2, radius3, radius_inv;

    OSL_HOSTDEVICE
    GaborParams (const NoiseParams &opt) :
        omega(opt.direction),  // anisotropy orientation
        anisotropic(opt.anisotropic),
        do_filter(opt.do_filter),
        weight(Gabor_Impulse_Weight),
        bandwidth(hostdevice::clamp(opt.bandwidth,0.01f,100.0f)),
        periodic(false)
    {
#if OSL_FAST_MATH
        float TWO_to_bandwidth = OIIO::fast_exp2(bandwidth);
#else
        float TWO_to_bandwidth = exp2f(bandwidth);
#endif

#ifndef __CUDA_ARCH__
        //static const float SQRT_PI_OVER_LN2 = sqrtf (M_PI / M_LN2);
        // To match GCC result of sqrtf (M_PI / M_LN2)
        static constexpr float SQRT_PI_OVER_LN2 = 2.128934e+00f;
#else
        #define SQRT_PI_OVER_LN2 (2.12893403886245235863f)
#endif
        a = Gabor_Frequency * ((TWO_to_bandwidth - 1.0) / (TWO_to_bandwidth + 1.0)) * SQRT_PI_OVER_LN2;
        // Calculate the maximum radius from which we consider the kernel
        // impulse centers -- derived from the threshold and bandwidth.
        radius = sqrtf(-logf(Gabor_Truncate) / float(M_PI)) / a;
        radius2 = radius * radius;
        radius3 = radius2 * radius;
        radius_inv = 1.0f / radius;
        // Lambda is the impulse density.
        float impulses = hostdevice::clamp (opt.impulses, 1.0f, 32.0f);
        lambda = impulses / (float(1.33333 * M_PI) * radius3);
        sqrt_lambda_inv = 1.0f / sqrtf(lambda);
    }
};

// Choose an omega and phi value for a particular gabor impulse,
// based on the user-selected noise mode.
//
// FIXME: x_c parameter seems unused. Is that correct?
static OSL_HOSTDEVICE void
gabor_sample (GaborParams &gp, const Vec3 &/*x_c*/, fast_rng &rng,
              Vec3 &omega, float &phi)
{
    // section 3.3, solid random-phase gabor noise
    if (gp.anisotropic == 1 /* anisotropic */) {
        omega = gp.omega;
    } else if (gp.anisotropic == 0 /* isotropic */) {
        float omega_t = float (M_TWO_PI) * rng();
        // float omega_p = acosf(lerp(-1.0f, 1.0f, rng()));
        float cos_omega_p = OIIO::lerp(-1.0f, 1.0f, rng());
        float sin_omega_p = sqrtf (std::max (0.0f, 1.0f - cos_omega_p*cos_omega_p));
        float sin_omega_t, cos_omega_t;
#if OSL_FAST_MATH
        OIIO::fast_sincos (omega_t, &sin_omega_t, &cos_omega_t);
#else
        OIIO::sincos (omega_t, &sin_omega_t, &cos_omega_t);
#endif
        omega = Vec3 (cos_omega_t*sin_omega_p, sin_omega_t*sin_omega_p, cos_omega_p).normalized();
    } else {
        // otherwise hybrid
        float omega_r = gp.omega.length();
        float omega_t =  float(M_TWO_PI) * rng();
        float sin_omega_t, cos_omega_t;
#if OSL_FAST_MATH
        OIIO::fast_sincos (omega_t, &sin_omega_t, &cos_omega_t);
#else
        OIIO::sincos (omega_t, &sin_omega_t, &cos_omega_t);
#endif
        omega = omega_r * Vec3(cos_omega_t, sin_omega_t, 0.0f);
    }
    phi = float(M_TWO_PI) * rng();
}


// Evaluate the summed contribution of all gabor impulses within the
// cell whose corner is c_i.  x_c_i is vector from x (the point
// we are trying to evaluate noise at) and c_i.
static OSL_HOSTDEVICE Dual2<float>
gabor_cell (GaborParams &gp, const Vec3 &c_i, const Dual2<Vec3> &x_c_i,
            int seed = 0)
{
    fast_rng rng (gp.periodic ? Vec3(wrap(c_i,gp.period)) : c_i, seed);
    int n_impulses = rng.poisson (gp.lambda * gp.radius3);
    Dual2<float> sum = 0;

    for (int i = 0; i < n_impulses; i++) {
        // OLD code: Vec3 x_i_c (rng(), rng(), rng());
        // Turned out that C++ spec says order of args are unspecified.
        // gcc appeared to do right-to-left, so to make sure our noise
        // function is locked down (and works identically for clang,
        // which evaluates left-to-right), we ask for the rng() calls
        // one at a time and match the way it looked before.
        float z_rng = rng(), y_rng = rng(), x_rng = rng();
        Vec3 x_i_c (x_rng, y_rng, z_rng);
        Dual2<Vec3> x_k_i = gp.radius * (x_c_i - x_i_c);        
        float phi_i;
        Vec3 omega_i;
        gabor_sample (gp, c_i, rng, omega_i, phi_i);
        if (x_k_i.val().length2() < gp.radius2) {
            if (! gp.do_filter) {
                // N.B. if determinant(gp.filter) is too small, we will
                // run into numerical problems.  But the filtering isn't
                // needed in that case anyway, so just don't filter.
                // This seems to only come up when the filter region is
                // tiny.
                sum += gabor_kernel (gp.weight, omega_i, phi_i, gp.a, x_k_i);  // 3D
            } else {
                // Transform the impulse's anisotropy into tangent space
                Vec3 omega_i_t;
                multMatrix (gp.local, omega_i, omega_i_t);

                // Slice to get a 2D kernel
                Dual2<float> d_i = -dot(gp.N, x_k_i);
                Dual2<float> w_i_t_s;
                Vec2 omega_i_t_s;
                Dual2<float> phi_i_t_s;
                slice_gabor_kernel_3d (d_i, gp.weight, gp.a,
                                       omega_i_t, phi_i,
                                       w_i_t_s, omega_i_t_s, phi_i_t_s);

                // Filter the 2D kernel
                Dual2<float> w_i_t_s_f;
                float a_i_t_s_f;
                Vec2 omega_i_t_s_f;
                Dual2<float> phi_i_t_s_f;
                filter_gabor_kernel_2d (gp.filter, w_i_t_s, gp.a, omega_i_t_s, phi_i_t_s, w_i_t_s_f, a_i_t_s_f, omega_i_t_s_f, phi_i_t_s_f);

                // Now evaluate the 2D filtered kernel
                Dual2<Vec3> xkit;
                multMatrix (gp.local, x_k_i, xkit);
                Dual2<Vec2> x_k_i_t = make_Vec2 (comp_x(xkit), comp_y(xkit));
                Dual2<float> gk = gabor_kernel (w_i_t_s_f, omega_i_t_s_f, phi_i_t_s_f, a_i_t_s_f, x_k_i_t); // 2D
                if (! OIIO::isfinite(gk.val())) {
                    // Numeric failure of the filtered version.  Fall
                    // back on the unfiltered.
                    gk = gabor_kernel (gp.weight, omega_i, phi_i, gp.a, x_k_i);  // 3D
                }
                sum += gk;
            }
        }
    }

    return sum;
}


// Sum the contributions of gabor impulses in all neighboring cells
// surrounding position x_g.
static OSL_HOSTDEVICE Dual2<float>
gabor_grid (GaborParams &gp, const Dual2<Vec3> &x_g, int seed=0)
{
    Vec3 floor_x_g (floor (x_g));  // Vec3 because floor has no derivs
    Dual2<Vec3> x_c = x_g - floor_x_g;
    Dual2<float> sum = 0;
    
    for (int k = -1; k <= 1; k++) {
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                Vec3 c (i,j,k);
                Vec3 c_i = floor_x_g + c;
                Dual2<Vec3> x_c_i = x_c - c;
                sum += gabor_cell (gp, c_i, x_c_i, seed);
            }
        }
    }
    return sum * gp.sqrt_lambda_inv;
}



inline OSL_HOSTDEVICE Dual2<float>
gabor_evaluate (GaborParams &gp, const Dual2<Vec3> &x, int seed=0)
{
    Dual2<Vec3> x_g = x * gp.radius_inv;
    return gabor_grid (gp, x_g, seed);
}




// set up the filter matrix
static OSL_HOSTDEVICE void
gabor_setup_filter (const Dual2<Vec3> &P, GaborParams &gp)
{
    // Make texture-space normal, tangent, bitangent
    Vec3 n, t, b;
    n = P.dx().cross (P.dy());  // normal to P
    if (n.dot(n) < 1.0e-6f) {  /* length of deriv < 1/1000 */
        // No way to do filter if we have no derivs, and no reason to
        // do it if it's too small to have any effect.
        gp.do_filter = false;
        return;   // we won't need anything else if filtering is off
    }
    make_orthonormals (n, t, b);

    // Rotations from tangent<->texture space
    Matrix33 Mtex_to_tan = make_matrix33_cols (t, b, n);  // M3_local
    Matrix33 Mscreen_to_tex = make_matrix33_cols (P.dx(), P.dy(), Vec3(0.0f,0.0f,0.0f));
    Matrix33 Mscreen_to_tan = Mscreen_to_tex * Mtex_to_tan;  // M3_scr_tan
    Matrix22 M_scr_tan (Mscreen_to_tan[0][0], Mscreen_to_tan[0][1],
                        Mscreen_to_tan[1][0], Mscreen_to_tan[1][1]);
    float sigma_f_scr = 0.5f;
    Matrix22 Sigma_f_scr (sigma_f_scr * sigma_f_scr, 0.0f,
                          0.0f, sigma_f_scr * sigma_f_scr);
    Matrix22 M_scr_tan_t = M_scr_tan.transposed();
    Matrix22 Sigma_f_tan = M_scr_tan_t * Sigma_f_scr * M_scr_tan;

    gp.N = n;
    gp.filter = Sigma_f_tan;
    gp.det_filter = determinant(Sigma_f_tan);
    gp.local  = Mtex_to_tan;
    if (gp.det_filter < 1.0e-18f) {
        gp.do_filter = false;
        // Turn off filtering when tiny values will lead to numerical
        // errors later if we filter.  Yes, it's kind of arbitrary.
    }
}



OSL_HOSTDEVICE Dual2<float>
gabor (const Dual2<float> &x, const NoiseParams *opt)
{
    // for now, just slice 3D
    return gabor (make_Vec3(x), opt);
}

OSL_HOSTDEVICE Dual2<float>
gabor (const Dual2<float> &x, const Dual2<float> &y, const NoiseParams *opt)
{
    // for now, just slice 3D
    return gabor (make_Vec3(x,y), opt);
}


OSL_HOSTDEVICE Dual2<float>
gabor (const Dual2<Vec3> &P, const NoiseParams *opt)
{
    OSL_DASSERT (opt);
    GaborParams gp (*opt);

    if (gp.do_filter)
        gabor_setup_filter (P, gp);

    Dual2<float> result = gabor_evaluate (gp, P);
    float gabor_variance = 1.0f / (4.0f*sqrtf(2.0) * (gp.a * gp.a * gp.a));
    float scale = 1.0f / (3.0f*sqrtf(gabor_variance));
    scale *= 0.5f;  // empirical -- make it fit in [-1..1]

    return result * scale;
}

OSL_HOSTDEVICE Dual2<Vec3>
gabor3 (const Dual2<float> &x, const NoiseParams *opt)
{
    // for now, just slice 3D
    return gabor3 (make_Vec3(x), opt);
}


OSL_HOSTDEVICE Dual2<Vec3>
gabor3 (const Dual2<float> &x, const Dual2<float> &y, const NoiseParams *opt)
{
    // for now, just slice 3D
    return gabor3 (make_Vec3(x,y), opt);
}


OSL_HOSTDEVICE Dual2<Vec3>
gabor3 (const Dual2<Vec3> &P, const NoiseParams *opt)
{
    OSL_DASSERT (opt);
    GaborParams gp (*opt);

    if (gp.do_filter)
        gabor_setup_filter (P, gp);

    Dual2<Vec3> result = make_Vec3 (gabor_evaluate (gp, P, 0),
                                    gabor_evaluate (gp, P, 1),
                                    gabor_evaluate (gp, P, 2));

    float gabor_variance = 1.0f / (4.0f*sqrtf(2.0) * (gp.a * gp.a * gp.a));
    float scale = 1.0f / (3.0f*sqrtf(gabor_variance));
    scale *= 0.5f;  // empirical -- make it fit in [-1..1]

    return result * scale;
}



OSL_HOSTDEVICE Dual2<float>
pgabor (const Dual2<float> &x, float xperiod, const NoiseParams *opt)
{
    // for now, just slice 3D
    return pgabor (make_Vec3(x), Vec3(xperiod,0.0f,0.0f), opt);
}



OSL_HOSTDEVICE Dual2<float>
pgabor (const Dual2<float> &x, const Dual2<float> &y,
        float xperiod, float yperiod, const NoiseParams *opt)
{
    // for now, just slice 3D
    return pgabor (make_Vec3(x,y), Vec3(xperiod,yperiod,0.0f), opt);
}



OSL_HOSTDEVICE Dual2<float>
pgabor (const Dual2<Vec3> &P, const Vec3 &Pperiod, const NoiseParams *opt)
{
    OSL_DASSERT (opt);
    GaborParams gp (*opt);

    gp.periodic = true;
    gp.period = Pperiod;

    if (gp.do_filter)
        gabor_setup_filter (P, gp);

    Dual2<float> result = gabor_evaluate (gp, P);
    float gabor_variance = 1.0f / (4.0f*sqrtf(2.0) * (gp.a * gp.a * gp.a));
    float scale = 1.0f / (3.0f*sqrtf(gabor_variance));
    scale *= 0.5f;  // empirical -- make it fit in [-1..1]

    return result * scale;
}


OSL_HOSTDEVICE Dual2<Vec3>
pgabor3 (const Dual2<float> &x, float xperiod, const NoiseParams *opt)
{
    // for now, just slice 3D
    return pgabor3 (make_Vec3(x), Vec3(xperiod,0.0f,0.0f), opt);
}

OSL_HOSTDEVICE Dual2<Vec3>
pgabor3 (const Dual2<float> &x, const Dual2<float> &y,
        float xperiod, float yperiod, const NoiseParams *opt)
{
    // for now, just slice 3D
    return pgabor3 (make_Vec3(x,y), Vec3(xperiod,yperiod,0.0f), opt);
}

OSL_HOSTDEVICE Dual2<Vec3>
pgabor3 (const Dual2<Vec3> &P, const Vec3 &Pperiod, const NoiseParams *opt)
{
    OSL_DASSERT (opt);
    GaborParams gp (*opt);

    gp.periodic = true;
    gp.period = Pperiod;

    if (gp.do_filter)
        gabor_setup_filter (P, gp);

    Dual2<Vec3> result = make_Vec3 (gabor_evaluate (gp, P, 0),
                                    gabor_evaluate (gp, P, 1),
                                    gabor_evaluate (gp, P, 2));

    float gabor_variance = 1.0f / (4.0f*sqrtf(2.0) * (gp.a * gp.a * gp.a));
    float scale = 1.0f / (3.0f*sqrtf(gabor_variance));
    scale *= 0.5f;  // empirical -- make it fit in [-1..1]

    return result * scale;
}


}; // namespace pvt

OSL_NAMESPACE_EXIT
