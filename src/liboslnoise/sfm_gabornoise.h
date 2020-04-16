// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#include <OSL/sfmath.h>

#include "gabornoise.h"


OSL_NAMESPACE_ENTER

namespace __OSL_WIDE_PVT {

// non SIMD version, should be scalar code meant to be used
// inside SIMD loops
// SIMD FRIENDLY MATH
namespace sfm
{

// Very fast random number generator based on [Borosh & Niederreiter 1983]
// linear congruential generator.
class fast_rng {
public:
    // seed based on the cell containing P
    OSL_FORCEINLINE fast_rng (const Vec3 &p, int seed=0) {
        // Use guts of cellnoise
        m_seed = inthash(unsigned(OIIO::ifloor(p.x)),
                         unsigned(OIIO::ifloor(p.y)),
                         unsigned(OIIO::ifloor(p.z)),
                         unsigned(seed));
        if (! m_seed)
            m_seed = 1;
    }

    OSL_FORCEINLINE fast_rng(const fast_rng &other)
    : m_seed(other.m_seed)
    {}

    // Return uniform on [0,1)
    OSL_FORCEINLINE float operator() () {
        //return (m_seed *= 3039177861u) / float(UINT_MAX);
        return (m_seed *= 3039177861u)*(1.0f/float(UINT_MAX));
    }

    // Return poisson distribution with the given mean
    OSL_FORCEINLINE int poisson (float mean) {
        float g = expf (-mean);
        unsigned int em = 0;
        float t = (*this)();
        while (t > g) {
            ++em;
            t *= (*this)();
        }
        return em;
    }
private:
    unsigned int m_seed;
};


// The Gabor kernel is a harmonic (cosine) modulated by a Gaussian
// envelope.  This version is augmented with a phase, per [Lagae2011].
//   \param  weight      magnitude of the pulse
//   \param  omega       orientation of the harmonic
//   \param  phi         phase of the harmonic.
//   \param  bandwidth   width of the gaussian envelope (called 'a'
//                          in [Lagae09].
//   \param  x           the position being sampled
// Looks identical to OSL::pvt::gabor_kernel, but will call sfm::exp & sfm::cos
template <class VEC>   // VEC should be Vec3 or Vec2
static OSL_FORCEINLINE Dual2<float>
gabor_kernel (const Dual2<float> &weight, const VEC &omega,
              const Dual2<float> &phi, float bandwidth, const Dual2<VEC> &x)
{
    // see Equation 1

    Dual2<float> g = OIIO::fast_exp (float(-M_PI) * (bandwidth * bandwidth) * dot(x,x));
    Dual2<float> h = OIIO::fast_cos (float(M_TWO_PI) * dot(omega,x) + phi);
    return weight * g * h;
}

// Looks identical to OSL::pvt::slice_gabor_kernel_3d, but will call sfm::exp
static OSL_FORCEINLINE void
slice_gabor_kernel_3d (const Dual2<float> &d, float w, float a,
                       const Vec3 &omega, float phi,
                       Dual2<float> &w_s, Vec2 &omega_s, Dual2<float> &phi_s)
{
    // Equation 6
    w_s = w * OIIO::fast_exp(float(-M_PI) * (a*a)*(d*d));
    //omega_s[0] = omega[0];
    //omega_s[1] = omega[1];
    //phi_s = phi - float(M_TWO_PI) * d * omega[2];
    omega_s.x = omega.x;
    omega_s.y = omega.y;
    // A.W. think this was a bug, supposed to be omega.z not omega.x;
    //phi_s = phi - float(M_TWO_PI) * d * omega.x;
    phi_s = phi - float(M_TWO_PI) * d * omega.z;
}

    struct GaborUniformParams {
        float a;
        float lambda;
        float sqrt_lambda_inv;
        float radius, radius2, radius3, radius_inv;

        OSL_FORCEINLINE GaborUniformParams (const NoiseParams &opt)
        {
            float bandwidth = Imath::clamp(opt.bandwidth,0.01f,100.0f);
            // NOTE: this could be a source of numerical differences
            // between the single point vs. batched
    #if OSL_FAST_MATH
            float TWO_to_bandwidth = OIIO::fast_exp2(bandwidth);
    #else
            float TWO_to_bandwidth = exp2f(bandwidth);
    #endif
            //static const float SQRT_PI_OVER_LN2 = sqrtf (M_PI / M_LN2);
            // To match GCC result of sqrtf (M_PI / M_LN2)
            static constexpr float SQRT_PI_OVER_LN2 = 2.128934e+00f;

            a = Gabor_Frequency * ((TWO_to_bandwidth - 1.0f) / (TWO_to_bandwidth + 1.0f)) * SQRT_PI_OVER_LN2;
            // Calculate the maximum radius from which we consider the kernel
            // impulse centers -- derived from the threshold and bandwidth.
            radius = sqrtf(-logf(Gabor_Truncate) / float(M_PI)) / a;
            radius2 = radius * radius;
            radius3 = radius2 * radius;
            radius_inv = 1.0f / radius;
            // Lambda is the impulse density.
            float impulses = Imath::clamp (opt.impulses, 1.0f, 32.0f);
            lambda = impulses / (float(1.33333 * M_PI) * radius3);
            sqrt_lambda_inv = 1.0f / sqrtf(lambda);
        }
    };



    struct GaborParams {
        Matrix22 filter;
        sfm::Matrix33 local;
        Vec3 N;
        Vec3 period;
        Vec3 omega; // anisotropy orientation
        float det_filter;
        int do_filter;

        OSL_FORCEINLINE GaborParams(const Vec3 &direction)
        : filter(Imath::UNINITIALIZED)
        , local(Imath::UNINITIALIZED)
        , omega(direction)
        {}


        OSL_FORCEINLINE GaborParams(const GaborParams &other)
        : filter(other.filter)
        , local(other.local)
        , N(other.N)
        , period(other.period)
        , omega(other.omega)
        , det_filter(other.det_filter)
        , do_filter (other.do_filter)
        {}

    };


    // set up the filter matrix
    static OSL_FORCEINLINE void
    gabor_setup_filter (const Dual2<Vec3> &P, sfm::GaborParams &gp)
    {
        // Make texture-space normal, tangent, bitangent
        Vec3 n, t, b;
        n = P.dx().cross (P.dy());  // normal to P
        int do_filter = 1;
        if (n.dot(n) < 1.0e-6f) {  /* length of deriv < 1/1000 */
            // No way to do filter if we have no derivs, and no reason to
            // do it if it's too small to have any effect.
            do_filter = 0;
        } else {
            make_orthonormals (n, t, b);

            // Rotations from tangent<->texture space
            sfm::Matrix33 Mtex_to_tan = sfm::make_matrix33_cols (t, b, n);  // M3_local
            sfm::Matrix33 Mscreen_to_tex = sfm::make_matrix33_cols (P.dx(), P.dy(), Vec3(0.0f,0.0f,0.0f));
            sfm::Matrix33 Mscreen_to_tan = Mscreen_to_tex * Mtex_to_tan;  // M3_scr_tan
            Matrix22 M_scr_tan (Mscreen_to_tan.x[0][0], Mscreen_to_tan.x[0][1],
                                Mscreen_to_tan.x[1][0], Mscreen_to_tan.x[1][1]);
            float sigma_f_scr = 0.5f;
            Matrix22 Sigma_f_scr (sigma_f_scr * sigma_f_scr, 0.0f,
                                  0.0f, sigma_f_scr * sigma_f_scr);
            Matrix22 M_scr_tan_t = M_scr_tan.transposed();
            Matrix22 Sigma_f_tan = M_scr_tan_t * Sigma_f_scr * M_scr_tan;

            gp.N = n;
            gp.filter = Sigma_f_tan;
            gp.local  = Mtex_to_tan;
            float det_filter = determinant(Sigma_f_tan);
            gp.det_filter = det_filter;
            if (det_filter < 1.0e-18f) {
                do_filter = 0;
                // Turn off filtering when tiny values will lead to numerical
                // errors later if we filter.  Yes, it's kind of arbitrary.
            }
        }
        gp.do_filter = do_filter;
    }

// Choose an omega and phi value for a particular gabor impulse,
// based on the user-selected noise mode.
    template<int AnisotropicT>
    static OSL_FORCEINLINE void
    gabor_sample (const sfm::GaborParams &gp, const Vec3 &x_c, sfm::fast_rng &rng,
                  Vec3 &omega, float &phi)
    {
        // section 3.3, solid random-phase gabor noise
        if (AnisotropicT == 1 /* anisotropic */) {
            omega = gp.omega;
        } else if (AnisotropicT == 0 /* isotropic */) {
            float omega_t = float (M_TWO_PI) * rng();
            // float omega_p = acosf(lerp(-1.0f, 1.0f, rng()));
            float cos_omega_p = OIIO::lerp(-1.0f, 1.0f, rng());
            float sin_omega_p = sqrtf (sfm::max_val (0.0f, 1.0f - cos_omega_p*cos_omega_p));
#if 0
            float sin_omega_t, cos_omega_t;
    #if OSL_FAST_MATH
            OIIO::fast_sincos (omega_t, &sin_omega_t, &cos_omega_t);
    #else
            OIIO::sincos (omega_t, &sin_omega_t, &cos_omega_t);
    #endif
#else
    #if OSL_FAST_MATH
            float sin_omega_t, cos_omega_t;
            OIIO::sincos (omega_t, sin_omega_t, cos_omega_t);
    #else
            // NOTE: optimizing compilers will see sin & cos
            // on the same value and call a single sincos function
            float sin_omega_t = sinf(omega_t);
            float cos_omega_t = cosf(omega_t);
    #endif
#endif
            //omega = Vec3 (cos_omega_t*sin_omega_p, sin_omega_t*sin_omega_p, cos_omega_p).normalized();
            omega = sfm::normalize(Vec3 (cos_omega_t*sin_omega_p, sin_omega_t*sin_omega_p, cos_omega_p));
        } else {
            // otherwise hybrid
            float omega_r = gp.omega.length();
            float omega_t =  float(M_TWO_PI) * rng();
#if 0
            float sin_omega_t, cos_omega_t;
    #if OSL_FAST_MATH
            OIIO::fast_sincos (omega_t, &sin_omega_t, &cos_omega_t);
    #else
            OIIO::sincos (omega_t, &sin_omega_t, &cos_omega_t);
    #endif
#else
    #if OSL_FAST_MATH
            float sin_omega_t, cos_omega_t;
            OIIO::sincos (omega_t, sin_omega_t, cos_omega_t);
    #else
            // NOTE: optimizing compilers will see sin & cos
            // on the same value and call a single sincos function
            float sin_omega_t = sinf(omega_t);
            float cos_omega_t = cosf(omega_t);
    #endif
#endif
            omega = omega_r * Vec3(cos_omega_t, sin_omega_t, 0.0f);
        }
        phi = float(M_TWO_PI) * rng();
    }

    // gabor_cell was unnecesarily calling down to sample even when outside the radius
    // however to match existing results, we need to bump the random number generator
    // an equivalent # of times.  Here the sample function stripped down to just
    // the rng() calls
    template<int AnisotropicT>
    static OSL_FORCEINLINE void
    gabor_no_sample (sfm::fast_rng &rng)
    {
        // section 3.3, solid random-phase gabor noise
        if (AnisotropicT == 1 /* anisotropic */) {
        } else if (AnisotropicT == 0 /* isotropic */) {
            rng();
            rng();
        } else {
            rng();
        }
        rng();
    }

    static OSL_FORCEINLINE void
    filter_gabor_kernel_2d (const Matrix22 &filter, const Dual2<float> &w, float a,
                            const Vec2 &omega, const Dual2<float> &phi,
                            Dual2<float> &w_f, float &a_f,
                            Vec2 &omega_f, Dual2<float> &phi_f)
    {
        //  Equation 10
        Matrix22 Sigma_f = filter;
        Dual2<float> c_G = w;
        Vec2 mu_G = omega;
        Matrix22 Sigma_G = (a * a / float(M_TWO_PI)) * Matrix22();
        float c_F = 1.0f / (float(M_TWO_PI) * sqrtf(determinant(Sigma_f)));
#if 0
        Matrix22 Sigma_F = float(1.0 / (4.0 * M_PI * M_PI)) * Sigma_f.inverse();
#else
        Matrix22 Sigma_f_inverse = Sigma_f.inverse();
        Matrix22 Sigma_F = float(1.0 / (4.0 * M_PI * M_PI)) * Sigma_f_inverse;
#endif
        Matrix22 Sigma_G_Sigma_F = Sigma_G + Sigma_F;
        Dual2<float> c_GF = c_F * c_G
            * (1.0f / (float(M_TWO_PI) * sqrtf(determinant(Sigma_G_Sigma_F))))
            * expf(-0.5f * dot(Sigma_G_Sigma_F.inverse()*mu_G, mu_G));
        Matrix22 Sigma_G_i = Sigma_G.inverse();
        Matrix22 Sigma_GF = (Sigma_F.inverse() + Sigma_G_i).inverse();
        Vec2 mu_GF;
        Matrix22 Sigma_GF_Gi = Sigma_GF * Sigma_G_i;
        Sigma_GF_Gi.multMatrix (mu_G, mu_GF);
        w_f = c_GF;
        a_f = sqrtf(M_TWO_PI * sqrtf(determinant(Sigma_GF)));
        omega_f = mu_GF;
        phi_f = phi;
    }


    // Evaluate the summed contribution of all gabor impulses within the
    // cell whose corner is c_i.  x_c_i is vector from x (the point
    // we are trying to evaluate noise at) and c_i.
    template<int AnisotropicT, typename FilterPolicyT, bool PeriodicT>
    static OSL_FORCEINLINE Dual2<float>
    gabor_cell (const sfm::GaborUniformParams &gup, const sfm::GaborParams &gp, const Vec3 &c_i, const Dual2<Vec3> &x_c_i,
                int seed = 0)
    {
        sfm::fast_rng rng (PeriodicT ? Vec3(wrap(c_i,gp.period)) : c_i, seed);
        int n_impulses = rng.poisson (gup.lambda * gup.radius3);
        Dual2<float> sum = 0.0f;

        for (int i = 0; i < n_impulses; i++) {
            // OLD code: Vec3 x_i_c (rng(), rng(), rng());
            // Turned out that C++ spec says order of args are unspecified.
            // gcc appeared to do right-to-left, so to make sure our noise
            // function is locked down (and works identically for clang,
            // which evaluates left-to-right), we ask for the rng() calls
            // one at a time and match the way it looked before.
            float z_rng = rng(), y_rng = rng(), x_rng = rng();
            Vec3 x_i_c (x_rng, y_rng, z_rng);
            Dual2<Vec3> x_k_i = gup.radius * (x_c_i - x_i_c);
            if (OSL_UNLIKELY(x_k_i.val().length2() < gup.radius2)) {
                float phi_i;
                Vec3 omega_i;
                // moved inside of conditional, notice workaround in the else
                // case to match random number generator of the original version
                sfm::gabor_sample<AnisotropicT>(gp, c_i, rng, omega_i, phi_i);

                if (OSL_LIKELY(FilterPolicyT::active && (gp.do_filter == 1))) {
                    // Transform the impulse's anisotropy into tangent space
                    Vec3 omega_i_t;
                    multMatrix (gp.local, omega_i, omega_i_t);

                    // Slice to get a 2D kernel
                    Dual2<float> d_i = -dot(gp.N, x_k_i);
                    Dual2<float> w_i_t_s;
                    Vec2 omega_i_t_s;
                    Dual2<float> phi_i_t_s;
                    slice_gabor_kernel_3d (d_i, Gabor_Impulse_Weight, gup.a,
                                           omega_i_t, phi_i,
                                           w_i_t_s, omega_i_t_s, phi_i_t_s);

                    // Filter the 2D kernel
                    Dual2<float> w_i_t_s_f;
                    float a_i_t_s_f;
                    Vec2 omega_i_t_s_f;
                    Dual2<float> phi_i_t_s_f;
                    sfm::filter_gabor_kernel_2d (gp.filter, w_i_t_s, gup.a, omega_i_t_s, phi_i_t_s, w_i_t_s_f, a_i_t_s_f, omega_i_t_s_f, phi_i_t_s_f);

                    // Now evaluate the 2D filtered kernel
                    Dual2<Vec3> xkit;
                    multMatrix (gp.local, x_k_i, xkit);
                    Dual2<Vec2> x_k_i_t = make_Vec2 (comp_x(xkit), comp_y(xkit));
                    Dual2<float> gk = gabor_kernel (w_i_t_s_f, omega_i_t_s_f, phi_i_t_s_f, a_i_t_s_f, x_k_i_t); // 2D
#if defined(__AVX512F__) && defined(__INTEL_COMPILER) && (__INTEL_COMPILER < 1800)
                    // icc17 with AVX512 had some incorrect results
                    // due to the not_finite code path executing even
                    // when the value was finite.  Workaround: using isnan | isinf
                    // instead of isfinite avoided the issue.
                    // icc18u3 doesn't exhibit the problem
                    // NOTE: tried using bitwise | to avoid branches and got internal compiler error
                    //bool not_finite = std::isnan(gk.val()) | std::isinf(gk.val());
                    bool not_finite = std::isnan(gk.val()) || std::isinf(gk.val());
#else
                    bool not_finite = !OIIO::isfinite(gk.val());
#endif
                    if (OSL_UNLIKELY(not_finite))
                    {
                        // Numeric failure of the filtered version.  Fall
                        // back on the unfiltered.
                        gk = gabor_kernel (Gabor_Impulse_Weight, omega_i, phi_i, gup.a, x_k_i);  // 3D
                    }
                    sum += gk;
                } else {
                    // N.B. if determinant(gp.filter) is too small, we will
                    // run into numerical problems.  But the filtering isn't
                    // needed in that case anyway, so just don't filter.
                    // This seems to only come up when the filter region is
                    // tiny.
                    sum += gabor_kernel (Gabor_Impulse_Weight, omega_i, phi_i, gup.a, x_k_i);  // 3D
                }
            } else {
                // Since we skipped the sample, we still need to bump the rng as if we had
                sfm::gabor_no_sample<AnisotropicT>(rng);
            }

        }

        return sum;
    }


    // Sum the contributions of gabor impulses in all neighboring cells
    // surrounding position x_g.
    template<int AnisotropicT, typename FilterPolicyT, bool PeriodicT>
    static OSL_FORCEINLINE Dual2<float>
    gabor_grid (const sfm::GaborUniformParams &gup, const sfm::GaborParams &gp, const Dual2<Vec3> &x_g, int seed=0)
    {
        using pvt::floor;
        Vec3 floor_x_g (floor (x_g));  // Vec3 because floor has no derivs
        Dual2<Vec3> x_c = x_g - floor_x_g;
        Dual2<float> sum = 0.0f;

OSL_INTEL_PRAGMA(nounroll_and_jam)
        for (int k = -1; k <= 1; k++) {
OSL_INTEL_PRAGMA(nounroll_and_jam)
            for (int j = -1; j <= 1; j++) {
OSL_INTEL_PRAGMA(nounroll_and_jam)
                for (int i = -1; i <= 1; i++) {
                    Vec3 c (i,j,k);
                    Vec3 c_i = floor_x_g + c;
                    Dual2<Vec3> x_c_i = x_c - c;
                    sum += sfm::gabor_cell<AnisotropicT, FilterPolicyT, PeriodicT>(gup, gp, c_i, x_c_i, seed);
                }
            }
        }
#if 0 // Alternative to the above loop nest, keeping around for experimentation
    {
#if 1
        const Vec3i c_ijk[27] = {
            {-1,-1,-1}, {0,-1,-1}, {1,-1,-1},
            {-1,0,-1}, {0,0,-1}, {1,0,-1},
            {-1,1,-1}, {0,1,-1}, {1,1,-1},

            {-1,-1,0}, {0,-1,0}, {1,-1,0},
            {-1,0,0}, {0,0,0}, {1,0,0},
            {-1,1,0}, {0,1,0}, {1,1,0},

            {-1,-1,1}, {0,-1,1}, {1,-1,1},
            {-1,0,1}, {0,0,1}, {1,0,1},
            {-1,1,1}, {0,1,1}, {1,1,1},
        };
#endif
#if 0
        //for(int c_index=0; c_index < 27; ++c_index)
        {
            Vec3 c  = c_ijk[0];
            //Vec3 c(-1,-1,-1);
            Vec3 c_i = floor_x_g + c;
            Dual2<Vec3> x_c_i = x_c - c;
            sum += sfm::gabor_cell<AnisotropicT, FilterPolicyT, PeriodicT>(gup, gp, c_i, x_c_i, seed);
        }
        {
            Vec3 c  = c_ijk[1];
            //Vec3 c(-1,-1,-1);
            Vec3 c_i = floor_x_g + c;
            Dual2<Vec3> x_c_i = x_c - c;
            sum += sfm::gabor_cell<AnisotropicT, FilterPolicyT, PeriodicT>(gup, gp, c_i, x_c_i, seed);
        }
        {
            Vec3 c  = c_ijk[2];
            //Vec3 c(-1,-1,-1);
            Vec3 c_i = floor_x_g + c;
            Dual2<Vec3> x_c_i = x_c - c;
            sum += sfm::gabor_cell<AnisotropicT, FilterPolicyT, PeriodicT>(gup, gp, c_i, x_c_i, seed);
        }
#else
            //index_loop(c_index,27,
            OSL_INTEL_PRAGMA(nounroll)
            for(int c_index=0; c_index < 27; ++c_index)
            {
                Vec3 c  = c_ijk[c_index];
                Vec3 c_i = floor_x_g + c;
                Dual2<Vec3> x_c_i = x_c - c;
                sum += sfm::gabor_cell<AnisotropicT, FilterPolicyT, PeriodicT>(gup, gp, c_i, x_c_i, seed);
            }
            //);

#endif
        }

#endif

        return sum * gup.sqrt_lambda_inv;
    }

    template<int AnisotropicT, typename FilterPolicyT, bool PeriodicT, int SeedT>
    static OSL_FORCEINLINE Dual2<float>
    gabor_evaluate (const sfm::GaborUniformParams &gup, const sfm::GaborParams &gp, const Dual2<Vec3> &x)
    {
        Dual2<Vec3> x_g = x * gup.radius_inv;
        return sfm::gabor_grid<AnisotropicT, FilterPolicyT, PeriodicT>(gup, gp, x_g, SeedT);
    }

    template<int AnisotropicT, typename FilterPolicyT, bool PeriodicT>
    static OSL_FORCEINLINE Dual2<float>
    gabor_evaluate (const sfm::GaborUniformParams &gup, const sfm::GaborParams &gp, const Dual2<Vec3> &x, int seed)
    {
        Dual2<Vec3> x_g = x * gup.radius_inv;
        return sfm::gabor_grid<AnisotropicT, FilterPolicyT, PeriodicT>(gup, gp, x_g, seed);
    }

    template<int AnisotropicT, typename FilterPolicyT>
    static OSL_FORCEINLINE  Dual2<float>
    scalar_gabor (const Dual2<Vec3> &P, const sfm::GaborUniformParams &gup, const Vec3 & direction)
    {
        sfm::GaborParams gp(direction);

        if (FilterPolicyT::active)
            sfm::gabor_setup_filter (P, gp);

        Dual2<float> result = sfm::gabor_evaluate<AnisotropicT, FilterPolicyT, false, 0 /*seed*/>(gup, gp, P);
        float gabor_variance = 1.0f / (4.0f*sqrtf(2.0f) * (gup.a * gup.a * gup.a));
        float scale = 1.0f / (3.0f*sqrtf(gabor_variance));
        scale *= 0.5f;  // empirical -- make it fit in [-1..1]

        return result * scale;
    }

    template<int AnisotropicT, typename FilterPolicyT>
    static OSL_FORCEINLINE  Dual2<Vec3>
    scalar_gabor3 (const Dual2<Vec3> &P, const sfm::GaborUniformParams &gup, const Vec3 & direction)
    {
        sfm::GaborParams gp(direction);

        if (FilterPolicyT::active)
            sfm::gabor_setup_filter (P, gp);

#if 0
        Dual2<Vec3> result = make_Vec3 (sfm::gabor_evaluate<AnisotropicT, FilterPolicyT, false /*periodic*/>(gup, gp, P, 0 /*seed*/),
                                        sfm::gabor_evaluate<AnisotropicT, FilterPolicyT, false /*periodic*/>(gup, gp, P, 1 /*seed*/),
                                        sfm::gabor_evaluate<AnisotropicT, FilterPolicyT, false /*periodic*/>(gup, gp, P, 2 /*seed*/));
#else
//        Dual2<float> rval = sfm::gabor_evaluate<AnisotropicT, FilterPolicyT, false /*periodic*/>(gup, gp, P, 0 /*seed*/);
//        Dual2<float> rdx = sfm::gabor_evaluate<AnisotropicT, FilterPolicyT, false /*periodic*/>(gup, gp, P, 1 /*seed*/);
//        Dual2<float> rdy = sfm::gabor_evaluate<AnisotropicT, FilterPolicyT, false /*periodic*/>(gup, gp, P, 2 /*seed*/);
        Dual2<float> resultParts[3];
        OSL_INTEL_PRAGMA(nounroll_and_jam)
        for(int seed=0; seed<3; ++seed) {
            resultParts[seed] = sfm::gabor_evaluate<AnisotropicT, FilterPolicyT, false /*periodic*/>(gup, gp, P, seed);
        }

        Dual2<Vec3> result = make_Vec3(resultParts[0], resultParts[1], resultParts[2]);
#endif
        float gabor_variance = 1.0f / (4.0f*sqrtf(2.0f) * (gup.a * gup.a * gup.a));
        float scale = 1.0f / (3.0f*sqrtf(gabor_variance));
        scale *= 0.5f;  // empirical -- make it fit in [-1..1]

        return result * scale;
    }

    template<int AnisotropicT, typename FilterPolicyT>
    static OSL_FORCEINLINE  Dual2<float>
    scalar_pgabor (const Dual2<Vec3> &P, const Vec3 & Pperiod, const sfm::GaborUniformParams &gup, const Vec3 & direction)
    {
        sfm::GaborParams gp(direction);
        gp.period = Pperiod;

        if (FilterPolicyT::active)
            sfm::gabor_setup_filter (P, gp);

        Dual2<float> result = sfm::gabor_evaluate<AnisotropicT, FilterPolicyT, true /*periodic*/, 0 /*seed*/>(gup, gp, P);
        float gabor_variance = 1.0f / (4.0f*sqrtf(2.0f) * (gup.a * gup.a * gup.a));
        float scale = 1.0f / (3.0f*sqrtf(gabor_variance));
        scale *= 0.5f;  // empirical -- make it fit in [-1..1]

        return result * scale;
    }


    template<int AnisotropicT, typename FilterPolicyT>
    static OSL_FORCEINLINE  Dual2<Vec3>
    scalar_pgabor3 (const Dual2<Vec3> &P, const Vec3 &Pperiod, const sfm::GaborUniformParams &gup, const Vec3 & direction)
    {
        sfm::GaborParams gp(direction);
        gp.period = Pperiod;

        if (FilterPolicyT::active)
            sfm::gabor_setup_filter (P, gp);

        Dual2<Vec3> result = make_Vec3 (sfm::gabor_evaluate<AnisotropicT, FilterPolicyT, true /*periodic*/, 0 /*seed*/>(gup, gp, P),
                                        sfm::gabor_evaluate<AnisotropicT, FilterPolicyT, true /*periodic*/, 1 /*seed*/>(gup, gp, P),
                                        sfm::gabor_evaluate<AnisotropicT, FilterPolicyT, true /*periodic*/, 2 /*seed*/>(gup, gp, P));
        float gabor_variance = 1.0f / (4.0f*sqrtf(2.0f) * (gup.a * gup.a * gup.a));
        float scale = 1.0f / (3.0f*sqrtf(gabor_variance));
        scale *= 0.5f;  // empirical -- make it fit in [-1..1]

        return result * scale;
    }

} // namespace sfm


} // namespace __OSL_WIDE_PVT

OSL_NAMESPACE_EXIT
