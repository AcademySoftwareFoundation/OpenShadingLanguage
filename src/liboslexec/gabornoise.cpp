/*
Copyright (c) 2012 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <limits>

#include "oslexec_pvt.h"
#include "noiseimpl.h"
#include "dual_vec.h"
#include "Imathx.h"

#include <OpenImageIO/fmath.h>

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
namespace pvt {


static const float Gabor_Frequency = 2.0;
static const float Gabor_Impulse_Weight = 1;

// The Gabor kernel in theory has infinite support (its envelope is
// a Gaussian).  To restrict the distance at which we must sum the
// kernels, we only consider those whose Gaussian envelopes are
// above the truncation threshold, as a portion of the Gaussian's
// peak value.
static const float Gabor_Truncate = 0.02f;



// Very fast random number generator based on [Borosh & Niederreiter 1983]
// linear congruential generator.
class fast_rng {
public:
    // seed based on the cell containing P
    fast_rng (const Vec3 &p, int seed=0) {
        // Use guts of cellnoise
        unsigned int pi[4] = { quick_floor(p[0]), quick_floor(p[1]),
                               quick_floor(p[2]), seed };
        m_seed = inthash<4>(pi);
        if (! m_seed)
            m_seed = 1;
    }
    // Return uniform on [0,1)
    float operator() () {
        return (m_seed *= 3039177861) / float(UINT_MAX);
    }
    // Return poisson distribution with the given mean
    int poisson (float mean) {
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

    GaborParams (const NoiseParams &opt) :
        omega(opt.direction),  // anisotropy orientation
        anisotropic(opt.anisotropic),
        do_filter(opt.do_filter),
        weight(Gabor_Impulse_Weight),
        bandwidth(Imath::clamp(opt.bandwidth,0.01f,100.0f)),
        periodic(false)
    {
        float TWO_to_bandwidth = powf (2.0f, bandwidth);
        static const float SQRT_PI_OVER_LN2 = sqrtf (M_PI / M_LN2);
        a = Gabor_Frequency * ((TWO_to_bandwidth - 1.0) / (TWO_to_bandwidth + 1.0)) * SQRT_PI_OVER_LN2;
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



// The Gabor kernel is a harmonic (cosine) modulated by a Gaussian
// envelope.  This version is augmented with a phase, per [Lagae2011].
//   \param  weight      magnitude of the pulse
//   \param  omega       orientation of the harmonic
//   \param  phi         phase of the harmonic.
//   \param  bandwidth   width of the gaussian envelope (called 'a'
//                          in [Lagae09].
//   \param  x           the position being sampled
template <class VEC>   // VEC should be Vec3 or Vec2
inline Dual2<float>
gabor_kernel (const Dual2<float> &weight, const VEC &omega,
              const Dual2<float> &phi, float bandwidth, const Dual2<VEC> &x)
{
    // see Equation 1
    Dual2<float> g = exp (float(-M_PI) * (bandwidth * bandwidth) * dot(x,x));
    Dual2<float> h = cos (float(M_TWO_PI) * dot(omega,x) + phi);
    return weight * g * h;
}



inline void
slice_gabor_kernel_3d (const Dual2<float> &d, float w, float a,
                       const Vec3 &omega, float phi,
                       Dual2<float> &w_s, Vec2 &omega_s, Dual2<float> &phi_s)
{
    // Equation 6
    w_s = w * exp(float(-M_PI) * (a*a)*(d*d));
    omega_s[0] = omega[0];
    omega_s[1] = omega[1];
    phi_s = phi - float(M_TWO_PI) * d * omega[2];
}



static void
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
    Matrix22 Sigma_F = float(1.0 / (4.0 * M_PI * M_PI)) * Sigma_f.inverse();
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



// Choose an omega and phi value for a particular gabor impulse,
// based on the user-selected noise mode.
static void
gabor_sample (GaborParams &gp, const Vec3 &x_c, fast_rng &rng,
              Vec3 &omega, float &phi)
{
    // section 3.3, solid random-phase gabor noise
    if (gp.anisotropic == 1 /* anisotropic */) {
        omega = gp.omega;
    } else if (gp.anisotropic == 0 /* isotropic */) {
        float omega_t = float (M_TWO_PI) * rng();
        // float omega_p = acosf(lerp(-1.0f, 1.0f, rng()));
        float cos_omega_p = lerp(-1.0f, 1.0f, rng());
        float sin_omega_p = sqrtf (std::max (0.0f, 1.0f - cos_omega_p*cos_omega_p));
        float sin_omega_t, cos_omega_t;
        OIIO::sincos (omega_t, &sin_omega_t, &cos_omega_t);
        omega = Vec3 (cos_omega_t*sin_omega_p, sin_omega_t*sin_omega_p, cos_omega_p).normalized();
    } else {
        // otherwise hybrid
        float omega_r = gp.omega.length();
        float omega_t =  float(M_TWO_PI) * rng();
        float sin_omega_t, cos_omega_t;
        OIIO::sincos (omega_t, &sin_omega_t, &cos_omega_t);
        omega = omega_r * Vec3(cos_omega_t, sin_omega_t, 0.0f);
    }
    phi = float(M_TWO_PI) * rng();
}



inline float
wrap (float s, float period)
{
    period = floorf (period);
    if (period < 1.0f)
        period = 1.0f;
    return s - period * floorf (s / period);
}



static Vec3
wrap (const Vec3 &s, const Vec3 &period)
{
    return Vec3 (wrap (s[0], period[0]),
                 wrap (s[1], period[1]),
                 wrap (s[2], period[2]));
}



// Evaluate the summed contribution of all gabor impulses within the
// cell whose corner is c_i.  x_c_i is vector from x (the point
// we are trying to evaluate noise at) and c_i.
Dual2<float>
gabor_cell (GaborParams &gp, const Vec3 &c_i, const Dual2<Vec3> &x_c_i,
            int seed = 0)
{
    fast_rng rng (gp.periodic ? Vec3(wrap(c_i,gp.period)) : c_i, seed);
    int n_impulses = rng.poisson (gp.lambda * gp.radius3);
    Dual2<float> sum = 0;

    for (int i = 0; i < n_impulses; i++) {
        Vec3 x_i_c (rng(), rng(), rng());
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
                Dual2<Vec2> x_k_i_t = make_Vec2 (comp(xkit,0), comp(xkit,1));
                Dual2<float> gk = gabor_kernel (w_i_t_s_f, omega_i_t_s_f, phi_i_t_s_f, a_i_t_s_f, x_k_i_t); // 2D
                if (! std::isfinite(gk.val())) {
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



// Normalize v and set a and b to be unit vectors (any two unit vectors)
// that are orthogonal to v and each other.  We get the first
// orthonormal by taking the cross product of v and (1,0,0), unless v
// points roughly toward (1,0,0), in which case we cross with (0,1,0).
// Either way, we get something orthogonal.  Then cross(v,a) is mutually
// orthogonal to the other two.
inline void
make_orthonormals (Vec3 &v, Vec3 &a, Vec3 &b)
{
    v.normalize();
    if (fabsf(v[0]) < 0.9f)
	a.setValue (0.0f, v[2], -v[1]);   // v X (1,0,0)
    else
        a.setValue (v[2], 0.0f, v[0]);    // v X (0,1,0)
    a.normalize ();
    b = v.cross (a);
//    b.normalize ();  // note: not necessary since v is unit length
}



// Helper function: per-component 'floor' of a Dual2<Vec3>.
inline Vec3
floor (const Dual2<Vec3> &vd)
{
    const Vec3 &v (vd.val());
    return Vec3 (floorf(v[0]), floorf(v[1]), floorf(v[2]));
}



// Sum the contributions of gabor impulses in all neighboring cells
// surrounding position x_g.
static Dual2<float>
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



inline Dual2<float>
gabor_evaluate (GaborParams &gp, const Dual2<Vec3> &x, int seed=0)
{
    Dual2<Vec3> x_g = x * gp.radius_inv;
    return gabor_grid (gp, x_g, seed);
}



inline Matrix33
make_matrix33_rows (const Vec3 &a, const Vec3 &b, const Vec3 &c)
{
    return Matrix33 (a[0], a[1], a[2],
                     b[0], b[1], b[2],
                     c[0], c[1], c[2]);
}



inline Matrix33
make_matrix33_cols (const Vec3 &a, const Vec3 &b, const Vec3 &c)
{
    return Matrix33 (a[0], b[0], c[0],
                     a[1], b[1], c[1],
                     a[2], b[2], c[2]);
}



// set up the filter matrix
static void
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



Dual2<float>
gabor (const Dual2<float> &x, const NoiseParams *opt)
{
    // for now, just slice 3D
    return gabor (make_Vec3(x), opt);
}



Dual2<float>
gabor (const Dual2<float> &x, const Dual2<float> &y, const NoiseParams *opt)
{
    // for now, just slice 3D
    return gabor (make_Vec3(x,y), opt);
}



Dual2<float>
gabor (const Dual2<Vec3> &P, const NoiseParams *opt)
{
    DASSERT (opt);
    GaborParams gp (*opt);

    if (gp.do_filter)
        gabor_setup_filter (P, gp);

    Dual2<float> result = gabor_evaluate (gp, P);
    float gabor_variance = 1.0f / (4.0f*sqrtf(2.0) * (gp.a * gp.a * gp.a));
    float scale = 1.0f / (3.0f*sqrtf(gabor_variance));
    scale *= 0.5f;  // empirical -- make it fit in [-1..1]

    return result * scale;
}




Dual2<Vec3>
gabor3 (const Dual2<float> &x, const NoiseParams *opt)
{
    // for now, just slice 3D
    return gabor3 (make_Vec3(x), opt);
}



Dual2<Vec3>
gabor3 (const Dual2<float> &x, const Dual2<float> &y, const NoiseParams *opt)
{
    // for now, just slice 3D
    return gabor3 (make_Vec3(x,y), opt);
}



Dual2<Vec3>
gabor3 (const Dual2<Vec3> &P, const NoiseParams *opt)
{
    DASSERT (opt);
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




Dual2<float>
pgabor (const Dual2<float> &x, float xperiod, const NoiseParams *opt)
{
    // for now, just slice 3D
    return pgabor (make_Vec3(x), Vec3(xperiod,0.0f,0.0f), opt);
}



Dual2<float>
pgabor (const Dual2<float> &x, const Dual2<float> &y,
        float xperiod, float yperiod, const NoiseParams *opt)
{
    // for now, just slice 3D
    return pgabor (make_Vec3(x,y), Vec3(xperiod,yperiod,0.0f), opt);
}



Dual2<float>
pgabor (const Dual2<Vec3> &P, const Vec3 &Pperiod, const NoiseParams *opt)
{
    DASSERT (opt);
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


Dual2<Vec3>
pgabor3 (const Dual2<float> &x, float xperiod, const NoiseParams *opt)
{
    // for now, just slice 3D
    return pgabor3 (make_Vec3(x), Vec3(xperiod,0.0f,0.0f), opt);
}



Dual2<Vec3>
pgabor3 (const Dual2<float> &x, const Dual2<float> &y,
        float xperiod, float yperiod, const NoiseParams *opt)
{
    // for now, just slice 3D
    return pgabor3 (make_Vec3(x,y), Vec3(xperiod,yperiod,0.0f), opt);
}



Dual2<Vec3>
pgabor3 (const Dual2<Vec3> &P, const Vec3 &Pperiod, const NoiseParams *opt)
{
    DASSERT (opt);
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
}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
using namespace OSL_NAMESPACE;
#endif


