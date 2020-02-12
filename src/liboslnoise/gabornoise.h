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
#include <OSL/oslnoise.h>
#include <OSL/dual_vec.h>
#include <OSL/Imathx.h>

#include <OpenImageIO/fmath.h>

OSL_NAMESPACE_ENTER

namespace pvt {

// TODO: It would be preferable to use the Imath versions of these functions in
//       all cases, but these templates should suffice until a more complete
//       device-friendly version of Imath is available.
namespace hostdevice {
template <typename T> inline OSL_HOSTDEVICE T clamp (T x, T lo, T hi);
#ifndef __CUDA_ARCH__
template <> inline OSL_HOSTDEVICE double clamp<double> (double x, double lo, double hi) { return Imath::clamp (x, lo, hi); }
template <> inline OSL_HOSTDEVICE float  clamp<float>  (float x, float lo, float hi)    { return Imath::clamp (x, lo, hi); }
#else
template <> inline OSL_HOSTDEVICE double clamp<double> (double x, double lo, double hi) { return (x < lo) ? lo : ((x > hi) ? hi : x); }
template <> inline OSL_HOSTDEVICE float  clamp<float>  (float x, float lo, float hi)    { return (x < lo) ? lo : ((x > hi) ? hi : x); }
#endif
}

static OSL_DEVICE constexpr float Gabor_Frequency = 2.0;
static OSL_DEVICE constexpr float Gabor_Impulse_Weight = 1.0f;

// The Gabor kernel in theory has infinite support (its envelope is
// a Gaussian).  To restrict the distance at which we must sum the
// kernels, we only consider those whose Gaussian envelopes are
// above the truncation threshold, as a portion of the Gaussian's
// peak value.
static OSL_DEVICE const float Gabor_Truncate = 0.02f;



// Very fast random number generator based on [Borosh & Niederreiter 1983]
// linear congruential generator.
class fast_rng {
public:
    // seed based on the cell containing P
    OSL_DEVICE
    fast_rng (const Vec3 &p, int seed=0) {
        // Use guts of cellnoise
        m_seed = inthash(unsigned(OIIO::ifloor(p.x)),
                         unsigned(OIIO::ifloor(p.y)),
                         unsigned(OIIO::ifloor(p.z)),
                         unsigned(seed));
        if (! m_seed)
            m_seed = 1;
    }
    // Return uniform on [0,1)
    OSL_HOSTDEVICE
    float operator() () {
        return (m_seed *= 3039177861u) / float(UINT_MAX);
    }
    // Return poisson distribution with the given mean
    OSL_HOSTDEVICE
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

// The Gabor kernel is a harmonic (cosine) modulated by a Gaussian
// envelope.  This version is augmented with a phase, per [Lagae2011].
//   \param  weight      magnitude of the pulse
//   \param  omega       orientation of the harmonic
//   \param  phi         phase of the harmonic.
//   \param  bandwidth   width of the gaussian envelope (called 'a'
//                          in [Lagae09].
//   \param  x           the position being sampled
template <class VEC>   // VEC should be Vec3 or Vec2
inline OSL_HOSTDEVICE Dual2<float>
gabor_kernel (const Dual2<float> &weight, const VEC &omega,
              const Dual2<float> &phi, float bandwidth, const Dual2<VEC> &x)
{
    // see Equation 1
    Dual2<float> g = exp (float(-M_PI) * (bandwidth * bandwidth) * dot(x,x));
    Dual2<float> h = cos (float(M_TWO_PI) * dot(omega,x) + phi);
    return weight * g * h;
}



inline OSL_HOSTDEVICE void
slice_gabor_kernel_3d (const Dual2<float> &d, float w, float a,
                       const Vec3 &omega, float phi,
                       Dual2<float> &w_s, Vec2 &omega_s, Dual2<float> &phi_s)
{
    // Equation 6
    w_s = w * exp(float(-M_PI) * (a*a)*(d*d));
    //omega_s[0] = omega[0];
    //omega_s[1] = omega[1];
    //phi_s = phi - float(M_TWO_PI) * d * omega[2];
    omega_s.x = omega.x;
    omega_s.y = omega.y;
    // A.W. think this was a bug, supposed to be omega.z not omega.x;
    //phi_s = phi - float(M_TWO_PI) * d * omega.x;
    phi_s = phi - float(M_TWO_PI) * d * omega.z;
}


static  OSL_HOSTDEVICE void
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


OSL_FORCEINLINE OSL_HOSTDEVICE float
wrap (float s, float period)
{
    period = floorf (period);
    if (period < 1.0f)
        period = 1.0f;
    return s - period * floorf (s / period);
}


// avoid aliasing issues
static OSL_FORCEINLINE OSL_HOSTDEVICE Vec3
wrap (const Vec3 &s, const Vec3 &period)
{
    return Vec3 (wrap (s.x, period.x),
                 wrap (s.y, period.y),
                 wrap (s.z, period.z));
}


// Normalize v and set a and b to be unit vectors (any two unit vectors)
// that are orthogonal to v and each other.  We get the first
// orthonormal by taking the cross product of v and (1,0,0), unless v
// points roughly toward (1,0,0), in which case we cross with (0,1,0).
// Either way, we get something orthogonal.  Then cross(v,a) is mutually
// orthogonal to the other two.
inline OSL_HOSTDEVICE void
make_orthonormals (Vec3 &v, Vec3 &a, Vec3 &b)
{
    // avoid aliasing issues by not using the [] operator
    v.normalize();
    if (fabsf(v.x) < 0.9f)
        a.setValue (0.0f, v.z, -v.y);   // v X (1,0,0)
    else
        a.setValue (-v.z, 0.0f, v.x);   // v X (0,1,0)
    a.normalize ();
    b = v.cross (a);
//    b.normalize ();  // note: not necessary since v is unit length
}



// Helper function: per-component 'floor' of a Dual2<Vec3>.
inline OSL_HOSTDEVICE Vec3
floor (const Dual2<Vec3> &vd)
{
    // avoid aliasing issues by not using the [] operator
    const Vec3 &v (vd.val());
    return Vec3 (floorf(v.x), floorf(v.y), floorf(v.z));
}

} // namespace pvt

OSL_NAMESPACE_EXIT
