/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
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

#include <cmath>

#include "oslops.h"
#include "oslexec_pvt.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {



/// Given values x and y on [0,1], convert them in place to values on
/// [-1,1] uniformly distributed over a unit sphere.  This code is
/// derived from Peter Shirley, "Realistic Ray Tracing", p. 103.
static void
to_unit_disk (float &x, float &y)
{
    float r, phi;
    float a = 2.0f * x - 1.0f;
    float b = 2.0f * y - 1.0f;
    if (a > -b) {
        if (a > b) {
            r = a;
	         phi = M_PI_4 * (b/a);
	     } else {
	         r = b;
	         phi = M_PI_4 * (2.0f - a/b);
	     }
    } else {
        if (a < b) {
            r = -a;
            phi = M_PI_4 * (4.0f + b/a);
        } else {
            r = -b;
            if (b != 0.0f)
                phi = M_PI_4 * (6.0f - a/b);
            else
                phi = 0.0f;
        }
    }
    x = r * cosf (phi);
    y = r * sinf (phi);
}



/// Make two unit vectors that are orthogonal to N and each other.  This
/// assumes that N is already normalized.  We get the first orthonormal
/// by taking the cross product of N and (1,1,1), unless N is 1,1,1, in
/// which case we cross with (-1,1,1).  Either way, we get something
/// orthogonal.  Then N x a is mutually orthogonal to the other two.
void
ClosurePrimitive::make_orthonormals (const Vec3 &N, Vec3 &a, Vec3 &b)
{
    if (N[0] != N[1] || N[0] != N[2])
        a = Vec3 (N[2]-N[1], N[0]-N[2], N[1]-N[0]);  // (1,1,1) x N
    else
        a = Vec3 (N[2]-N[1], N[0]+N[2], -N[1]-N[0]);  // (-1,1,1) x N
    a.normalize ();
    b = N.cross (a);
}

void
ClosurePrimitive::make_orthonormals (const Vec3 &N, const Vec3& T, Vec3 &x, Vec3& y)
{
    y = N.cross(T);
    x = y.cross(N);
}


void
ClosurePrimitive::sample_cos_hemisphere (const Vec3 &N, const Vec3 &omega_out,
                                         float randu, float randv,
                                         Vec3 &omega_in, float &pdf)
{
    // Default closure BSDF implementation: uniformly sample
    // cosine-weighted hemisphere above the point.
    to_unit_disk (randu, randv);
    float costheta = sqrtf (std::max(1 - randu * randu - randv * randv, 0.0f));
    Vec3 T, B;
    make_orthonormals (N, T, B);
    omega_in = randu * T + randv * B + costheta * N;
    pdf = costheta * (float) M_1_PI;
}



float
ClosurePrimitive::pdf_cos_hemisphere (const Vec3 &N, const Vec3 &omega_in)
{
    // Default closure BSDF implementation: cosine-weighted hemisphere
    // above the point.
    float costheta = N.dot (omega_in);
    return costheta > 0 ? (costheta * (float) M_1_PI) : 0;
}



namespace pvt {


class DiffuseClosure : public BSDFClosure {
public:
    DiffuseClosure () : BSDFClosure ("diffuse", "n") { }

    struct params_t {
        Vec3 N;
    };

    bool get_cone(const void *paramsptr,
                  const Vec3 &omega_out, Vec3 &axis, float &angle) const
    {
        const params_t *params = (const params_t *) paramsptr;
        float cosNO = params->N.dot(omega_out);
        if (cosNO > 0) {
           // we are viewing the surface from the same side as the normal
           axis = params->N;
           angle = (float) M_PI;
           return true;
        }
        // we are below the surface
        return false;
    }

    Color3 eval (const void *paramsptr, const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &omega_in, Labels &labels) const
    {
        const params_t *params = (const params_t *) paramsptr;
        float cos_pi = params->N.dot(omega_in) * (float) M_1_PI;
        labels = Labels( Labels::SURFACE | Labels::REFLECT | Labels::DIFFUSE );
        return Color3 (cos_pi, cos_pi, cos_pi);
    }

    void sample (const void *paramsptr, const Vec3 &Ng,
                 const Vec3 &omega_out, float randu, float randv,
                 Vec3 &omega_in, float &pdf, Color3 &eval, Labels &labels) const
    {
        const params_t *params = (const params_t *) paramsptr;
        float cosNO = params->N.dot(omega_out);
        if (cosNO > 0) {
           // we are viewing the surface from above - send a ray out with cosine
           // distribution over the hemisphere
           sample_cos_hemisphere (params->N, omega_out, randu, randv, omega_in, pdf);
           eval.setValue(pdf, pdf, pdf);
        } else {
           // no samples if we look at the surface from the wrong side
           pdf = 0; 
           omega_in.setValue(0.0f, 0.0f, 0.0f);
           eval.setValue(0.0f, 0.0f, 0.0f);
        }
        labels = Labels( Labels::SURFACE | Labels::REFLECT | Labels::DIFFUSE );
    }

    float pdf (const void *paramsptr, const Vec3 &Ng,
               const Vec3 &omega_out, const Vec3 &omega_in) const
    {
        const params_t *params = (const params_t *) paramsptr;
        return pdf_cos_hemisphere (params->N, omega_in);
    }

};


class TransparentClosure : public BSDFClosure {
public:
    TransparentClosure () : BSDFClosure ("transparent", "") { }

    bool get_cone(const void *paramsptr,
                  const Vec3 &omega_out, Vec3 &axis, float &angle) const
    {
        // does not need to be integrated directly
        return false;
    }

    Color3 eval (const void *paramsptr, const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &omega_in, Labels &labels) const
    {
        // should never be called - because get_cone is empty
        return Color3 (0.0f, 0.0f, 0.0f);
    }

    void sample (const void *paramsptr, const Vec3 &Ng,
                 const Vec3 &omega_out, float randu, float randv,
                 Vec3 &omega_in, float &pdf, Color3 &eval, Labels &labels) const
    {
        // only one direction is possible
        omega_in = -omega_out;
        pdf = 1;
        eval.setValue(1, 1, 1);
        labels = Labels(Labels::SURFACE | Labels::REFLECT | Labels::SINGULAR);
    }

    float pdf (const void *paramsptr, const Vec3 &Ng,
               const Vec3 &omega_out, const Vec3 &omega_in) const
    {
        // the pdf for an arbitrary direction is 0 because only a single
        // direction is actually possible
        return 0;
    }

};

// vanilla phong - leaks energy at grazing angles
// see Global Illumination Compendium entry (66) 
class PhongClosure : public BSDFClosure {
public:
    PhongClosure () : BSDFClosure ("phong", "nf") { }

    struct params_t {
        Vec3 N;
        float exponent;
    };

    bool get_cone(const void *paramsptr,
                  const Vec3 &omega_out, Vec3 &axis, float &angle) const
    {
        const params_t *params = (const params_t *) paramsptr;
        float cosNO = params->N.dot(omega_out);
        if (cosNO > 0) {
            // we are viewing the surface from the same side as the normal
            axis = params->N;
            angle = (float) M_PI;
            return true;
        }
        // we are below the surface
        return false;
    }

    Color3 eval (const void *paramsptr, const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &omega_in, Labels &labels) const
    {
        const params_t *params = (const params_t *) paramsptr;
        float cosNO = params->N.dot(omega_out);
        float cosNI = params->N.dot(omega_in);
        // reflect the view vector
        Vec3 R = (2 * cosNO) * params->N - omega_out;
        float out = cosNI * ((params->exponent + 2) * 0.5f * (float) M_1_PI * powf(R.dot(omega_in), params->exponent));
        labels = Labels(Labels::SURFACE | Labels::REFLECT | Labels::GLOSSY);
        return Color3 (out, out, out);
    }

    void sample (const void *paramsptr, const Vec3 &Ng,
                 const Vec3 &omega_out, float randu, float randv,
                 Vec3 &omega_in, float &pdf, Color3 &eval, Labels &labels) const
    {
        const params_t *params = (const params_t *) paramsptr;
        float cosNO = params->N.dot(omega_out);
        labels = Labels(Labels::SURFACE | Labels::REFLECT | Labels::GLOSSY);
        if (cosNO > 0) {
            // reflect the view vector
            Vec3 R = (2 * cosNO) * params->N - omega_out;
            Vec3 T, B;
            make_orthonormals (R, T, B);
            float phi = 2 * (float) M_PI * randu;
            float cosTheta = powf(randv, 1 / (params->exponent + 1));
            float sinTheta = sqrtf(1 - cosTheta * cosTheta);
            omega_in = (cosf(phi) * sinTheta) * T +
                       (sinf(phi) * sinTheta) * B +
                       (            cosTheta) * R;
            if ((Ng ^ omega_in) > 0.0f)
            {
                // common terms for pdf and eval
                float common = 0.5f * (float) M_1_PI * powf(R.dot(omega_in), params->exponent);
                float cosNI = params->N.dot(omega_in);
                float power;
                // make sure the direction we chose is still in the right hemisphere
                if (cosNI > 0)
                {
                    pdf = (params->exponent + 1) * common;
                    power = cosNI * (params->exponent + 2) * common;
                }
                else
                    power = pdf = 0.0f;
                eval.setValue(power, power, power);
                return;
            }
        }
        pdf = 0; 
        omega_in.setValue(0.0f, 0.0f, 0.0f);
        eval.setValue(0.0f, 0.0f, 0.0f);
    }

    float pdf (const void *paramsptr, const Vec3 &Ng,
               const Vec3 &omega_out, const Vec3 &omega_in) const
    {
        const params_t *params = (const params_t *) paramsptr;
        float cosNO = params->N.dot(omega_out);
        Vec3 R = (2 * cosNO) * params->N - omega_out;
        return (params->exponent + 1) * 0.5f * (float) M_1_PI * powf(R.dot(omega_in), params->exponent);
    }

};


// anisotropic ward - leaks energy at grazing angles
// see http://www.graphics.cornell.edu/~bjw/wardnotes.pdf 
class WardClosure : public BSDFClosure {
public:
    WardClosure () : BSDFClosure ("ward", "nvff") { }

    struct params_t {
        Vec3 N;
        Vec3 T;
        float ax, ay;
    };

    bool get_cone(const void *paramsptr,
                  const Vec3 &omega_out, Vec3 &axis, float &angle) const
    {
        const params_t *params = (const params_t *) paramsptr;
        float cosNO = params->N.dot(omega_out);
        if (cosNO > 0) {
            // we are viewing the surface from the same side as the normal
            axis = params->N;
            angle = (float) M_PI;
            return true;
        }
        // we are below the surface
        return false;
    }

    Color3 eval (const void *paramsptr, const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &omega_in, Labels &labels) const
    {
        const params_t *params = (const params_t *) paramsptr;
        float cosNO = params->N.dot(omega_out);
        float cosNI = params->N.dot(omega_in);
        if (cosNI * cosNO <= 0.0f)
           return Color3 (0,0,0);
        // get half vector and get x,y basis on the surface for anisotropy
        Vec3 H = omega_in + omega_out; // no need to normalize
        Vec3 X, Y;
        make_orthonormals(params->N, params->T, X, Y);
        // eq. 4
        float dotx = H.dot(X) / params->ax;
        float doty = H.dot(Y) / params->ay;
        float dotn = H.dot(params->N);
        float exp_arg = (dotx * dotx + doty * doty) / (dotn * dotn);
        float denom = (4 * (float) M_PI * params->ax * params->ay * sqrtf(cosNO * cosNI));
        float out = cosNI * expf(-exp_arg) / denom;
        labels = Labels(Labels::SURFACE | Labels::REFLECT | Labels::GLOSSY);
        return Color3 (out, out, out);
    }

    void sample (const void *paramsptr, const Vec3 &Ng,
                 const Vec3 &omega_out, float randu, float randv,
                 Vec3 &omega_in, float &pdf, Color3 &eval, Labels &labels) const
    {
        const params_t *params = (const params_t *) paramsptr;
        float cosNO = params->N.dot(omega_out);
        labels = Labels(Labels::SURFACE | Labels::REFLECT | Labels::GLOSSY);
        if (cosNO > 0) {
            // get x,y basis on the surface for anisotropy
            Vec3 X, Y;
            make_orthonormals(params->N, params->T, X, Y);
            // generate random angles for the half vector
            // eq. 7 (taking care around discontinuities to keep
            //        output angle in the right quadrant)
            // we take advantage of cos(atan(x)) == 1/sqrt(1+x^2)
            //                  and sin(atan(x)) == x/sqrt(1+x^2)
            float alphaRatio = params->ay / params->ax;
            float cosPhi, sinPhi;
            if (randu < 0.25f) {
                float val = 4 * randu;
                float tanPhi = alphaRatio * tanf((float) M_PI_2 * val);
                cosPhi = 1 / sqrtf(1 + tanPhi * tanPhi);
                sinPhi = tanPhi * cosPhi;
            } else if (randu < 0.5) {
                float val = 1 - 4 * (0.5f - randu);
                float tanPhi = alphaRatio * tanf((float) M_PI_2 * val);
                // phi = (float) M_PI - phi;
                cosPhi = -1 / sqrtf(1 + tanPhi * tanPhi);
                sinPhi = -tanPhi * cosPhi;
            } else if (randu < 0.75f) {
                float val = 4 * (randu - 0.5f);
                float tanPhi = alphaRatio * tanf((float) M_PI_2 * val);
                //phi = (float) M_PI + phi;
                cosPhi = -1 / sqrtf(1 + tanPhi * tanPhi);
                sinPhi = tanPhi * cosPhi;
            } else {
                float val = 1 - 4 * (1 - randu);
                float tanPhi = alphaRatio * tanf((float) M_PI_2 * val);
                // phi = 2 * (float) M_PI - phi;
                cosPhi = 1 / sqrtf(1 + tanPhi * tanPhi);
                sinPhi = -tanPhi * cosPhi;
            }
            // eq. 6
            // we take advantage of cos(atan(x)) == 1/sqrt(1+x^2)
            //                  and sin(atan(x)) == x/sqrt(1+x^2)
            float thetaDenom = (cosPhi * cosPhi) / (params->ax * params->ax) + (sinPhi * sinPhi) / (params->ay * params->ay);
            float tanTheta2 = -logf(1 - randv) / thetaDenom;
            float cosTheta  = 1 / sqrtf(1 + tanTheta2);
            float sinTheta  = cosTheta * sqrtf(tanTheta2);

            Vec3 h; // already normalized becaused expressed from spherical coordinates
            h.x = sinTheta * cosPhi;
            h.y = sinTheta * sinPhi;
            h.z = cosTheta;
            // compute terms that are easier in local space
            float dotx = h.x / params->ax;
            float doty = h.y / params->ay;
            float dotn = h.z;
            // transform to world space
            h = h.x * X + h.y * Y + h.z * params->N;
            // generate the final sample
            float oh = h.dot(omega_out);
            omega_in.x = 2 * oh * h.x - omega_out.x;
            omega_in.y = 2 * oh * h.y - omega_out.y;
            omega_in.z = 2 * oh * h.z - omega_out.z;
            if ((Ng ^ omega_in) > 0.0f) {
                // eq. 9
                float exp_arg = (dotx * dotx + doty * doty) / (dotn * dotn);
                float denom = 4 * (float) M_PI * params->ax * params->ay * oh * dotn * dotn * dotn;
                pdf = expf(-exp_arg) / denom;
                float cosNI = params->N ^ omega_in;
                // compiler will reuse expressions already computed
                denom = (4 * (float) M_PI * params->ax * params->ay * sqrtf(cosNO * cosNI));
                float power = cosNI * expf(-exp_arg) / denom;
                eval.setValue(power, power, power);
                return;
            }
        }
        pdf = 0;
        omega_in.setValue(0.0f, 0.0f, 0.0f);
        eval.setValue(0.0f, 0.0f, 0.0f);
    }

    float pdf (const void *paramsptr, const Vec3 &Ng,
               const Vec3 &omega_out, const Vec3 &omega_in) const
    {
        const params_t *params = (const params_t *) paramsptr;
        Vec3 H = omega_in + omega_out;
        H.normalize(); // needed for denominator
        Vec3 X, Y;
        make_orthonormals(params->N, params->T, X, Y);
        // eq. 9
        float dotx = H.dot(X) / params->ax;
        float doty = H.dot(Y) / params->ay;
        float dotn = H.dot(params->N);
        float exp_arg = (dotx * dotx + doty * doty) / (dotn * dotn);
        float denom = 4 * (float) M_PI * params->ax * params->ay * H.dot(omega_out) * dotn * dotn * dotn;
        return expf(-exp_arg) / denom;
    }
};


// microfacet model with GGX facet distribution
// see http://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf 
class MicrofacetGGXClosure : public BSDFClosure {
public:
    MicrofacetGGXClosure () : BSDFClosure ("microfacet_ggx", "nff") { }

    struct params_t {
        Vec3 N;
        float ag;   // width parameter (roughness)
        float R0;   // fresnel reflectance at incidence
    };

    bool get_cone(const void *paramsptr,
                  const Vec3 &omega_out, Vec3 &axis, float &angle) const
    {
        const params_t *params = (const params_t *) paramsptr;
        float cosNO = params->N.dot(omega_out);
        if (cosNO > 0) {
            // we are viewing the surface from the same side as the normal
            axis = params->N;
            angle = (float) M_PI;
            return true;
        }
        // we are below the surface
        return false;
    }

    Color3 eval (const void *paramsptr, const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &omega_in, Labels &labels) const
    {
        const params_t *params = (const params_t *) paramsptr;
        float cosNO = params->N.dot(omega_out);
        float cosNI = params->N.dot(omega_in);
        // get half vector
        Vec3 Hr = omega_in + omega_out;
        Hr.normalize();
        // eq. 20: (F*G*D)/(4*in*on)
        // eq. 33: first we calculate D(m) with m=Hr:
        float alpha2 = params->ag * params->ag;
        float cosThetaM = Hr.dot(params->N);
        float cosThetaM2 = cosThetaM * cosThetaM;
        float tanThetaM2 = (1 - cosThetaM2) / cosThetaM2;
        float cosThetaM4 = cosThetaM2 * cosThetaM2;
        float D = alpha2 / ((float) M_PI * cosThetaM4 * (alpha2 + tanThetaM2) * (alpha2 + tanThetaM2));
        // eq. 34: now calculate G1(i,m) and G1(o,m)
        float G1o = 2 / (1 + sqrtf(1 + alpha2 * (1 - cosNO * cosNO) / (cosNO * cosNO)));
        float G1i = 2 / (1 + sqrtf(1 + alpha2 * (1 - cosNI * cosNI) / (cosNI * cosNI))); 
        float G = G1o * G1i;
        // fresnel term between outgoing direction and microfacet
        float F = fresnel_shlick(Hr.dot(omega_out), params->R0);
        float out = (F * G * D) * 0.25f / cosNI;
        labels = Labels(Labels::SURFACE | Labels::REFLECT | Labels::GLOSSY);
        return Color3 (out, out, out);
    }

    void sample (const void *paramsptr, const Vec3 &Ng,
                 const Vec3 &omega_out, float randu, float randv,
                 Vec3 &omega_in, float &pdf, Color3 &eval, Labels &labels) const
    {
        const params_t *params = (const params_t *) paramsptr;
        float cosNO = params->N.dot(omega_out);
        labels = Labels(Labels::SURFACE | Labels::REFLECT | Labels::GLOSSY);
        if (cosNO > 0) {
            Vec3 X, Y;
            make_orthonormals(params->N, X, Y);
            // generate a random microfacet normal m
            // eq. 35,36:
            // we take advantage of cos(atan(x)) == 1/sqrt(1+x^2)
            //                  and sin(atan(x)) == x/sqrt(1+x^2)
            float alpha2 = params->ag * params->ag;
            float tanThetaM2 = alpha2 * randu / (1 - randu);
            float cosThetaM  = 1 / sqrtf(1 + tanThetaM2);
            float sinThetaM  = cosThetaM * sqrtf(tanThetaM2);
            float phiM = 2 * float(M_PI) * randv;
            Vec3 m = (cosf(phiM) * sinThetaM) * X +
                     (sinf(phiM) * sinThetaM) * Y +
                                   cosThetaM  * params->N;
            float cosMO = m.dot(omega_out);
            if (cosMO > 0) {
                // microfacet normal is visible to this ray
                // eq. 33
                float cosThetaM2 = cosThetaM * cosThetaM;
                float cosThetaM4 = cosThetaM2 * cosThetaM2;
                float D = alpha2 / (float(M_PI) * cosThetaM4 * (alpha2 + tanThetaM2) * (alpha2 + tanThetaM2));
                // eq. 24
                float pm = D * cosThetaM;
                // convert into pdf of the sampled direction
                // eq. 38 - but see also:
                // eq. 17 in http://www.graphics.cornell.edu/~bjw/wardnotes.pdf
                pdf = pm * 0.25f / cosMO;
                // eq. 39 - compute actual reflected direction
                omega_in = 2 * cosMO * m - omega_out;
                if ((Ng ^ omega_in) > 0.0f) {
                    float cosNI = params->N.dot(omega_in);
                    float G1o = 2 / (1 + sqrtf(1 + alpha2 * (1 - cosNO * cosNO) / (cosNO * cosNO)));
                    float G1i = 2 / (1 + sqrtf(1 + alpha2 * (1 - cosNI * cosNI) / (cosNI * cosNI))); 
                    float G = G1o * G1i;
                    float F = fresnel_shlick(m.dot(omega_out), params->R0);
                    float power = (F * G * D) * 0.25f / cosNI;
                    eval.setValue(power, power, power);
                    return;
                }
            }
        }
        pdf = 0; 
        omega_in.setValue(0.0f, 0.0f, 0.0f);
        eval.setValue(0.0f, 0.0f, 0.0f);
    }

    float pdf (const void *paramsptr, const Vec3 &Ng,
               const Vec3 &omega_out, const Vec3 &omega_in) const
    {
        const params_t *params = (const params_t *) paramsptr;
        // get microfacet normal m (half-vector)
        Vec3 m = omega_in + omega_out;
        m.normalize();
        float cosMO = m.dot(omega_out);
        if (cosMO > 0) {
            // eq. 33
            float cosThetaM = params->N.dot(m);
            float cosThetaM2 = cosThetaM * cosThetaM;
            float tanThetaM2 = (1 - cosThetaM2) / cosThetaM2;
            float cosThetaM4 = cosThetaM2 * cosThetaM2;
            float alpha2 = params->ag * params->ag;
            float D = alpha2 / (float(M_PI) * cosThetaM4 * (alpha2 + tanThetaM2) * (alpha2 + tanThetaM2));
            // eq. 24
            float pm = D * cosThetaM;
            // convert into pdf of the sampled direction
            // eq. 38 - but see also:
            // eq. 17 in http://www.graphics.cornell.edu/~bjw/wardnotes.pdf
            return pm * 0.25f / cosMO;
        }
        return 0;
    }
};


// microfacet model with Beckmann facet distribution
// see http://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf 
class MicrofacetBeckmannClosure : public BSDFClosure {
public:
    MicrofacetBeckmannClosure () : BSDFClosure ("microfacet_beckmann", "nff") { }

    struct params_t {
        Vec3 N;
        float ab;   // width parameter (roughness)
        float R0;   // fresnel reflectance at incidence
    };

    bool get_cone(const void *paramsptr,
                  const Vec3 &omega_out, Vec3 &axis, float &angle) const
    {
        const params_t *params = (const params_t *) paramsptr;
        float cosNO = params->N.dot(omega_out);
        if (cosNO > 0) {
            // we are viewing the surface from the same side as the normal
            axis = params->N;
            angle = (float) M_PI;
            return true;
        }
        // we are below the surface
        return false;
    }

    Color3 eval (const void *paramsptr, const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &omega_in, Labels &labels) const
    {
        const params_t *params = (const params_t *) paramsptr;
        float cosNO = params->N.dot(omega_out);
        float cosNI = params->N.dot(omega_in);
        // get half vector
        Vec3 Hr = omega_in + omega_out;
        Hr.normalize();
        // eq. 20: (F*G*D)/(4*in*on)
        // eq. 25: first we calculate D(m) with m=Hr:
        float alpha2 = params->ab * params->ab;
        float cosThetaM = Hr.dot(params->N);
        float cosThetaM2 = cosThetaM * cosThetaM;
        float tanThetaM2 = (1 - cosThetaM2) / cosThetaM2;
        float cosThetaM4 = cosThetaM2 * cosThetaM2;
        float D = expf(-tanThetaM2 / alpha2) / (float(M_PI) * alpha2 *  cosThetaM4);
        // eq. 26, 27: now calculate G1(i,m) and G1(o,m)
        float ao = 1 / (params->ab * sqrtf((1 - cosNO * cosNO) / (cosNO * cosNO)));
        float ai = 1 / (params->ab * sqrtf((1 - cosNI * cosNI) / (cosNI * cosNI)));
        float G1o = ao < 1.6f ? (3.535f * ao + 2.181f * ao * ao) / (1 + 2.276f * ao + 2.577f * ao * ao) : 1.0f;
        float G1i = ai < 1.6f ? (3.535f * ai + 2.181f * ai * ai) / (1 + 2.276f * ai + 2.577f * ai * ai) : 1.0f;
        float G = G1o * G1i;
        // fresnel term between outgoing direction and microfacet
        float F = fresnel_shlick(Hr.dot(omega_out), params->R0);
        float out = (F * G * D) * 0.25f / cosNI;
        labels = Labels(Labels::SURFACE | Labels::REFLECT | Labels::GLOSSY);
        return Color3 (out, out, out);
    }

    void sample (const void *paramsptr, const Vec3 &Ng,
                 const Vec3 &omega_out, float randu, float randv,
                 Vec3 &omega_in, float &pdf, Color3 &eval, Labels &labels) const
    {
        const params_t *params = (const params_t *) paramsptr;
        float cosNO = params->N.dot(omega_out);
        labels = Labels(Labels::SURFACE | Labels::REFLECT | Labels::GLOSSY);
        if (cosNO > 0) {
            Vec3 X, Y;
            make_orthonormals(params->N, X, Y);
            // generate a random microfacet normal m
            // eq. 35,36:
            // we take advantage of cos(atan(x)) == 1/sqrt(1+x^2)
            //                  and sin(atan(x)) == x/sqrt(1+x^2)
            float alpha2 = params->ab * params->ab;
            float tanThetaM = -alpha2 * logf(1 - randu);
            float cosThetaM = 1 / sqrtf(1 + tanThetaM * tanThetaM);
            float sinThetaM = cosThetaM * tanThetaM;
            float phiM = 2 * float(M_PI) * randv;
            Vec3 m = (cosf(phiM) * sinThetaM) * X +
                     (sinf(phiM) * sinThetaM) * Y +
                                   cosThetaM  * params->N;
            float cosMO = m.dot(omega_out);
            if (cosMO > 0) {
                // microfacet normal is visible to this ray
                // eq. 25
                float cosThetaM2 = cosThetaM * cosThetaM;
                float tanThetaM2 = tanThetaM * tanThetaM;
                float cosThetaM4 = cosThetaM2 * cosThetaM2;
                float D = expf(-tanThetaM2 / alpha2) / (float(M_PI) * alpha2 *  cosThetaM4);
                // eq. 24
                float pm = D * cosThetaM;
                // convert into pdf of the sampled direction
                // eq. 38 - but see also:
                // eq. 17 in http://www.graphics.cornell.edu/~bjw/wardnotes.pdf
                pdf = pm * 0.25f / cosMO;
                // eq. 39 - compute actual reflected direction
                omega_in = 2 * cosMO * m - omega_out;
                if ((Ng ^ omega_in) > 0.0f) {
                    float cosNI = params->N.dot(omega_in);
                    float ao = 1 / (params->ab * sqrtf((1 - cosNO * cosNO) / (cosNO * cosNO)));
                    float ai = 1 / (params->ab * sqrtf((1 - cosNI * cosNI) / (cosNI * cosNI)));
                    float G1o = ao < 1.6f ? (3.535f * ao + 2.181f * ao * ao) / (1 + 2.276f * ao + 2.577f * ao * ao) : 1.0f;
                    float G1i = ai < 1.6f ? (3.535f * ai + 2.181f * ai * ai) / (1 + 2.276f * ai + 2.577f * ai * ai) : 1.0f;
                    float G = G1o * G1i;
                    float F = fresnel_shlick(m.dot(omega_out), params->R0);
                    float power = (F * G * D) * 0.25f / cosNI;
                    eval.setValue(power, power, power);
                    return;
                }
            }
        }
        pdf = 0;
        omega_in.setValue(0.0f, 0.0f, 0.0f);
        eval.setValue(0.0f, 0.0f, 0.0f);
    }

    float pdf (const void *paramsptr, const Vec3 &Ng,
               const Vec3 &omega_out, const Vec3 &omega_in) const
    {
        const params_t *params = (const params_t *) paramsptr;
        // get microfacet normal m (half-vector)
        Vec3 m = omega_in + omega_out;
        m.normalize();
        float cosMO = m.dot(omega_out);
        if (cosMO > 0) {
            // eq. 25
            float alpha2 = params->ab * params->ab;
            float cosThetaM = params->N.dot(m);
            float cosThetaM2 = cosThetaM * cosThetaM;
            float tanThetaM2 = (1 - cosThetaM2) / cosThetaM2;
            float cosThetaM4 = cosThetaM2 * cosThetaM2;
            float D = expf(-tanThetaM2 / alpha2) / (float(M_PI) * alpha2 *  cosThetaM4);
            // eq. 24
            float pm = D * cosThetaM;
            // convert into pdf of the sampled direction
            // eq. 38 - but see also:
            // eq. 17 in http://www.graphics.cornell.edu/~bjw/wardnotes.pdf
            return pm * 0.25f / cosMO;
        }
        return 0;
    }
};

// these are all singletons
DiffuseClosure diffuse_closure_primitive;
TransparentClosure transparent_closure_primitive;
PhongClosure phong_closure_primitive;
WardClosure ward_closure_primitive;
MicrofacetGGXClosure microfacet_ggx_closure;
MicrofacetBeckmannClosure microfacet_beckmann_closure;


}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
