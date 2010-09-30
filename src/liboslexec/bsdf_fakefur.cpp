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

#include "genclosure.h"
#include "oslops.h"
#include "oslexec_pvt.h"

// based on the implementation by Dan B. Goldman
// http://www.danbgoldman.com/misc/fakefur/fakefur.pdf


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
namespace pvt {


inline float smoothstep(float edge0, float edge1, float x) {
    float result;
    if (x < edge0) result = 0.0f;
    else if (x >= edge1) result = 1.0f;
    else {
        float t = (x - edge0)/(edge1 - edge0);
        result = (3.0f-2.0f*t)*(t*t);
    }
    return result;
}

inline float furOpacity(float cosNI, float cosTI, 
				        float density, float averageRadius, 
                        float length)
{
	float area  = length*(averageRadius)/2.f;
	float g = sqrtf(std::max(1.f-cosTI*cosTI, 0.f))/cosNI; 
	float interp = 1.f-(1.f/expf(density*area*g)); 
	return 1.f-smoothstep(0.f, 1.f, interp);      
}


class FakefurDiffuseClosure : public BSDFClosure {
public:
    Vec3 m_N;
    Vec3 m_T;

    // fake fur fur illumination related stuff
    float m_fur_reflectivity;
    float m_fur_transmission;

    float m_shadow_start;
    float m_shadow_end;

    float m_fur_attenuation;

    // fake fur opacity function related stuff...
    float m_fur_density;
    float m_fur_avg_radius;
    float m_fur_length;

    float m_fur_shadow_fraction;

    FakefurDiffuseClosure() : BSDFClosure(Labels::DIFFUSE) { }

    void setup() {};

    bool mergeable (const ClosurePrimitive *other) const {
        const FakefurDiffuseClosure *comp = (const FakefurDiffuseClosure *)other;
        return m_N == comp->m_N && m_T == comp->m_T &&
            m_fur_reflectivity == comp->m_fur_reflectivity &&
            m_fur_transmission == comp->m_fur_transmission &&
            m_shadow_start == comp->m_shadow_start &&
            m_shadow_end == comp->m_shadow_end &&
            m_fur_attenuation == comp->m_fur_attenuation &&
            m_fur_density == comp->m_fur_density &&
            m_fur_avg_radius == comp->m_fur_avg_radius &&
            m_fur_length == comp->m_fur_length &&
            m_fur_shadow_fraction == comp->m_fur_shadow_fraction &&
            BSDFClosure::mergeable(other);
    }

    size_t memsize () const { return sizeof(*this); }

    const char *name () const { return "fakefur_diffuse"; }

    void print_on (std::ostream &out) const
    {
        out << "fakefur_diffuse_N ((" << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << ")";
        out << "fakefur_diffuse_T ((" << m_T[0] << ", " << m_T[1] << ", " << m_T[2] << "))";
    }

    float albedo (const Vec3 &omega_out) const
    {
        return 1.0f;
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {
        // T from fur tangent map is expected to be in world space
        // 
        // fake fur illumination
        Vec3 xTO = m_T.cross(omega_out);
        Vec3 xTI = m_T.cross(omega_in);      
        float kappa = xTI.dot(xTO);

        float sigmaDir  = (1.f+kappa)*0.5f * m_fur_reflectivity;
        sigmaDir += (1.f-kappa)*0.5f * m_fur_transmission;

        float cosNI = m_N.dot(omega_in);
        float sigmaSurfaceA = smoothstep(m_shadow_start, m_shadow_end, cosNI);
        float sigmaSurface = m_fur_attenuation * sigmaSurfaceA;
    
        float furIllum = sigmaDir * sigmaSurface;

        float cosTI = m_T.dot(omega_in);
        // fur over fur opacity http
        float furOpac = 1.f-m_fur_shadow_fraction*furOpacity(cosNI, cosTI, m_fur_density, 
                                                             m_fur_avg_radius, m_fur_length);
        
        float cos_a = m_T.dot(omega_in);
        float bsdf = sqrtf(std::max(1 - cos_a*cos_a, 0.0f)) * (float) (M_1_PI * M_1_PI);
        bsdf *= furIllum * furOpac;

        float cos_pi = std::max(cosNI,0.0f) * (float) M_1_PI;
        pdf = cos_pi;

        return Color3 (bsdf, bsdf, bsdf);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {
       return Color3 (0, 0, 0);
    }

    ustring sample (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                 float randu, float randv,
                 Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                 float &pdf, Color3 &eval) const
    {
        // we are viewing the surface from the right side - send a ray out with cosine
        // distribution over the hemisphere
        sample_cos_hemisphere (m_N, omega_out, randu, randv, omega_in, pdf);

        if (Ng.dot(omega_in) > 0) {
            Vec3 xTO = m_T.cross(omega_out);
            Vec3 xTI = m_T.cross(omega_in);      
            float kappa = xTI.dot(xTO);

            float sigmaDir  = (1.f+kappa)*0.5f * m_fur_reflectivity;
            sigmaDir += (1.f-kappa)*0.5f * m_fur_transmission;

            float cosNI = m_N.dot(omega_in);
            float sigmaSurfaceA = smoothstep(m_shadow_start, m_shadow_end, cosNI);
            float sigmaSurface = m_fur_attenuation * sigmaSurfaceA;
    
            float furIllum = sigmaDir * sigmaSurface;

            float cosTI = m_T.dot(omega_in);
            // fake fur over skin opacity 
            float furOpac = 1.f-furOpacity(cosNI, cosTI, m_fur_density, 
                                    m_fur_avg_radius, m_fur_length);

            float result = pdf * furOpac * furIllum;
            eval.setValue(result, result, result);
            // TODO: find a better approximation for the diffuse bounce
            domega_in_dx = (2 * m_N.dot(domega_out_dx)) * m_N - domega_out_dx;
            domega_in_dy = (2 * m_N.dot(domega_out_dy)) * m_N - domega_out_dy;
            domega_in_dx *= 125;
            domega_in_dy *= 125;
        } else
             pdf = 0.0f;
        return Labels::NONE;
    }
};



class FakefurSpecularClosure : public BSDFClosure {
public:

    Vec3 m_N;
    Vec3 m_T;
    float m_offset, m_cos_off, m_sin_off;
    float m_exp;

    // fake fur fur illumination related stuff
    float m_fur_reflectivity;
    float m_fur_transmission;

    float m_shadow_start;
    float m_shadow_end;

    float m_fur_attenuation;

    // fake fur opacity function related stuff...
    float m_fur_density;
    float m_fur_avg_radius;
    float m_fur_length;   

    float m_fur_shadow_fraction; 

    FakefurSpecularClosure() : BSDFClosure(Labels::GLOSSY) { }

    void setup()
    {
        m_cos_off = cosf(m_offset);
        m_sin_off = sinf(m_offset);
    }

    bool mergeable (const ClosurePrimitive *other) const {
        const FakefurSpecularClosure *comp = (const FakefurSpecularClosure *)other;
        return m_N == comp->m_N && m_T == comp->m_T &&
            m_offset == comp->m_offset && m_exp == comp->m_exp &&
            m_fur_reflectivity == comp->m_fur_reflectivity &&
            m_fur_transmission == comp->m_fur_transmission &&
            m_shadow_start == comp->m_shadow_start &&
            m_shadow_end == comp->m_shadow_end &&
            m_fur_attenuation == comp->m_fur_attenuation &&
            m_fur_density == comp->m_fur_density &&
            m_fur_avg_radius == comp->m_fur_avg_radius &&
            m_fur_length == comp->m_fur_length &&
            BSDFClosure::mergeable(other);
    }

    size_t memsize () const { return sizeof(*this); }

    const char *name () const { return "fakefur_specular"; }

    void print_on (std::ostream &out) const
    {
        out << "fakefur_specular_N ((" << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << ")";
        out << "fakefur_specular_T ((" << m_T[0] << ", " << m_T[1] << ", " << m_T[2] << "), " << m_offset << ")";
    }

    float albedo (const Vec3 &omega_out) const
    {
        // we don't know how to sample this
        return 0.0f;
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {

        // T from fur tangent map is expected to be in world space
        // 
        // fake fur illumination
        Vec3 xTO = m_T.cross(omega_out);
        Vec3 xTI = m_T.cross(omega_in);      
        float kappa = xTI.dot(xTO);

        float sigmaDir  = (1.f+kappa)*0.5f * m_fur_reflectivity;
        sigmaDir += (1.f-kappa)*0.5f * m_fur_transmission;

        float cosNI = m_N.dot(omega_in);
    
        float sigmaSurfaceA = smoothstep(m_shadow_start, m_shadow_end, cosNI);
        float sigmaSurface = m_fur_attenuation * sigmaSurfaceA;
    
        float furIllum = sigmaDir * sigmaSurface;

        float cos_i = m_T.dot(omega_in);
        // fur over fur opacity 
        float furOpac = 1.f-m_fur_shadow_fraction*furOpacity(cosNI, cos_i, m_fur_density, 
                                                             m_fur_avg_radius, m_fur_length);
        
        //float angle_i = acosf(m_T.dot(omega_in));
        //float angle_o = M_PI - (acosf(m_T.dot(omega_out)) + m_offset);
        //float cos_diff = cosf(angle_i - angle_o);
        //
        // Optimized version of the above commented code
        
        float cos_o = m_T.dot(omega_out);
        
        float sin_i = sqrtf (std::max (1 - cos_i*cos_i, 0.0f));
        float sin_o = sqrtf (std::max (1 - cos_o*cos_o, 0.0f));
        float cos_diff = sin_i * sin_o * m_cos_off +
                         sin_i * cos_o * m_sin_off +
                         cos_i * sin_o * m_sin_off -
                         cos_i * cos_o * m_cos_off;
         
        // TODO: normalization? ha!
        float bsdf = cos_diff > 0.0f ? powf(cos_diff, m_exp) : 0.0f;
        bsdf *= furIllum * furOpac;
        bsdf *= (float) (M_1_PI * M_1_PI);
        pdf = 0.0f;
        return Color3 (bsdf, bsdf, bsdf);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {
       return Color3 (0, 0, 0);
    }

    ustring sample (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                 float randu, float randv,
                 Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                 float &pdf, Color3 &eval) const
    {
        // TODO: we don't know how to do this, sorry
        pdf = 0.0f;
        return Labels::NONE;
    }
};


class FakefurSkinClosure : public BSDFClosure {
public:

    Vec3 m_N;
    Vec3 m_T;

    // fake fur fur illumination related stuff
    float m_fur_reflectivity;
    float m_fur_transmission;

    float m_shadow_start;
    float m_shadow_end;

    float m_fur_attenuation;

    // fake fur opacity function related stuff...
    float m_fur_density;
    float m_fur_avg_radius;
    float m_fur_length;    
  
    FakefurSkinClosure() : BSDFClosure(Labels::DIFFUSE) { }

    void setup() {};

    bool mergeable (const ClosurePrimitive *other) const {
        const FakefurSkinClosure *comp = (const FakefurSkinClosure *)other;
        return m_N == comp->m_N && m_T == comp->m_T &&
            m_fur_reflectivity == comp->m_fur_reflectivity &&
            m_fur_transmission == comp->m_fur_transmission &&
            m_shadow_start == comp->m_shadow_start &&
            m_shadow_end == comp->m_shadow_end &&
            m_fur_attenuation == comp->m_fur_attenuation &&
            m_fur_density == comp->m_fur_density &&
            m_fur_avg_radius == comp->m_fur_avg_radius &&
            m_fur_length == comp->m_fur_length &&
            BSDFClosure::mergeable(other);
    }

    size_t memsize () const { return sizeof(*this); }

    const char *name () const { return "fakefur_skin"; }

    void print_on (std::ostream &out) const
    {
        out << "fakefur_skin_N ((" << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << "))";
        out << "fakefur_skin_T ((" << m_T[0] << ", " << m_T[1] << ", " << m_T[2] << "), ";
    }

    float albedo (const Vec3 &omega_out) const
    {
        return 1.0f;
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {

        // T from fur tangent map is expected to be in world space
        // 
        // fake fur illumination
        Vec3 xTO = m_T.cross(omega_out);
        Vec3 xTI = m_T.cross(omega_in);      
        float kappa = xTI.dot(xTO);

        float sigmaDir  = (1.f+kappa)*0.5f * m_fur_reflectivity;
        sigmaDir += (1.f-kappa)*0.5f * m_fur_transmission;

        float cosNI = m_N.dot(omega_in);
        float sigmaSurfaceA = smoothstep(m_shadow_start, m_shadow_end, cosNI);
        float sigmaSurface = m_fur_attenuation * sigmaSurfaceA;
    
        float furIllum = sigmaDir * sigmaSurface;

        float cosTI = m_T.dot(omega_in);
        // fake fur over skin opacity 
        float furOpac = 1.f-furOpacity(cosNI, cosTI, m_fur_density, 
                                       m_fur_avg_radius, m_fur_length);
 
        float cos_pi = std::max(cosNI,0.0f) * (float) M_1_PI;

        pdf = cos_pi;
        cos_pi *= furIllum * furOpac;

        return Color3 (cos_pi, cos_pi, cos_pi);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {
       return Color3 (0, 0, 0);
    }

    ustring sample (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                 float randu, float randv,
                 Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                 float &pdf, Color3 &eval) const
    {
        // we are viewing the surface from the right side - send a ray out with cosine
        // distribution over the hemisphere
        sample_cos_hemisphere (m_N, omega_out, randu, randv, omega_in, pdf);

        if (Ng.dot(omega_in) > 0) {
            Vec3 xTO = m_T.cross(omega_out);
            Vec3 xTI = m_T.cross(omega_in);      
            float kappa = xTI.dot(xTO);

            float sigmaDir  = (1.f+kappa)*0.5f * m_fur_reflectivity;
            sigmaDir += (1.f-kappa)*0.5f * m_fur_transmission;

            float cosNI = m_N.dot(omega_in);
            float sigmaSurfaceA = smoothstep(m_shadow_start, m_shadow_end, cosNI);
            float sigmaSurface = m_fur_attenuation * sigmaSurfaceA;
    
            float furIllum = sigmaDir * sigmaSurface;

            float cosTI = m_T.dot(omega_in);
            // fake fur over skin opacity 
            float furOpac = 1.f-furOpacity(cosNI, cosTI, m_fur_density, 
                                    m_fur_avg_radius, m_fur_length);

            float result = pdf * furOpac * furIllum;
            eval.setValue(result, result, result);
            // TODO: find a better approximation for the diffuse bounce
            domega_in_dx = (2 * m_N.dot(domega_out_dx)) * m_N - domega_out_dx;
            domega_in_dy = (2 * m_N.dot(domega_out_dy)) * m_N - domega_out_dy;
            domega_in_dx *= 125;
            domega_in_dy *= 125;
        } else
             pdf = 0.0f;
        return Labels::NONE;
    }
};



ClosureParam bsdf_fakefur_diffuse_params[] = {
    CLOSURE_VECTOR_PARAM(FakefurDiffuseClosure, m_N),
    CLOSURE_VECTOR_PARAM(FakefurDiffuseClosure, m_T),
    CLOSURE_FLOAT_PARAM (FakefurDiffuseClosure, m_fur_reflectivity),
    CLOSURE_FLOAT_PARAM (FakefurDiffuseClosure, m_fur_transmission),
    CLOSURE_FLOAT_PARAM (FakefurDiffuseClosure, m_shadow_start),
    CLOSURE_FLOAT_PARAM (FakefurDiffuseClosure, m_shadow_end),
    CLOSURE_FLOAT_PARAM (FakefurDiffuseClosure, m_fur_attenuation),
    CLOSURE_FLOAT_PARAM (FakefurDiffuseClosure, m_fur_density),
    CLOSURE_FLOAT_PARAM (FakefurDiffuseClosure, m_fur_avg_radius),
    CLOSURE_FLOAT_PARAM (FakefurDiffuseClosure, m_fur_length),
    CLOSURE_FLOAT_PARAM (FakefurDiffuseClosure, m_fur_shadow_fraction),
    CLOSURE_STRING_KEYPARAM("label"),
    CLOSURE_FINISH_PARAM(FakefurDiffuseClosure) };

ClosureParam bsdf_fakefur_specular_params[] = {
    CLOSURE_VECTOR_PARAM(FakefurSpecularClosure, m_N),
    CLOSURE_VECTOR_PARAM(FakefurSpecularClosure, m_T),
    CLOSURE_FLOAT_PARAM (FakefurSpecularClosure, m_offset),
    CLOSURE_FLOAT_PARAM (FakefurSpecularClosure, m_exp),
    CLOSURE_FLOAT_PARAM (FakefurSpecularClosure, m_fur_reflectivity),
    CLOSURE_FLOAT_PARAM (FakefurSpecularClosure, m_fur_transmission),
    CLOSURE_FLOAT_PARAM (FakefurSpecularClosure, m_shadow_start),
    CLOSURE_FLOAT_PARAM (FakefurSpecularClosure, m_shadow_end),
    CLOSURE_FLOAT_PARAM (FakefurSpecularClosure, m_fur_attenuation), 
    CLOSURE_FLOAT_PARAM (FakefurSpecularClosure, m_fur_density),
    CLOSURE_FLOAT_PARAM (FakefurSpecularClosure, m_fur_avg_radius),
    CLOSURE_FLOAT_PARAM (FakefurSpecularClosure, m_fur_length),
    CLOSURE_FLOAT_PARAM (FakefurSpecularClosure, m_fur_shadow_fraction),
    CLOSURE_STRING_KEYPARAM("label"),
    CLOSURE_FINISH_PARAM(FakefurSpecularClosure) };

ClosureParam bsdf_fakefur_skin_params[] = {
    CLOSURE_VECTOR_PARAM(FakefurSkinClosure, m_N),
    CLOSURE_VECTOR_PARAM(FakefurSkinClosure, m_T),
    CLOSURE_FLOAT_PARAM (FakefurSkinClosure, m_fur_reflectivity),
    CLOSURE_FLOAT_PARAM (FakefurSkinClosure, m_fur_transmission),
    CLOSURE_FLOAT_PARAM (FakefurSkinClosure, m_shadow_start),
    CLOSURE_FLOAT_PARAM (FakefurSkinClosure, m_shadow_end),
    CLOSURE_FLOAT_PARAM (FakefurSkinClosure, m_fur_attenuation),
    CLOSURE_FLOAT_PARAM (FakefurSkinClosure, m_fur_density),
    CLOSURE_FLOAT_PARAM (FakefurSkinClosure, m_fur_avg_radius),
    CLOSURE_FLOAT_PARAM (FakefurSkinClosure, m_fur_length),
    CLOSURE_STRING_KEYPARAM("label"),
    CLOSURE_FINISH_PARAM(FakefurSkinClosure) };

CLOSURE_PREPARE(bsdf_fakefur_diffuse_prepare,  FakefurDiffuseClosure)
CLOSURE_PREPARE(bsdf_fakefur_specular_prepare, FakefurSpecularClosure)
CLOSURE_PREPARE(bsdf_fakefur_skin_prepare,     FakefurSkinClosure)

}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
