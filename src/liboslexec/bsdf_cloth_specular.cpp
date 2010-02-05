/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
fmodification, are permitted provided that the following conditions are
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
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO evenT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, even IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cmath>
#include <OpenEXR/ImathMatrix.h>
#include <OpenEXR/ImathFun.h>

#include "oslops.h"
#include "oslexec_pvt.h"

#include "bsdf_cloth_fncs.h"

inline void set_ellipse_axes(Point2 &semimajor, Point2 &semiminor, float alpha, float eccentricity, float angle)
{
    semimajor.x = alpha;
    semimajor.y = 0.f;
    semiminor.x = 0.f;
    semiminor.y = semimajor.x * sqrtf(1.f-eccentricity*eccentricity);

    // rotate ellipse
    rotate_2D(semimajor, angle);
    rotate_2D(semiminor, angle);
}

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
namespace pvt {


enum threadType
{
    LONGWARP,
    SHORTWARP,
    LONGWEFT,
    SHORTWEFT
};


class ClothSpecularClosure : public BSDFClosure {
    Vec3 m_N;

    Color3 m_spec_col[4];
    float m_eta[4];

    int    m_thread_pattern[4];
    float  m_pattern_weight[4];
    int    m_current_thread;

    float  m_brdf_interp;
    float  m_btf_interp;

    float m_uux;
    float m_vvx;

    float m_area_scaled;  // effective filter width for highlight ellipse
    Vec3 m_dPdu;          // informs highlight ellipse direction on surface

    float m_eccentricity[4];
    float m_angle[4];
    float m_Kx[4];
    float m_Ky[4];
    float m_Sx[4];
    float m_Sy[4];
           
private:
    int threadTypeCount;
    float alpha[4];
    float AR[4];
    Point2 semiminor[4], semimajor[4];

public:
    CLOSURE_CTOR (ClothSpecularClosure) : BSDFClosure(side, Labels::DIFFUSE)
    {
        CLOSURE_FETCH_ARG (m_N, 1);

        CLOSURE_FETCH_ARG_ARRAY (m_spec_col, 4, 2);
        CLOSURE_FETCH_ARG_ARRAY (m_eta, 4, 3);

        CLOSURE_FETCH_ARG_ARRAY (m_thread_pattern, 4, 4); 
        CLOSURE_FETCH_ARG_ARRAY (m_pattern_weight, 4, 5); 
        CLOSURE_FETCH_ARG (m_current_thread, 6);

        CLOSURE_FETCH_ARG (m_brdf_interp, 7);
        CLOSURE_FETCH_ARG (m_btf_interp, 8);

        CLOSURE_FETCH_ARG (m_uux, 9);  
        CLOSURE_FETCH_ARG (m_vvx, 10);  

        CLOSURE_FETCH_ARG (m_area_scaled, 11);
        CLOSURE_FETCH_ARG (m_dPdu, 12);

        CLOSURE_FETCH_ARG_ARRAY (m_eccentricity, 4, 13);
        CLOSURE_FETCH_ARG_ARRAY (m_angle, 4, 14);
        CLOSURE_FETCH_ARG_ARRAY (m_Kx, 4, 15);
        CLOSURE_FETCH_ARG_ARRAY (m_Ky, 4, 16);
        CLOSURE_FETCH_ARG_ARRAY (m_Sx, 4, 17);
        CLOSURE_FETCH_ARG_ARRAY (m_Sy, 4, 18);

        threadTypeCount = 0; 
    
        for(int i = 0; i < 4; i++){
            if(m_thread_pattern[i] != -1)
                threadTypeCount++;

            alpha[i] = m_Sy[i] * 0.5f;
            AR[i] = m_Sx[i]*m_Sy[i];
            set_ellipse_axes(semimajor[i],semiminor[i], alpha[i], m_eccentricity[i], m_angle[i]);   
        }         
    }

    void print_on (std::ostream &out) const
    {
        out << "cloth (";
        out << "(" << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << "), ";

        out << m_area_scaled << ",";
        out << "(" << m_dPdu[0] << ", " << m_dPdu[1] << ", " << m_dPdu[2] << "), ";

        out << ")";
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float normal_sign, float &pdf) const
    {
        Vec3 Nn = m_N * normal_sign;
        float cosNI = Nn.dot(omega_in);
        float cosNO = Nn.dot(omega_out);

        if (!(cosNI > 0 && cosNO > 0))
            return Color3 (0, 0, 0);

        Vec3 H = omega_in + omega_out;
        H.normalize();
        Point2 H2 = H_projected(H, Nn, m_dPdu);

        // per thread segment fresnel
        float F[4];
        F[LONGWARP]  = fresnel_dielectric(cosNO, m_eta[LONGWARP]);
        F[SHORTWARP] = F[LONGWARP];
        F[LONGWEFT]  = fresnel_dielectric(cosNO, m_eta[LONGWEFT]);
        F[SHORTWEFT] = F[LONGWEFT];
               
        Color3 AE_st (0, 0, 0);   // specular area, current single thread segment
        Color3 AE_f  (0, 0, 0);   // specular area, filtered aggregate <-- filtered spec
        Color3 brdf_spec (0, 0, 0);

        // stuff for brdf_spec derived from btf_spec "thread segment" highlight ellipse area here...       
        if(m_btf_interp > 0) {
            for(int i=0; i<threadTypeCount; i++)
            {
                Point2 rect[4];
                rect[0].x=0;
                rect[0].y=0;

                rect[1].x=0;
                rect[1].y=1*m_Sy[m_thread_pattern[i]];

                rect[2].x=1*m_Sx[m_thread_pattern[i]];
                rect[2].y=1*m_Sy[m_thread_pattern[i]];

                rect[3].x=1*m_Sx[m_thread_pattern[i]];
                rect[3].y=0;

                // get ellipse center for 2D half vector
                Point2 center = ellipse_center(m_Sx[m_thread_pattern[i]], m_Sy[m_thread_pattern[i]],
                                               m_Kx[m_thread_pattern[i]], m_Ky[m_thread_pattern[i]], H2);

                // determinant of 2x2  (ellipse axes)
                float det = semimajor[m_thread_pattern[i]].x*semiminor[m_thread_pattern[i]].y
                          - semimajor[m_thread_pattern[i]].y*semiminor[m_thread_pattern[i]].x;
                float inv = 1.f/det;

                Point2 smaj_inv, smin_inv;
                 // cheap inverse of 2x2
                smin_inv.y = inv*semiminor[m_thread_pattern[i]].y;
                smaj_inv.y = inv*-semimajor[m_thread_pattern[i]].y;

                smin_inv.x = inv*-semiminor[m_thread_pattern[i]].x;
                smaj_inv.x = inv*semimajor[m_thread_pattern[i]].x;

                // xform highlight
                Point2 highlight_xfmd;
                highlight_xfmd.x = smin_inv.y*center.x+smin_inv.x*center.y;
                highlight_xfmd.y = smaj_inv.y*center.x+smaj_inv.x*center.y;

                // now move window relative to highlight and pass to compute_AC to get highlight area
                Point2 rect_xfmd[4];
                rect_xfmd[1].x = -highlight_xfmd.x;
                rect_xfmd[1].y = -highlight_xfmd.y;

                rect_xfmd[2].x = (smin_inv.x*rect[1].y) - highlight_xfmd.x;
                rect_xfmd[2].y = (smaj_inv.x*rect[1].y) - highlight_xfmd.y;

                rect_xfmd[3].x = (smin_inv.y*rect[2].x + smin_inv.x*rect[2].y) - highlight_xfmd.x;
                rect_xfmd[3].y = (smaj_inv.y*rect[2].x + smaj_inv.x*rect[2].y) - highlight_xfmd.y;

                rect_xfmd[0].x = (smin_inv.y*rect[3].x) - highlight_xfmd.x;
                rect_xfmd[0].y = (smaj_inv.y*rect[3].x) - highlight_xfmd.y;

                // It's far easier to tell if we're inside or outside of the untransformed rectangle
                // instead of testing to see if the circle center is inside the transformed rectangle, which
                // is generally speaking, a parallelogram.
                // SO, pass boolean "isOutside" as hint to computeAC()
                bool isOutside = (center.x < rect[0].x || center.y < rect[0].y || center.x > rect[2].x || center.y > rect[2].y);

                float AC = compute_AC(rect_xfmd, isOutside);

                // ellipse segment area
                // apply per-thread type fresnel here
                float AE = (fabs(det)*AC)/AR[m_thread_pattern[i]] * F[m_thread_pattern[i]];

                //  grab the current single thread ellipse segment area in AE_st
                //  - to be used in LERP with AE_f (for when feature scale is under Nyquist limit (hopefully))
                //
                if(m_thread_pattern[i]== m_current_thread)
                    AE_st = AE * m_spec_col[m_thread_pattern[i]];
                
                // filtered AE and diffuse
                AE_f += AE * m_spec_col[m_thread_pattern[i]] * m_pattern_weight[m_thread_pattern[i]];

                }  // end for threadTypeCount
                // blend between single thread and antialiased, multiple thread segment BRDF
                AE_f = Imath::lerp(AE_st, AE_f, m_brdf_interp);
            } // end if for BRDF

        float G = computeG_Smith(Nn, H, omega_out, cosNI, cosNO);
        Color3 btf_spec (0, 0, 0);

        if(m_btf_interp < 1.f){    
            //get foci of ellipse
            Point2 F1, F2;
            Point2 center = ellipse_center(m_Sx[m_current_thread], m_Sy[m_current_thread],
                                           m_Kx[m_current_thread], m_Ky[m_current_thread], H2);

            ellipse_foci(semimajor[m_current_thread], m_eccentricity[m_current_thread],
                         center, &F1, &F2);

            // check to see if current shaded point (uu,vv) is inside the ellipse
            float I = inside_ellipse(F1, F2, m_uux*m_Sx[m_current_thread], m_vvx*m_Sy[m_current_thread],
                               alpha[m_current_thread], m_area_scaled);

            // 'I' replaces slope distribution term 'D' found in generalized cookTorrance microfacet model
            // Unlike 'D' in the canonical microfacet model which is 1.0/steradians, our term is 0->1 by design.
            float se = (I*G*F[m_current_thread]) / cosNO;
            btf_spec = Color3(se, se, se) * m_spec_col[m_current_thread];
        }

        brdf_spec = AE_f*G; // F is already figured in AE_f, above...
        brdf_spec /= cosNO;

        Color3 out;
        out = Imath::lerp(btf_spec, brdf_spec, m_btf_interp); 

        pdf = 0.f; 

        return Color3 (out[0], out[1], out[2]);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float normal_size, float &pdf) const
    {
        return Color3 (0.f, 0.f, 0.f);
    }

    ustring sample (const Vec3 &Ng,
                    const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                    float randu, float randv,
                    Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                    float &pdf, Color3 &eval) const
    {

        return Labels::REFLECT;
    }

};

DECLOP (OP_cloth_specular)
{
    closure_op_guts<ClothSpecularClosure, 19> (exec, nargs, args,
            runflags, beginpoint, endpoint);
}

}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif


