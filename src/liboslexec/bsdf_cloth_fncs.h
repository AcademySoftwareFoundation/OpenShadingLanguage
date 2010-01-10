/*
Copyright (c) 2010 Sony Pictures Imageworks, et al.
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



#include "oslops.h"
#include "oslexec_pvt.h"
#include <OpenEXR/ImathMatrix.h>


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {

const float EPSILON  = 1.0e-4;

float computeG_Smith(const Vec3 &N, Vec3 &H, const Vec3 &omega_out, float cosNI, float cosNO);


// stuff for "I" term which replaces microfacet slope term "D" in generalized microfacet model
// follows...

struct Point2
{
    float x;
    float y;

    Point2 () { }
    Point2 (float x_, float y_) : x(x_), y(y_) { }
};

const Point2 P2_ZERO (0.f, 0.f);

typedef struct Intersection
{
    Point2 p;
    int edge;
} Intersection;

float smoothstep(float edge0, float edge1, float x);

float schlick_fresnel(float cosNO, float R0);

Point2 H_projected(Vec3 &H, const Vec3 &N, const Vec3 &dPdu);

Point2 ellipse_center(float Sx, float Sy, float Kx, float Ky, Point2 H2);

void rotate_2D(Point2 &point, float angle, Point2 origin = P2_ZERO);

void ellipse_foci(Point2 alpha, float eta, Point2 center, Point2 *F1, Point2 *F2);

float inside_ellipse(Point2 F1, Point2 F2, float uu, float vv, float alpha, float width);

bool intersect_circle_segment(Point2 center, float radius, Point2 p1, Point2 p2);

bool ray_circle(Point2 p1, Point2 p2, Point2 sc, float r, float *mu1, float *mu2);

Point2 point_on_line(float mu, Point2 p1, Point2 p2);

inline float seg_area(float theta);

inline float t_area(Point2 PO, Point2 P1, Point2 P2);

// remember: it's atan2(y,x) not atan2(x,y)
inline float atan2_zero_to_pi(float y, float x);

//float compute_AC(Point2 highlight, Point2 *rect3, bool OUTSIDE);
float compute_AC(Point2 *rect3, bool OUTSIDE);

}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
using namespace OSL_NAMESPACE;
#endif

