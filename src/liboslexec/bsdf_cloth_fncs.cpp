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


#include <math.h>

#include "bsdf_cloth_fncs.h"

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {

/*
float
lerp(float t, float a, float b) {
    return (1.0f - t) * a + t * b;
}
*/

float
smoothstep(float edge0, float edge1, float x) {
    float result;
    if (x < edge0) result = 0.0f;
    else if (x >= edge1) result = 1.0f;
    else {
        float t = (x - edge0)/(edge1 - edge0);
        result = (3.0f-2.0f*t)*(t*t);
    }
    return result;
}

float
schlick_fresnel(float cosNO, float R0) {
    // Schlick approximation of Fresnel reflectance
    float cosi2 = cosNO * cosNO;
    float cosi5 = cosi2 * cosi2 * cosNO;
    float F =  R0 + (1 - cosi5) * (1 - R0);
    return F;
}


float computeG_Smith(const Vec3 &N, Vec3 &H, const Vec3 &omega_out, float cosNI, float cosNO)
{
    float cosNH = N.dot(H);
    float cosHO = fabs(omega_out.dot(H));

    float cosNHdivHO = cosNH/cosHO;
    cosNHdivHO = std::max(cosNHdivHO, 0.00001f);

    float fac1 = 2.f * fabs(cosNHdivHO * cosNO);
    float fac2 = 2.f * fabs(cosNHdivHO * cosNI);

    return std::min(1.f, std::min(fac1, fac2));
}

Point2 H_projected(Vec3 &H, const Vec3 &N, const Vec3 &dPdu)
{
        // project half vector into plane defined by N
        // H and N are assumed to be unit length

        Point2 H_projected2D;
        Vec3   H_projected3D, tmp;
        Vec3 base1, base2;

        tmp = H.cross(N);
        H_projected3D =  N.cross(tmp);

        base1 = dPdu;
        base1.normalize();
        base2 = base1.cross(N);

        // get direction cosines on plane
        //
        H_projected2D.x = base1.dot(H_projected3D);
        H_projected2D.y = base2.dot(H_projected3D);

        return H_projected2D;
}

Point2 ellipse_center(float Sx, float Sy,
                      float Kx, float Ky,
                      Point2 H2)
{
    // Sx, Sy are the highlight segment width:height ratio
    // Kx, Ky scale the offset to control how far the highlight moves from the
    // center of the rectangular window
    // H2 is the halfvector projected onto the BTF plane

    Point2 center;

    center.x = 0.5f*Sx*(Kx*H2.x+1.f);
    center.y = 0.5f*Sy*(Ky*H2.y+1.f);


    return center;
}

void rotate_2D(Point2 &point, float angle, Point2 origin) // origin was defaulted to zero?
{
    Point2 rotatedPoint;

    angle *= M_PI/180.f;

    rotatedPoint.x = (point.x-origin.x)*cosf(angle) - (point.y-origin.y)*sinf(angle) + origin.x;
    rotatedPoint.y = (point.y-origin.y)*cosf(angle) + (point.x-origin.x)*sinf(angle) + origin.y;

    point = rotatedPoint;
}

void ellipse_foci(Point2 alpha, float eta, Point2 center,
                  Point2 *F1, Point2 *F2)


{
    // alpha is the semimajor axis, assumed to be rotated slightly
    // off vertical (just to the left of pi/2 at about 95 degrees for a warp thread, for instance)
    // eta is the eccentricity
    // F1 and F2 are the ellipse foci

    F1->x = center.x+alpha.x*eta;
    F1->y = center.y+alpha.y*eta;

    F2->x = center.x-alpha.x*eta;
    F2->y = center.y-alpha.y*eta;
}

float inside_ellipse (Point2 F1, Point2 F2,
                      float uu, float vv,
                      float alpha, float width)
{
    float d;

    Vec3 F1_d, F2_d;

    float I = 0.f;


    // to check whether the point (x,y) lies inside the ellipse, recall
    // that an ellipse is the set of points for which the sum of the
    // distance from the two foci is exactly 2*alpha
    F1_d.x = F1.x-uu;  F1_d.y = F1.y-vv;
    F2_d.x = F2.x-uu;  F2_d.y = F2.y-vv;

    d = sqrtf(F1_d.x*F1_d.x + F1_d.y*F1_d.y) + sqrtf(F2_d.x*F2_d.x + F2_d.y*F2_d.y);
    //d = F1_d.length2() + F2_d.length2();  eh?  

    float ellipse = 2.f*alpha;

    float b = ellipse + width;
    float a = std::max(ellipse - width, 0.f);

    // here, we smoothstep based on width, which is area*filterwidth
    //
    I = 1.f - smoothstep(a, b, d);

    return I;
}


// line seqment circle intersection check, take two
//
bool intersect_circle_segment(Point2 center, float radius, Point2 p1, Point2 p2)
{
    Point2 dir;
    Point2 diff;

    dir.x = p2.x-p1.x;
    dir.y = p2.y-p1.y;

    diff.x = center.x-p1.x;
    diff.y = center.y-p1.y;

    float t = (diff.x*dir.x + diff.y*dir.y) / (dir.x*dir.x + dir.y*dir.y);
    if (t < 0.0f)
        t = 0.0f;
    if (t > 1.0f)
        t = 1.0f;

    dir.x *= t;
    dir.y *= t;

    Point2 closest;
    closest.x = p1.x+dir.x;
    closest.y = p1.y+dir.y;

    Point2 d;
    d.x = center.x - closest.x;
    d.y = center.y - closest.y;

    float distsqr = d.x*d.x + d.y*d.y; //AiV2Dot(d,d);

    return distsqr <= radius * radius;
}


// Calculate the intersection of a ray and a sphere
// The line segment is defined from p1 to p2
// The sphere is of radius r and centered at sc
// There are potentially two points of intersection given by
// p = p1 + mu1 (p2 - p1)
// p = p1 + mu2 (p2 - p1)
// Return false if the ray doesn't intersect the sphere.
bool ray_circle(Point2 p1, Point2 p2, Point2 sc, float r, float *mu1, float *mu2)
{

    if(!intersect_circle_segment(sc, r, p1, p2))
        return false;

    float a,b,c;
    float bb4ac;
    Point2 dp;

    dp.x = p2.x - p1.x;
    dp.y = p2.y - p1.y;
    a = dp.x * dp.x + dp.y * dp.y;
    b = 2.f * (dp.x * (p1.x - sc.x) + dp.y * (p1.y - sc.y));
    c = sc.x * sc.x + sc.y * sc.y;
    c += p1.x * p1.x + p1.y * p1.y;
    c -= 2.f * (sc.x * p1.x + sc.y * p1.y);
    c -= r * r;

    // if the discriminant 'b*b-4*a*c'
    //   is less than zero, the line doesn't intersect
    // if it equals zero, then the line is tangent to the circle,
    //   intersecting it at one point (we don't care about this case)
    // if it's greater than zero the line intersects at two points.
    bb4ac = b * b - 4 * a * c;
    if (fabs(a) < EPSILON || bb4ac <= 0.f) {
        *mu1 = 0;
        *mu2 = 0;
        return false;
    }

    *mu1 = (-b + sqrtf(bb4ac)) / (2.f * a);
    *mu2 = (-b - sqrtf(bb4ac)) / (2.f * a);

    return true;
}


inline Point2 point_on_line(float mu, Point2 p1, Point2 p2)
{
    Point2 pol;

    pol.x = Imath::lerp(p1.x, p2.x, mu);
    pol.y = Imath::lerp(p1.y, p2.y, mu);

    return pol; // point on line
}


// note that this formula works for angles from zero to 2PI!
// hint: when theta is > PI the circle segment contains the origin
// so subtract from 2PI with appropriate logic condition.
inline float seg_area(float theta)
{
    return 0.5f*(theta-sinf(theta));
}


inline float t_area(Point2 P0, Point2 P1, Point2 P2)
{
    Point2 tmp, tmp2;

    tmp.x = P1.x-P0.x;
    tmp.y = P1.y-P0.y;

    tmp2.x = P2.x-P0.x;
    tmp2.y = P2.y-P0.y;

    return ((tmp.x*tmp2.y)-(tmp.y*tmp2.x))/2.f;
}

// remember: it's atan2(y,x) not atan2(x,y)
//
inline float atan2_zero_to_pi(float y, float x)
{
    float z = atan2f(y, x);
    if(z < 0.f)
        z += 2.f * M_PI;

    return z;
}


// computeAC returns the area of the circular segment contained
// by the surrounding M inverse transformed rectangular thread segment window
//
float compute_AC(Point2 *rect, bool OUTSIDE)

{
    Point2 center;
    center.x = 0.f;
    center.y = 0.f;

    float area = M_PI;
    float radius = 1.f; // unit circle

    float mu1=0.f;

    float mu2=0.f;
    int i=0;
    int k=0;

    float theta[2] = {0.f, 0.f};      // two possible theta angles

    // four possible intersections
    Intersection pList[4];

    Point2 P0(0.0f,0.0f), P1(0.0f,0.0f), P2(0.0f,0.0f);

    // find potential intersections of circle with the rectangle perimeter
    // moving clockwise from edge to edge, mu2 is the intersection nearest the
    // starting point of the edge.  mu1 is closer to the edge end point.
    //

    // first edge
    if(ray_circle(rect[0], rect[1], center, radius, &mu1, &mu2))
    {
        if(mu2 > 0.f && mu2 < 1.f)
        {
            pList[i].p = point_on_line(mu2, rect[0], rect[1]);
            pList[i].edge = 1;
            i++;
        }
        if(mu1 > 0.f && mu1 < 1.f)
        {
            pList[i].p = point_on_line(mu1, rect[0], rect[1]);
            pList[i].edge = 1;
            i++;

        }
    }

    // second edge
    if(ray_circle(rect[1], rect[2], center, radius, &mu1, &mu2))
    {
        if(mu2 > 0.f  && mu2 < 1.f)
        {
            pList[i].p = point_on_line(mu2, rect[1], rect[2]);
            pList[i].edge = 2;
            i++;
        }
        if(mu1 > 0.f && mu1 < 1.f)
        {
            pList[i].p= point_on_line(mu1, rect[1], rect[2]);
            pList[i].edge = 2;
            i++;
        }
    }

    // third edge
    if(ray_circle(rect[2], rect[3], center, radius, &mu1, &mu2))
    {
        if(mu2 > 0.f  && mu2 < 1.f)
        {
            pList[i].p = point_on_line(mu2, rect[2], rect[3]);
            pList[i].edge = 3;
            i++;
        }
        if(mu1 > 0.f && mu1 < 1.f)
        {
            pList[i].p = point_on_line(mu1, rect[2], rect[3]);
            pList[i].edge = 3;
            i++;
        }
    }

    // forth edge
    if(ray_circle(rect[3], rect[0], center, radius, &mu1, &mu2))
    {
        if(mu2 > 0.f  && mu2 < 1.f)
        {
            pList[i].p = point_on_line(mu2, rect[3], rect[0]);
            pList[i].edge = 4;
            i++;
        }
        if(mu1 > 0.f && mu1 < 1.f)
        {
            pList[i].p = point_on_line(mu1, rect[3], rect[0]);
            pList[i].edge = 4;
            i++;
        }
    }

    // early out after intersection testing... skip area finding routine
    if(i == 0 && OUTSIDE == false)
        return area;  // inside and no intersections gives area of pi

    if(i == 0 && OUTSIDE == true)
        return 0.f;   // no intersections & not inside gives area of zero

    // subtract the arctangents of pairs of intersections to obtain the directed angles from the first
    // to the second point of each segment of rectangle perimeter that is intersected by the circle
    //
    for(int j=0; j<=i-1; j=j+2) // count by pairs of intersection points
    {
        //printf("i: %d  j: %d\n", i, j);

        float angle1 = atan2_zero_to_pi(pList[j].p.y, pList[j].p.x);
        float angle2 = atan2_zero_to_pi(pList[j+1].p.y, pList[j+1].p.x);
        float ang;

        if(angle2 > angle1)
        {
            ang = angle2-angle1;

            if(i==4)
            {
                // handle for when circle strides lower right corner,
                // intersecting edges 3 & 4 each twice
                if(ang < M_PI && OUTSIDE==true && pList[0].edge==3)
                    ang = 2.f * M_PI - ang;
                // handles for when circle strides upper left corner,
                // intersecting edges 1 & 2 each twice
                else if(ang > M_PI || k==1)
                    ang = 2.f * M_PI - ang;
            }
        }
        else
        {
            ang = angle1-angle2;
            if(i == 2) // we need to subtract ang from 2PI when angle1 < angle2 for 2 intersections
                ang = 2.f * M_PI - ang;
        }

        theta[k] = ang;

        // test for 2 intersections on adjacent edges
        // NOTE: we only care about the first two intersections stored in *pList in this case.
        if(i == 2 && (pList[0].edge != pList[1].edge))
        {
            // find rect corner
            // and set triangle points
            int corner = pList[j].edge;  //edge 4 gives corner zero
            P0 = rect[corner];
            P1 = pList[j].p;
            P2 = pList[j+1].p;
            i=5;

            if(OUTSIDE == true)
            {
                if(angle1 > angle2)
                    theta[k] = 2.f * M_PI - (angle1 - angle2);
                else
                    theta[k] = angle2 - angle1;
            }
            else
            {
                if(angle2 > angle1)
                    theta[k] = (angle2 - angle1);
                else
                    theta[k] = 2.f * M_PI - (angle1 - angle2);

            }

            // intersection striding edge 4 to edge 1
            if(pList[j].edge == 1 && pList[j+1].edge == 4)
            {
                P0 = rect[0];
                float tmp=angle1;
                angle1 = angle2;
                angle2 = tmp;

                // intersection order relationship changes in order to keep going clockwise around the rectangle
                if(OUTSIDE == true)
                {
                    if(angle1 > angle2)
                        theta[k] = 2.f * M_PI - (angle1 - angle2);
                    else
                        theta[k] = angle2-angle1;
                }
                else
                {
                    if(angle2 > angle1)
                        theta[k] = angle2 - angle1;
                    else
                        theta[k] = 2.f * M_PI - (angle1 - angle2);
                }
            }

            break;
        }

        k++;
    }

    // if i > 0 there has been one or more intersections with the rectangle segment
    //
    if(i == 2) {
        area = seg_area(theta[0]);
        return area;
    }

    if(i == 4){
        area = M_PI - seg_area(theta[0]) - seg_area(theta[1]);
        return area;
    }

    // when there are two intersections on different edges (a corner is strided), a line is drawn between
    // each intersection, forming a triangle with the rectangle corner.
    // The summation of the triangle and circle segment area give the true segment area.
    //
    // Note: there aren't really 5 intersections, here.  This case of 4 intersections is just set to 5.
    //
    if(i == 5){
        area = seg_area(theta[0]) + fabs(t_area(P0, P1, P2));
        return area;
    }

    return area;
}


}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
using namespace OSL_NAMESPACE;
#endif

