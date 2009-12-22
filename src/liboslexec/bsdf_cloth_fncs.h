
#include "oslops.h"
#include "oslexec_pvt.h"
#include <OpenEXR/ImathMatrix.h>


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {

const float PITIMES2 = 6.2831853071795862;
const float EPSILON  = 1.0e-4;

float computeG_Smith(const Vec3 &N, Vec3 &H, const Vec3 &omega_in, const Vec3 &omega_out);


// stuff for "I" term which replaces microfacet slope term "D" in generalized microfacet model
// follows...

typedef struct Point2
{
    float x;
    float y;
} Point2;

const Point2 P2_ZERO = {0.f, 0.f};

typedef struct Intersection
{
    Point2 p;
    int edge;
} Intersection;

//float lerp(float t, float a, float b);

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

float compute_AC(Vec3 highlight, Vec3 *rect3, bool OUTSIDE);

// N.B. Determinant implementation graciously contributed by Marcos Fajardo.

// Calculate the determinant of a 2x2 matrix.
template <typename F>
inline F det2x2(F a, F b, F c, F d)
{
    return a * d - b * c;
}

// calculate the determinant of a 3x3 matrix in the form:
//     | a1,  b1,  c1 |
//     | a2,  b2,  c2 |
//     | a3,  b3,  c3 |
template <typename F>
inline F det3x3(F a1, F a2, F a3, F b1, F b2, F b3, F c1, F c2, F c3)
{
    return a1 * det2x2( b2, b3, c2, c3 )
         - b1 * det2x2( a2, a3, c2, c3 )
         + c1 * det2x2( a2, a3, b2, b3 );
}

// calculate the determinant of a 4x4 matrix.
template <typename F>
inline F det4x4(const Imath::Matrix44<F> &m)
{
    // assign to individual variable names to aid selecting correct elements
    F a1 = m[0][0], b1 = m[0][1], c1 = m[0][2], d1 = m[0][3];
    F a2 = m[1][0], b2 = m[1][1], c2 = m[1][2], d2 = m[1][3];
    F a3 = m[2][0], b3 = m[2][1], c3 = m[2][2], d3 = m[2][3];
    F a4 = m[3][0], b4 = m[3][1], c4 = m[3][2], d4 = m[3][3];
    return a1 * det3x3( b2, b3, b4, c2, c3, c4, d2, d3, d4)
         - b1 * det3x3( a2, a3, a4, c2, c3, c4, d2, d3, d4)
         + c1 * det3x3( a2, a3, a4, b2, b3, b4, d2, d3, d4)
         - d1 * det3x3( a2, a3, a4, b2, b3, b4, c2, c3, c4);
}


}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
using namespace OSL_NAMESPACE;
#endif

