#pragma once

#include "OSL/oslconfig.h"
#include <OpenImageIO/fmath.h>
#include <algorithm>
#include <cmath>

OSL_NAMESPACE_ENTER

/// Tiny random number generator
/// http://burtleburtle.net/bob/rand/smallprng.html
struct Rng {
  Rng(int s) {
      a = 0xF1EA5EED;
      b = c = d = s;
      // do 20 rounds to mix up the initial state
      for (s = 20; s--; (void)(float)*this);
  }

  operator float() {
      // mix internal state
      unsigned int e;
      e = a - ((b << 27) | (b >>  5));
      a = b ^ ((c << 17) | (c >> 15));
      b = c + d;
      c = d + e;
      d = e + a;
      // convert "d" to a float in the range [0,1)
      union {
          float f;
          unsigned int i;
      } x;
      x.i = 0x3F800000 | (0x3F7FFFFF & d);
      return x.f - 1.0f;
  }

private:
  unsigned int a, b, c, d;
};

struct TangentFrame {
    // build frame from unit normal
    TangentFrame(const Vec3& n) : w(n) {
        u = (fabsf(w.x) >.01f ? Vec3(w.z, 0, -w.x) :
                                Vec3(0, -w.z, w.y)).normalize();
        v = w.cross(u);
    }

    // build frame from unit normal and unit tangent
    TangentFrame(const Vec3& n, const Vec3& t) : w(n) {
        v = w.cross(t);
        u = v.cross(w);
    }

    // transform vector
    Vec3 get(float x, float y, float z) const {
        return x * u + y * v + z * w;
    }

    // untransform vector
    float getx(const Vec3& a) const { return a.dot(u); }
    float gety(const Vec3& a) const { return a.dot(v); }
    float getz(const Vec3& a) const { return a.dot(w); }

private:
    Vec3 u, v, w;
};

struct Sampling {
    /// Warp the unit disk onto the unit sphere
    /// http://psgraphics.blogspot.com/2011/01/improved-code-for-concentric-map.html
    static void to_unit_disk(float& x, float& y) {
        const float PI_OVER_4 = float(M_PI_4);
        const float PI_OVER_2 = float(M_PI_2);
        float phi, r;
        float a = 2 * x - 1;
        float b = 2 * y - 1;
        if (a * a > b * b) { // use squares instead of absolute values
            r = a;
            phi = PI_OVER_4 * (b / a);
        } else if (b != 0) { // b is largest
            r = b;
            phi = PI_OVER_2 - PI_OVER_4 * (a / b);
        } else { // a == b == 0
            r = 0;
            phi = 0;
        }
        OIIO::sincos(phi, &x, &y);
        x *= r;
        y *= r;
    }

    static void sample_cosine_hemisphere(const Vec3& N, float rndx, float rndy, Vec3& out, float& invpdf) {
        to_unit_disk(rndx, rndy);
        float cos_theta = sqrtf(std::max(1 - rndx * rndx - rndy * rndy, 0.0f));
        TangentFrame f(N);
        out = f.get(rndx, rndy, cos_theta);
        invpdf = float(M_PI) / cos_theta;
    }
};

OSL_NAMESPACE_EXIT
