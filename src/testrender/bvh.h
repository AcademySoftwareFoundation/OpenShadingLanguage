#pragma once

#include <OSL/oslconfig.h>
#include <memory>

OSL_NAMESPACE_ENTER

struct BVHNode {
    float bounds[6];
    unsigned child, nprims;

    void set(const Vec3& lo, const Vec3& hi)
    {
        bounds[0] = lo.x;
        bounds[1] = hi.x;
        bounds[2] = lo.y;
        bounds[3] = hi.y;
        bounds[4] = lo.z;
        bounds[5] = hi.z;
    }

    float half_area() const
    {
        float vx = bounds[1] - bounds[0];
        float vy = bounds[3] - bounds[2];
        float vz = bounds[5] - bounds[4];
        return vx * vy + vy * vz + vz * vx;
    }
};
struct Intersection {
    float t, u, v;
    unsigned id;
};

struct BVH {
    std::unique_ptr<BVHNode[]> nodes;
    std::unique_ptr<unsigned[]> indices;
};

OSL_NAMESPACE_EXIT
