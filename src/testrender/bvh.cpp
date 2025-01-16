// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "bvh.h"
#include "raytracer.h"

#include <Imath/ImathBox.h>
#include <OpenImageIO/timer.h>

OSL_NAMESPACE_BEGIN

using Box3 = Imath::Box3f;

struct BuildNode {
    Box3 centroid;
    unsigned left, right;
    unsigned nodeIndex;
    int depth;
};

float
half_area(const Box3& b)
{
    Vec3 d = b.max - b.min;
    return d.x * d.y + d.y * d.z + d.z * d.x;
}


// Workaround for Imath::Vec3 undefined behavior of operator[]
inline float
comp(const Vec3& vec, int i)
{
    return (reinterpret_cast<const float*>(&vec))[i];
}



static constexpr int NumBins  = 16;
static constexpr int MaxDepth = 64;

static std::unique_ptr<BVH>
build_bvh(OIIO::cspan<Vec3> verts, OIIO::cspan<TriangleIndices> triangles,
          OIIO::ErrorHandler& errhandler)
{
    std::unique_ptr<BVH> bvh = std::make_unique<BVH>();
    OIIO::Timer timer;
    bvh->indices = std::make_unique<unsigned[]>(triangles.size());

    std::vector<BVHNode> buildnodes;
    std::vector<Box3> triangle_bounds;
    buildnodes.reserve(2 * triangles.size() + 1);
    buildnodes.emplace_back();
    triangle_bounds.reserve(triangles.size());
    BuildNode current;
    Box3 shape_bounds;
    for (unsigned i = 0; i < triangles.size(); i++) {
        bvh->indices[i] = i;
        Vec3 va         = verts[triangles[i].a];
        Vec3 vb         = verts[triangles[i].b];
        Vec3 vc         = verts[triangles[i].c];
        Box3 b(va);
        b.extendBy(vb);
        b.extendBy(vc);
        triangle_bounds.emplace_back(b);
        current.centroid.extendBy(b.center());
        shape_bounds.extendBy(b);
    }
    buildnodes[0].set(shape_bounds.min, shape_bounds.max);
    current.left      = 0;
    current.right     = triangles.size();
    current.depth     = 1;
    current.nodeIndex = 0;
    int stackPtr      = 0;
    BuildNode stack[MaxDepth];
    while (true) {
        const unsigned numPrims = current.right - current.left;
        if (numPrims > 1 && current.depth < MaxDepth) {
            // try to split this set of primitives
            Box3 binBounds[3][NumBins];
            unsigned binN[3][NumBins];
            memset(binN, 0, sizeof(binN));

            float binFactor[3];
            for (int axis = 0; axis < 3; axis++) {
                binFactor[axis] = comp(current.centroid.max, axis)
                                  - comp(current.centroid.min, axis);
                binFactor[axis] = (binFactor[axis] > 0)
                                      ? float(0.999f * NumBins)
                                            / binFactor[axis]
                                      : 0;
            }
            // for each primitive, figure out in which bin it lands per axis
            for (unsigned i = current.left; i < current.right; i++) {
                unsigned prim = bvh->indices[i];
                Box3 bbox     = triangle_bounds[prim];
                Vec3 center   = bbox.center();
                for (int axis = 0; axis < 3; axis++) {
                    int binID = (int)((comp(center, axis)
                                       - comp(current.centroid.min, axis))
                                      * binFactor[axis]);
                    OSL_ASSERT(binID >= 0 && binID < NumBins);
                    binN[axis][binID]++;
                    binBounds[axis][binID].extendBy(bbox);
                }
            }
            // compute the SAH cost of partitioning at each bin
            const float invArea = 1 / buildnodes[current.nodeIndex].half_area();
            float bestCost      = numPrims;
            int bestAxis        = -1;
            int bestBin         = -1;
            unsigned bestNL     = 0;
            unsigned bestNR     = 0;
            for (int axis = 0; axis < 3; axis++) {
                // skip if the current bbox is flat along this axis (splitting would not make sense)
                if (binFactor[axis] == 0)
                    continue;
                unsigned countL = 0;
                Box3 bbox;
                unsigned numL[NumBins];
                float areaL[NumBins];
                for (int i = 0; i < NumBins; i++) {
                    countL += binN[axis][i];
                    numL[i] = countL;
                    bbox.extendBy(binBounds[axis][i]);
                    areaL[i] = half_area(bbox);
                }
                OSL_ASSERT(countL == numPrims);
                bbox = binBounds[axis][NumBins - 1];
                for (int i = NumBins - 2; i >= 0; i--) {
                    if (numL[i] == 0 || numL[i] == numPrims)
                        continue;  // skip if this candidate split does not partition the prims
                    float areaR = half_area(bbox);
                    const float trav_cost
                        = 4;  // TODO: tune this if intersection function changes
                    const float cost = trav_cost
                                       + invArea
                                             * (areaL[i] * numL[i]
                                                + areaR * (numPrims - numL[i]));
                    if (cost < bestCost) {
                        bestCost = cost;
                        bestAxis = axis;
                        bestBin  = i;
                        bestNL   = numL[i];
                        bestNR   = numPrims - bestNL;
                    }
                    bbox.extendBy(binBounds[axis][i]);
                }
            }
            if (bestAxis != -1) {
                // split along the found best split
                Box3 boundsL, boundsR;
                BuildNode bn[2];
                bn[0].depth = bn[1].depth = current.depth + 1;
                unsigned rightOrig        = current.right;
                for (unsigned i = current.left; i < current.right;) {
                    unsigned prim = bvh->indices[i];
                    Box3 bbox     = triangle_bounds[prim];
                    float center  = comp(bbox.center(), bestAxis);
                    int binID
                        = (int)((center - comp(current.centroid.min, bestAxis))
                                * binFactor[bestAxis]);
                    OSL_ASSERT(binID >= 0 && binID < NumBins);
                    if (binID <= bestBin) {
                        boundsL.extendBy(bbox);
                        bn[0].centroid.extendBy(bbox.center());
                        i++;
                    } else {
                        boundsR.extendBy(bbox);
                        bn[1].centroid.extendBy(bbox.center());
                        std::swap(bvh->indices[i],
                                  bvh->indices[--current.right]);
                    }
                }
                OSL_ASSERT(bestNL == (current.right - current.left));
                OSL_ASSERT(bestNR == (rightOrig - current.right));
                OSL_ASSERT(bestNL + bestNR == numPrims);
                // allocate 2 child nodes
                unsigned nextIndex = buildnodes.size();
                buildnodes.emplace_back();
                buildnodes.emplace_back();
                // write to current node
                buildnodes[current.nodeIndex].child  = nextIndex;
                buildnodes[current.nodeIndex].nprims = 0;
                bn[0].left                           = current.left;
                bn[0].right                          = current.right;
                bn[1].left                           = current.right;
                bn[1].right                          = rightOrig;
                bn[0].nodeIndex                      = nextIndex + 0;
                bn[1].nodeIndex                      = nextIndex + 1;
                buildnodes[nextIndex + 0].set(boundsL.min, boundsL.max);
                buildnodes[nextIndex + 1].set(boundsR.min, boundsR.max);
                current           = bn[0];
                stack[stackPtr++] = bn[1];
                continue;  // keep building
            }
        }
        // nothing more to be done with this node - create a leaf
        buildnodes[current.nodeIndex].child  = current.left;
        buildnodes[current.nodeIndex].nprims = numPrims;
        // pop the stack
        if (stackPtr == 0)
            break;
        current = stack[--stackPtr];
    }
    bvh->nodes = std::make_unique<BVHNode[]>(buildnodes.size());
    memcpy(bvh->nodes.get(), buildnodes.data(),
           buildnodes.size() * sizeof(BVHNode));
    double loadtime = timer();
    errhandler.infofmt("BVH built {} nodes over {} triangles in {}",
                       buildnodes.size(), triangles.size(),
                       OIIO::Strutil::timeintervalformat(loadtime, 2));
    errhandler.infofmt("Root bounding box {}, {}, {} to {}, {}, {}",
                       shape_bounds.min.x, shape_bounds.min.y,
                       shape_bounds.min.z, shape_bounds.max.x,
                       shape_bounds.max.y, shape_bounds.max.z);
    return bvh;
}

// min and max, written such that any NaNs in 'b' get ignored
static inline float
minf(float a, float b)
{
    return b < a ? b : a;
}
static inline float
maxf(float a, float b)
{
    return b > a ? b : a;
}

static inline bool
box_intersect(const Vec3& org, const Vec3& rdir, float tmax,
              const float* bounds, float* dist)
{
    const float tx1 = (bounds[0] - org.x) * rdir.x;
    const float tx2 = (bounds[1] - org.x) * rdir.x;
    const float ty1 = (bounds[2] - org.y) * rdir.y;
    const float ty2 = (bounds[3] - org.y) * rdir.y;
    const float tz1 = (bounds[4] - org.z) * rdir.z;
    const float tz2 = (bounds[5] - org.z) * rdir.z;
    float tmin      = minf(tx1, tx2);
    tmax            = minf(tmax, maxf(tx1, tx2));
    tmin            = maxf(tmin, minf(ty1, ty2));
    tmax            = minf(tmax, maxf(ty1, ty2));
    tmin            = maxf(tmin, minf(tz1, tz2));
    tmax            = minf(tmax, maxf(tz1, tz2));
    *dist           = tmin;  // actual distance to near plane on the box
    tmin            = maxf(0.0f, tmin);  // clip to valid portion of ray
    return tmin <= tmax;
}

static inline unsigned
signmask(float a)
{
    return OIIO::bitcast<unsigned>(a) & 0x80000000u;
}
static inline float
xorf(float a, unsigned b)
{
    return OIIO::bitcast<float>(OIIO::bitcast<unsigned>(a) ^ b);
}

Intersection
Scene::intersect(const Ray& ray, const float tmax, unsigned skipID1,
                 unsigned skipID2) const
{
    struct StackItem {
        BVHNode* node;
        float dist;
    } stack[MaxDepth];
    Intersection result;
    result.t       = tmax;
    stack[0]       = { bvh->nodes.get(), result.t };
    const Vec3 org = ray.origin;
    const Vec3 dir = ray.direction;
    const Vec3 rdir(1 / dir.x, 1 / dir.y, 1 / dir.z);
    int kz = 0;
    if (fabsf(dir.y) > fabsf(comp(dir, kz)))
        kz = 1;
    if (fabsf(dir.z) > fabsf(comp(dir, kz)))
        kz = 2;
    int kx = kz == 2 ? 0 : kz + 1;
    int ky = kx == 2 ? 0 : kx + 1;
    const Vec3 shearDir(comp(dir, kx) / comp(dir, kz),
                        comp(dir, ky) / comp(dir, kz), comp(rdir, kz));
    for (int stackPtr = 1; stackPtr != 0;) {
        if (result.t < stack[--stackPtr].dist)
            continue;
        BVHNode* node = stack[stackPtr].node;
        if (node->nprims) {
            for (unsigned i = 0, nprims = node->nprims; i < nprims; i++) {
                unsigned id = bvh->indices[node->child + i];
                // Watertight Ray/Triangle Intersection - JCGT 2013
                // https://jcgt.org/published/0002/01/05/
                const Vec3 A   = verts[triangles[id].a] - org;
                const Vec3 B   = verts[triangles[id].b] - org;
                const Vec3 C   = verts[triangles[id].c] - org;
                const float Ax = comp(A, kx) - shearDir.x * comp(A, kz);
                const float Ay = comp(A, ky) - shearDir.y * comp(A, kz);
                const float Bx = comp(B, kx) - shearDir.x * comp(B, kz);
                const float By = comp(B, ky) - shearDir.y * comp(B, kz);
                const float Cx = comp(C, kx) - shearDir.x * comp(C, kz);
                const float Cy = comp(C, ky) - shearDir.y * comp(C, kz);
                // TODO: could use Kahan's algorithm here
                // https://pharr.org/matt/blog/2019/11/03/difference-of-floats
                const float U = Cx * By - Cy * Bx;
                const float V = Ax * Cy - Ay * Cx;
                const float W = Bx * Ay - By * Ax;
                if ((U < 0 || V < 0 || W < 0) && (U > 0 || V > 0 || W > 0))
                    continue;
                const float det = U + V + W;
                if (det == 0)
                    continue;
                const float Az      = comp(A, kz);
                const float Bz      = comp(B, kz);
                const float Cz      = comp(C, kz);
                const float T       = shearDir.z * (U * Az + V * Bz + W * Cz);
                const unsigned mask = signmask(det);
                if (xorf(T, mask) < 0)
                    continue;
                if (xorf(T, mask) > result.t * xorf(det, mask))
                    continue;
                if (id == skipID1)
                    continue;  // skip source triangle
                if (id == skipID2)
                    continue;  // skip target triangle
                // we know this is a valid hit, record as closest
                const float rcpDet = 1 / det;
                result.t           = T * rcpDet;
                result.u           = V * rcpDet;
                result.v           = W * rcpDet;
                result.id          = id;
            }
        } else {
            BVHNode* child1 = bvh->nodes.get() + node->child;
            BVHNode* child2 = child1 + 1;
            float dist1     = 0;
            bool hit1       = box_intersect(org, rdir, result.t, child1->bounds,
                                            &dist1);
            float dist2     = 0;
            bool hit2       = box_intersect(org, rdir, result.t, child2->bounds,
                                            &dist2);
            if (dist1 > dist2) {
                std::swap(hit1, hit2);
                std::swap(dist1, dist2);
                std::swap(child1, child2);
            }
            stack[stackPtr] = { child2, dist2 };
            stackPtr += hit2;
            stack[stackPtr] = { child1, dist1 };
            stackPtr += hit1;
        }
    }
    return result;
}

void
Scene::prepare(OIIO::ErrorHandler& errhandler)
{
    verts.shrink_to_fit();
    normals.shrink_to_fit();
    uvs.shrink_to_fit();
    triangles.shrink_to_fit();
    n_triangles.shrink_to_fit();
    uv_triangles.shrink_to_fit();
    OSL_DASSERT(triangles.size() == n_triangles.size());
    OSL_DASSERT(triangles.size() == uv_triangles.size());
    shaderids.shrink_to_fit();
    bvh = build_bvh(verts, triangles, errhandler);
}

OSL_NAMESPACE_END
