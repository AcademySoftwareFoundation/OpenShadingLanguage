#include "bvh.h"
#include "raytracer.h"

#include <OpenImageIO/timer.h>
#include <Imath/ImathBox.h>

OSL_NAMESPACE_ENTER

using Box3 = Imath::Box3f;

struct BuildNode {
    Box3 centroid;
    unsigned left, right;
    unsigned nodeIndex;
    int depth;
};

float half_area(const Box3& b) {
    Vec3 d = b.max - b.min;
    return d.x * d.y + d.y * d.z + d.z * d.x;
}

static constexpr int NumBins = 16;
static constexpr int MaxDepth = 64;

static std::unique_ptr<BVH> build_bvh(const Vec3* verts, const unsigned* triangles, const unsigned ntris, OIIO::ErrorHandler& errhandler) {
    std::unique_ptr<BVH> bvh = std::make_unique<BVH>();
    OIIO::Timer timer;
    bvh->indices = std::make_unique<unsigned[]>(ntris);

    std::vector<BVHNode> buildnodes;
    std::vector<Box3> triangle_bounds;
    buildnodes.reserve(2 * ntris + 1);
    buildnodes.emplace_back();
    triangle_bounds.reserve(ntris);
    BuildNode current;
    Box3 shape_bounds;
    for (unsigned i = 0; i < ntris; i++) {
        bvh->indices[i] = i;
        Vec3 A = verts[triangles[3 * i + 0]];
        Vec3 B = verts[triangles[3 * i + 1]];
        Vec3 C = verts[triangles[3 * i + 2]];
        Box3 b(A);
        b.extendBy(B);
        b.extendBy(C);
        triangle_bounds.emplace_back(b);
        current.centroid.extendBy(b.center());
        shape_bounds.extendBy(b);
    }
    buildnodes[0].set(shape_bounds.min, shape_bounds.max);
    current.left = 0;
    current.right = ntris;
    current.depth = 1;
    current.nodeIndex = 0;
    int stackPtr = 0;
    BuildNode stack[MaxDepth];
    while (true) {
        const unsigned numPrims = current.right - current.left;
        if (numPrims > 1 && current.depth < MaxDepth) {
            // try to split this set of primitives
            Box3 binBounds[3][NumBins];
            unsigned binN[3][NumBins]; memset(binN, 0, sizeof(binN));

            float binFactor[3];
            for (int axis = 0; axis < 3; axis++) {
                binFactor[axis] = current.centroid.max[axis] - current.centroid.min[axis];
                binFactor[axis] = (binFactor[axis] > 0) ? float(0.999f * NumBins) / binFactor[axis] : 0;
            }
            // for each primitive, figure out in which bin it lands per axis
            for (unsigned i = current.left; i < current.right; i++) {
                unsigned prim = bvh->indices[i];
                Box3 bbox = triangle_bounds[prim];
                Vec3 center = bbox.center();
                for (int axis = 0; axis < 3; axis++) {
                    int binID = (int) ((center[axis] - current.centroid.min[axis]) * binFactor[axis]);
                    OSL_ASSERT(binID >= 0 && binID < NumBins);
                    binN[axis][binID]++;
                    binBounds[axis][binID].extendBy(bbox);
                }
            }
            // compute the SAH cost of partitioning at each bin
            const float invArea = 1 / buildnodes[current.nodeIndex].half_area();
            float bestCost = numPrims;
            int bestAxis = -1;
            int bestBin = -1;
            unsigned bestNL = 0;
            unsigned bestNR = 0;
            for (int axis = 0; axis < 3; axis++) {
                // skip if the current bbox is flat along this axis (splitting would not make sense)
                if (binFactor[axis] == 0) continue;
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
                    if (numL[i] == 0 || numL[i] == numPrims) continue; // skip if this candidate split does not partition the prims
                    float areaR = half_area(bbox);
                    const float trav_cost = 4; // TODO: tune this if intersection function changes
                    const float cost = trav_cost + invArea * (areaL[i] * numL[i] + areaR * (numPrims - numL[i]));
                    if (cost < bestCost) {
                        bestCost = cost;
                        bestAxis = axis;
                        bestBin = i;
                        bestNL = numL[i];
                        bestNR = numPrims - bestNL;
                    }
                    bbox.extendBy(binBounds[axis][i]);
                }
            }
            if (bestAxis != -1) {
                // split along the found best split
                Box3 boundsL, boundsR;
                BuildNode bn[2];
                bn[0].depth = bn[1].depth = current.depth + 1;
                unsigned rightOrig = current.right;
                for (unsigned i = current.left; i < current.right;) {
                    unsigned prim = bvh->indices[i];
                    Box3 bbox = triangle_bounds[prim];
                    float center = bbox.center()[bestAxis];
                    int binID = (int) ((center - current.centroid.min[bestAxis]) * binFactor[bestAxis]);
                    OSL_ASSERT(binID >= 0 && binID < NumBins);
                    if (binID <= bestBin) {
                        boundsL.extendBy(bbox);
                        bn[0].centroid.extendBy(bbox.center());
                        i++;
                    } else {
                        boundsR.extendBy(bbox);
                        bn[1].centroid.extendBy(bbox.center());
                        std::swap(bvh->indices[i], bvh->indices[--current.right]);
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
                buildnodes[current.nodeIndex].child = nextIndex;
                buildnodes[current.nodeIndex].nprims = 0;
                bn[0].left  = current.left;
                bn[0].right = current.right;
                bn[1].left  = current.right;
                bn[1].right = rightOrig;
                bn[0].nodeIndex = nextIndex + 0;
                bn[1].nodeIndex = nextIndex + 1;
                buildnodes[nextIndex + 0].set(boundsL.min, boundsL.max);
                buildnodes[nextIndex + 1].set(boundsR.min, boundsR.max);
                current           = bn[0];
                stack[stackPtr++] = bn[1];
                continue; // keep building
            }
        }
        // nothing more to be done with this node - create a leaf
        buildnodes[current.nodeIndex].child = current.left;
        buildnodes[current.nodeIndex].nprims = numPrims;
        // pop the stack
        if (stackPtr == 0)
            break;
        current = stack[--stackPtr];
    }
    bvh->nodes = std::make_unique<BVHNode[]>(buildnodes.size());
    memcpy(bvh->nodes.get(), buildnodes.data(), buildnodes.size() * sizeof(BVHNode));
    double loadtime = timer();
    errhandler.infofmt("BVH built {} nodes over {} triangles in {}", buildnodes.size(), ntris, OIIO::Strutil::timeintervalformat(loadtime, 2));
    errhandler.infofmt("Root bounding box {}, {}, {} to {}, {}, {}",
        shape_bounds.min.x, shape_bounds.min.y, shape_bounds.min.z,
        shape_bounds.max.x, shape_bounds.max.y, shape_bounds.max.z
    );
    return bvh;
}

// min and max, written such that any NaNs in 'b' get ignored
static inline float minf(float a, float b) { return b < a ? b : a; }
static inline float maxf(float a, float b) { return b > a ? b : a; }

static inline float box_intersect(const Vec3& org, const Vec3& rdir, float tmax, const float* bounds) {
    const float tx1 = (bounds[0] - org.x) * rdir.x;
    const float tx2 = (bounds[1] - org.x) * rdir.x;
    const float ty1 = (bounds[2] - org.y) * rdir.y;
    const float ty2 = (bounds[3] - org.y) * rdir.y;
    const float tz1 = (bounds[4] - org.z) * rdir.z;
    const float tz2 = (bounds[5] - org.z) * rdir.z;
    float tmin = 0.0f;
    tmin = maxf(tmin, minf(tx1, tx2)); tmax = minf(tmax, maxf(tx1, tx2));
    tmin = maxf(tmin, minf(ty1, ty2)); tmax = minf(tmax, maxf(ty1, ty2));
    tmin = maxf(tmin, minf(tz1, tz2)); tmax = minf(tmax, maxf(tz1, tz2));
    return tmin <= tmax ? tmin : -1;
}


static inline unsigned signmask(float a) {
    return OIIO::bitcast<unsigned>(a) & 0x80000000u;
}
static inline float xorf(float a, unsigned b) {
    return OIIO::bitcast<float>(OIIO::bitcast<unsigned>(a) ^ b);
}

Intersection BVH::intersect(const Vec3& org, const Vec3& dir, const float tmax, const Vec3* verts, const unsigned* triangles, const unsigned skipID) {
    struct StackItem {
        BVHNode* node;
        float dist;
    }  stack[MaxDepth];
    Intersection result;
    result.t = tmax;
    stack[0] = { nodes.get(), result.t };
    const Vec3 rdir(1 / dir.x, 1 / dir.y, 1 / dir.z );
    int kz = 0;
    if (fabsf(dir.y) > fabsf(dir[kz])) kz = 1;
    if (fabsf(dir.z) > fabsf(dir[kz])) kz = 2;
    int kx = kz == 2 ? 0 : kz + 1;
    int ky = kx == 2 ? 0 : kx + 1;
    const Vec3 shearDir(dir[kx] / dir[kz], dir[ky] / dir[kz], rdir[kz]);
	for (int stackPtr = 1; stackPtr != 0;) {
        if (result.t < stack[--stackPtr].dist) continue;
        BVHNode* node = stack[stackPtr].node;
		if (node->nprims) {
			for (unsigned i = 0, nprims = node->nprims; i < nprims; i++) {
                unsigned id = indices[node->child + i];
                // Watertight Ray/Triangle Intersection - JCGT 2013
                // https://jcgt.org/published/0002/01/05/
                const Vec3 A = verts[triangles[3 * id + 0]] - org;
                const Vec3 B = verts[triangles[3 * id + 1]] - org;
                const Vec3 C = verts[triangles[3 * id + 2]] - org;
                const float Ax = A[kx] - shearDir.x * A[kz];
                const float Ay = A[ky] - shearDir.y * A[kz];
                const float Bx = B[kx] - shearDir.x * B[kz];
                const float By = B[ky] - shearDir.y * B[kz];
                const float Cx = C[kx] - shearDir.x * C[kz];
                const float Cy = C[ky] - shearDir.y * C[kz];
                // TODO: could use Kahan's algorithm here
                // https://pharr.org/matt/blog/2019/11/03/difference-of-floats
                const float U = Cx * By - Cy * Bx;
                const float V = Ax * Cy - Ay * Cx;
                const float W = Bx * Ay - By * Ax;
                if ((U < 0 || V < 0 || W < 0) &&
                    (U > 0 || V > 0 || W > 0))
                    continue;
                const float det = U + V + W;
                if (det == 0) continue;
                const float Az = A[kz];
                const float Bz = B[kz];
                const float Cz = C[kz];
                const float T = shearDir.z * (U * Az + V * Bz + W * Cz);
                const unsigned mask = signmask(det);
                if (xorf(T, mask) < 0) continue;
                if (xorf(T, mask) > result.t * xorf(det, mask)) continue;
                if (id == skipID) continue; // skip self
                // we know this is a valid hit, record as closest
                const float rcpDet = 1 / det;
                result.t = T * rcpDet;
                result.u = V * rcpDet;
                result.v = W * rcpDet;
                result.id = id;
			}
		} else {
            BVHNode* child1 = nodes.get() + node->child;
            BVHNode* child2 = child1 + 1;
            float dist1 = box_intersect(org, rdir, result.t, child1->bounds);
            float dist2 = box_intersect(org, rdir, result.t, child2->bounds);
            if (dist1 > dist2) {
                std::swap(dist1 , dist2 ); 
                std::swap(child1, child2);
            }
            stack[stackPtr] = { child2, dist2 }; stackPtr += (dist2 >= 0);
            stack[stackPtr] = { child1, dist1 }; stackPtr += (dist1 >= 0);
        }
	}
    return result;
}

void Scene::prepare(OIIO::ErrorHandler& errhandler) {
    verts.shrink_to_fit();
    indices.shrink_to_fit();
    bvh = build_bvh(verts.data(), indices.data(), indices.size() / 3, errhandler);
}

OSL_NAMESPACE_EXIT
