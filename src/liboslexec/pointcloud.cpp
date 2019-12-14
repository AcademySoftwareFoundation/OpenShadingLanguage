/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
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

#include <cstdarg>

#include "oslexec_pvt.h"
using namespace OSL;
using namespace OSL::pvt;

#ifdef USE_PARTIO
#include <Partio.h>
#include <unordered_map>
#endif



namespace { // anon

#ifdef USE_PARTIO

class PointCloud {
public:
    PointCloud (ustring filename, Partio::ParticlesDataMutable *partio_cloud, bool write);
    ~PointCloud ();
    static PointCloud *get (ustring filename, bool write = false);

    typedef std::unordered_map<ustring, std::shared_ptr<Partio::ParticleAttribute>, ustringHash> AttributeMap;
    // N.B./FIXME(C++11): shared_ptr is probably overkill, but
    // scoped_ptr is not copyable and therefore can't be used in
    // standard containers.  When C++11 is uniquitous, unique_ptr is the
    // one that should really be used.

    const Partio::ParticlesData* read_access() const { OSL_DASSERT(!m_write); return m_partio_cloud; }
    Partio::ParticlesDataMutable* write_access() const { OSL_DASSERT(m_write); return m_partio_cloud; }

    ustring m_filename;
private:
    // hide just this field, because we want to control how it is accessed
    Partio::ParticlesDataMutable *m_partio_cloud;
public:

    AttributeMap m_attributes;
    bool m_write;
    Partio::ParticleAttribute m_position_attribute;
    spin_mutex m_mutex;
};


typedef std::unordered_map<ustring, std::shared_ptr<PointCloud>, ustringHash> PointCloudMap;
// See above note about shared_ptr vs unique_ptr.
static PointCloudMap pointclouds;
static spin_mutex pointcloudmap_mutex;
static ustring u_position ("position");


// some helper classes to make the sort easy
typedef std::pair<float,int> SortedPointRecord;  // dist,index
struct SortedPointCompare {
    bool operator() (const SortedPointRecord &a, const SortedPointRecord &b) {
        return a.first < b.first;
    }
};



PointCloud *
PointCloud::get (ustring filename, bool write)
{
    if (filename.empty())
        return NULL;
    spin_lock lock (pointcloudmap_mutex);
    PointCloudMap::const_iterator found = pointclouds.find(filename);
    if (found != pointclouds.end())
        return found->second.get();
    // Not found. Create a new one.
    Partio::ParticlesDataMutable *partio_cloud = NULL;
    if (!write) {
        partio_cloud = Partio::read(filename.c_str());
        if (! partio_cloud)
            return NULL;
    } else {
        partio_cloud = Partio::create();
    }
    PointCloud *pc = new PointCloud (filename, partio_cloud, write);
    pointclouds[filename].reset (pc);
    return pc;
}


PointCloud::PointCloud (ustring filename,
                        Partio::ParticlesDataMutable *partio_cloud, bool write)
    : m_filename(filename), m_partio_cloud(partio_cloud), m_write(write)
{
    if (! m_partio_cloud)
        return;   // empty cloud

    if (!m_write) {
        // partio requires this for accelerated lookups
        m_partio_cloud->sort();

        // Create & stash a ParticleAttribute record for each attribute.
        // These will be automatically freed by ~PointCloud when the map
        // destructs.
        for (int i = 0, e = m_partio_cloud->numAttributes();  i < e;  ++i) {
            Partio::ParticleAttribute *a = new Partio::ParticleAttribute();
            m_partio_cloud->attributeInfo (i, *a);
            m_attributes[ustring(a->name)].reset (a);
        }
    }
}



PointCloud::~PointCloud ()
{
    // Save the file if we wrote to it
    if (m_write && !m_filename.empty())
        Partio::write (m_filename.c_str(), *m_partio_cloud);
    if (m_partio_cloud)
        m_partio_cloud->release ();
}



inline Partio::ParticleAttributeType
PartioType (TypeDesc t)
{
    if (t == TypeDesc::TypeFloat)
        return Partio::FLOAT;
    if (t.basetype == TypeDesc::FLOAT && t.aggregate == TypeDesc::VEC3)
        return Partio::VECTOR;
    if (t == TypeDesc::TypeInt)
        return Partio::INT;
    if (t == TypeDesc::TypeString)
        return Partio::INDEXEDSTR;
    return Partio::NONE;
}



// Helper: number of base values
inline int
basevals (TypeDesc t)
{
    return t.numelements() * int(t.aggregate);
}



bool
compatiblePartioType (TypeDesc partio_type, TypeDesc osl_element_type)
{
    // Matching types (treating all VEC3 aggregates as equivalent)...
    if (equivalent (partio_type, osl_element_type))
        return true;

    // Consider arrays and aggregates as interchangeable, as long as the
    // totals are the same.
    if (partio_type.basetype == osl_element_type.basetype &&
        basevals(partio_type) == basevals(osl_element_type))
        return true;

    // The Partio file may contain an array size that OSL can't exactly
    // represent, for example the partio type may be float[4], and the
    // OSL array will be float[] but the element type will be just float
    // because OSL doesn't permit multi-dimensional arrays.
    // Just allow it anyway and fill in the OSL array.
    if (TypeDesc::BASETYPE(partio_type.basetype) == osl_element_type)
        return true;

    return false;
}



TypeDesc
TypeDescOfPartioType (const Partio::ParticleAttribute *ptype)
{
    TypeDesc type;  // default to UNKNOWN
    switch (ptype->type) {
    case Partio::INT:
        type = TypeDesc::INT;
        if (ptype->count > 1)
            type.arraylen = ptype->count;
        break;
    case Partio::FLOAT:
        type = TypeDesc::FLOAT;
        if (ptype->count > 1)
            type.arraylen = ptype->count;
        break;
    case Partio::VECTOR:
        type = TypeDesc (TypeDesc::FLOAT, TypeDesc::VEC3, TypeDesc::NOSEMANTICS);
        if (ptype->count != 3)
            type = TypeDesc::UNKNOWN;  // Must be 3: punt
        break;
    default:
        break;   // Any other future types -- return UNKNOWN
    }
    return type;
}

#endif

}  // anon namespace



int
RendererServices::pointcloud_search (ShaderGlobals *sg,
                                     ustring filename, const OSL::Vec3 &center,
                                     float radius, int max_points, bool sort,
                                     size_t *out_indices,
                                     float *out_distances, int derivs_offset)
{
#ifdef USE_PARTIO
    if (filename.empty())
        return 0;
    PointCloud *pc = PointCloud::get(filename);
    if (pc == NULL) { // The file failed to load
        sg->context->errorf("pointcloud_search: could not open \"%s\"", filename);
        return 0;
    }

    const Partio::ParticlesData *cloud = pc->read_access();
    if (cloud == NULL) { // The file failed to load
        sg->context->errorf("pointcloud_search: could not open \"%s\"", filename);
        return 0;
    }

    // Early exit if the pointcloud contains no particles.
    if (cloud->numParticles() == 0)
       return 0;

    // If we need derivs of the distances, we'll need access to the 
    // found point's positions.
    Partio::ParticleAttribute *pos_attr = NULL;
    if (derivs_offset) {
        pos_attr = pc->m_attributes[u_position].get();
        if (! pos_attr)
            return 0;   // No "position" attribute -- fail
    }

    static_assert (sizeof(size_t) == sizeof(Partio::ParticleIndex),
                   "Partio ParticleIndex should be the size of a size_t");
    // FIXME -- if anybody cares about an architecture in which that is not
    // the case, we can easily allocate local space to retrieve the indices,
    // then copy them back to the caller's indices.

    Partio::ParticleIndex *indices = (Partio::ParticleIndex *)out_indices;
    float *dist2 = out_distances;
    if (! dist2)  // If not supplied, allocate our own
        dist2 = (float *)sg->context->alloc_scratch (max_points*sizeof(float), sizeof(float));

    float finalRadius;
    int count = cloud->findNPoints (&center[0], max_points, radius,
                                    indices, dist2, &finalRadius);

    // If sorting, allocate some temp space and sort the distances and
    // indices at the same time.
    if (sort && count > 1) {
        SortedPointRecord *sorted = (SortedPointRecord *) sg->context->alloc_scratch (count * sizeof(SortedPointRecord), sizeof(SortedPointRecord));
        for (int i = 0;  i < count;  ++i)
            sorted[i] = SortedPointRecord (dist2[i], indices[i]);
        std::sort (sorted, sorted+count, SortedPointCompare());
        for (int i = 0;  i < count;  ++i) {
            dist2[i] = sorted[i].first;
            indices[i] = sorted[i].second;
        }
    }

    if (out_distances) {
        // Convert the squared distances to straight distances
        for (int i = 0; i < count; ++i)
            out_distances[i] = sqrtf(dist2[i]);

        if (derivs_offset) {
            // We are going to need the positions if we need to compute
            // distance derivs
            OSL::Vec3 *positions = (OSL::Vec3 *) sg->context->alloc_scratch (sizeof(OSL::Vec3) * count, sizeof(float));
            // FIXME(Partio): this function really should be marked as const because it is just a wrapper of a private const method
            const_cast<Partio::ParticlesData*>(cloud)->data (*pos_attr, count, indices, true, (void *)positions);
            const OSL::Vec3 &dCdx = (&center)[1];
            const OSL::Vec3 &dCdy = (&center)[2];
            float *d_distance_dx = out_distances + derivs_offset;
            float *d_distance_dy = out_distances + derivs_offset * 2;
            for (int i = 0; i < count; ++i) {
                if (out_distances[i] > 0) {
                    d_distance_dx[i] = 1.0f / out_distances[i] *
                                            ((center.x - positions[i].x) * dCdx.x +
                                             (center.y - positions[i].y) * dCdx.y +
                                             (center.z - positions[i].z) * dCdx.z);
                    d_distance_dy[i] = 1.0f / out_distances[i] *
                                            ((center.x - positions[i].x) * dCdy.x +
                                             (center.y - positions[i].y) * dCdy.y +
                                             (center.z - positions[i].z) * dCdy.z);
                } else {
                    // distance is 0, derivs would be infinite which could cause trouble downstream
                    d_distance_dx[i] = 0;
                    d_distance_dy[i] = 0;
                }
            }
        }
    }
    return count;
#else
    return 0;
#endif
}



int
RendererServices::pointcloud_get (ShaderGlobals *sg,
                                  ustring filename, size_t *indices, int count,
                                  ustring attr_name, TypeDesc attr_type,
                                  void *out_data)
{
#ifdef USE_PARTIO
    if (! count)
        return 1;  // always succeed if not asking for any data

    PointCloud *pc = PointCloud::get(filename);
    if (pc == NULL) { // The file failed to load
        sg->context->errorf("pointcloud_get: could not open \"%s\"", filename);
        return 0;
    }

    const Partio::ParticlesData *cloud = pc->read_access();
    if (cloud == NULL) { // The file failed to load
        sg->context->errorf("pointcloud_get: could not open \"%s\"", filename);
        return 0;
    }

    // lookup the ParticleAttribute pointer needed for a query
    Partio::ParticleAttribute *attr = pc->m_attributes[attr_name].get();
    if (! attr) {
        sg->context->errorf("Accessing unexisting attribute %s in pointcloud \"%s\"", attr_name, filename);
        return 0;
    }

    // Type the partio file contains:
    TypeDesc partio_type = TypeDescOfPartioType (attr);
    // Type the OSL shader has provided in destination array:
    TypeDesc element_type = attr_type.elementtype ();

    // Finally check for some equivalent types like float3 and vector
    if (!compatiblePartioType(partio_type, element_type)) {
        sg->context->errorf("Type of attribute \"%s\" : %s not compatible with OSL's %s in \"%s\" pointcloud",
                            attr_name, partio_type, element_type, filename);
        return 0;
    }

    // For safety, clamp the count to the most that will fit in the output
    int maxn = basevals(attr_type) / basevals(partio_type);
    if (maxn < count) {
        sg->context->errorf("Point cloud attribute \"%s\" : %s with retrieval count %d will not fit in %s",
                            attr_name, partio_type, count, attr_type);
        count = maxn;
    }

    static_assert (sizeof(size_t) == sizeof(Partio::ParticleIndex),
                   "Partio ParticleIndex should be the size of a size_t");
    // FIXME -- if anybody cares about an architecture in which that is not
    // the case, we can easily allocate local space to retrieve the indices,
    // then copy them back to the caller's indices.

    // Actual data query
    const_cast<Partio::ParticlesData *>(cloud)->data (*attr, count, (Partio::ParticleIndex *)indices,
                 true, out_data);
    return 1;
#else
    return 0;
#endif
}



bool
RendererServices::pointcloud_write (ShaderGlobals *sg,
                                    ustring filename, const OSL::Vec3 &pos,
                                    int nattribs, const ustring *names,
                                    const TypeDesc *types,
                                    const void **data)
{
#ifdef USE_PARTIO
    if (filename.empty())
        return false;
    PointCloud *pc = PointCloud::get(filename, true /* create file to write */);
    spin_lock lock (pc->m_mutex);
    Partio::ParticlesDataMutable *cloud = pc->write_access();
    if (cloud == NULL) // The file failed to load
        return false;

    // Mark the pointcloud as written, so we will save it later
    pc->m_write = true;

    // first time only -- add "position" attribute
    if (cloud->numParticles() == 0)
        pc->m_position_attribute = cloud->addAttribute ("position", Partio::VECTOR, 3);

    // Make sure all the attributes mentioned have been added properly
    bool ok = true;
    std::vector<Partio::ParticleAttribute *> partattrs;
    partattrs.reserve (nattribs);
    for (int i = 0;  i < nattribs;  ++i) {
        Partio::ParticleAttribute *a = pc->m_attributes[names[i]].get();
        if (!a) {  // attribute needs to be added
            Partio::ParticleAttributeType pt = PartioType (types[i]);
            if (pt == Partio::NONE) {
                ok = false;
            } else {
                a = new Partio::ParticleAttribute ();
                *a = cloud->addAttribute (names[i].c_str(), pt,
                                          pt==Partio::VECTOR ? 3 : 1 /*count*/);
                pc->m_attributes[names[i]].reset(a);
            }
        }
        partattrs.push_back (a);
    }

    // Make a new particle
    Partio::ParticleIndex p = cloud->addParticle();
    *(Vec3 *)cloud->dataWrite<float>(pc->m_position_attribute, p) = pos;
    for (int i = 0;  i < nattribs;  ++i) {
        Partio::ParticleAttribute *a = partattrs[i];
        if (a  &&  PartioType(types[i]) == a->type) {
            switch (a->type) {
            case Partio::FLOAT :
                *(float *)cloud->dataWrite<float>(*a, p) = *(float *)(data[i]);
                break;
            case Partio::VECTOR :
                *(Vec3 *)cloud->dataWrite<float>(*a, p) = *(Vec3 *)(data[i]);
                break;
            case Partio::INT :
                *(int *)cloud->dataWrite<int>(*a, p) = *(int *)(data[i]);
                break;
            case Partio::INDEXEDSTR :
                // FIXME? do we care?
                break;
            case Partio::NONE :
                break;
            }
        }
    }

    return ok;
#else
    return false;
#endif
}



OSL_SHADEOP int
osl_pointcloud_search (ShaderGlobals *sg, const char *filename, void *center, float radius,
                       int max_points, int sort, void *out_indices, void *out_distances, int derivs_offset,
                       int nattrs, ...)
{
    ShadingSystemImpl &shadingsys (sg->context->shadingsys());
    if (shadingsys.no_pointcloud()) // Debug mode to skip pointcloud expense
        return 0;

    // RS::pointcloud_search takes size_t index array (because of the
    // presumed use of Partio underneath), but OSL only has int, so we
    // have to allocate and copy out.  But, on architectures where int
    // and size_t are the same, we can take a shortcut and let
    // pointcloud_search fill in the array in place (assuming it's
    // passed in the first place).
    size_t *indices;
    if (sizeof(int) == sizeof(size_t) && out_indices)
        indices = (size_t *)out_indices;
    else
        indices = OIIO_ALLOCA(size_t, max_points);

    int count = sg->renderer->pointcloud_search (sg, USTR(filename),
                                                 *((Vec3 *)center), radius, max_points, sort,
                                                 indices, (float *)out_distances, derivs_offset);
    va_list args;
    va_start (args, nattrs);
    for (int i = 0; i < nattrs; i++) {
        ustring  attr_name = ustring::from_unique ((const char *)va_arg (args, const char *));
        long long lltype = va_arg (args, long long);
        TypeDesc attr_type = TYPEDESC (lltype);
        void     *out_data = va_arg (args, void*);
        sg->renderer->pointcloud_get (sg, USTR(filename), indices,
                                      count, attr_name, attr_type, out_data);
    }
    va_end (args);

    // Only copy out if we need to
    if (out_indices  &&  sizeof(int) != sizeof(size_t))
        for(int i = 0; i < count; ++i)
            ((int *)out_indices)[i] = indices[i];

    shadingsys.pointcloud_stats (1, 0, count);

    return count;
}



OSL_SHADEOP int
osl_pointcloud_get (ShaderGlobals *sg, const char *filename, void *in_indices, int count,
                    const char *attr_name, long long attr_type, void *out_data)
{
    ShadingSystemImpl &shadingsys (sg->context->shadingsys());
    if (shadingsys.no_pointcloud()) // Debug mode to skip pointcloud expense
        return 0;

    size_t *indices = OIIO_ALLOCA(size_t, count);
    for (int i = 0; i < count; ++i)
        indices[i] = ((int *)in_indices)[i];

    shadingsys.pointcloud_stats (0, 1, 0);

    return sg->renderer->pointcloud_get (sg, USTR(filename), (size_t *)indices, count, USTR(attr_name),
                                         TYPEDESC(attr_type), out_data);
}



OSL_SHADEOP void
osl_pointcloud_write_helper (ustring *names, TypeDesc *types, void **values,
                             int index, const char *name, long long type,
                             void *val)
{
    names[index] = USTR(name);
    types[index] = TYPEDESC(type);
    values[index] = val;
}



OSL_SHADEOP int
osl_pointcloud_write (ShaderGlobals *sg, const char *filename, const Vec3 *pos,
                      int nattribs, const ustring *names,
                      const TypeDesc *types, const void **values)
{
    ShadingSystemImpl &shadingsys (sg->context->shadingsys());
    if (shadingsys.no_pointcloud()) // Debug mode to skip pointcloud expense
        return 0;

    shadingsys.pointcloud_stats (0, 0, 0, 1);
    return sg->renderer->pointcloud_write (sg, USTR(filename), *pos,
                                           nattribs, names, types, values);
}



