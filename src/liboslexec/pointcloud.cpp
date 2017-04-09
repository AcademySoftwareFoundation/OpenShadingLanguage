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

#if USE_PARTIO
#include <Partio.h>
#include <boost/unordered_map.hpp>
#endif



namespace { // anon

#if USE_PARTIO

class PointCloud {
public:
    PointCloud (ustring filename, Partio::ParticlesDataMutable *partio_cloud, bool write);
    ~PointCloud ();
    static PointCloud *get (ustring filename, bool write = false);

    typedef boost::unordered_map<ustring, shared_ptr<Partio::ParticleAttribute>, ustringHash> AttributeMap;
    // N.B./FIXME(C++11): shared_ptr is probably overkill, but
    // scoped_ptr is not copyable and therefore can't be used in
    // standard containers.  When C++11 is uniquitous, unique_ptr is the
    // one that should really be used.

    const Partio::ParticlesData* read_access() const { DASSERT(!m_write); return m_partio_cloud; }
    Partio::ParticlesDataMutable* write_access() const { DASSERT(m_write); return m_partio_cloud; }

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


typedef boost::unordered_map<ustring, shared_ptr<PointCloud>, ustringHash> PointCloudMap;
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
    if (! filename)
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
    if (m_write && m_filename)
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



inline bool
compatiblePartioType (Partio::ParticleAttribute *received, int expected)
{
    return ((expected == Partio::VECTOR) &&
            (received->type == expected || received->type == Partio::FLOAT) &&
            received->count == 3) ||
        (received->type == expected && received->count == 1);
}



inline const char *
partioTypeString(Partio::ParticleAttribute *ptype)
{
    switch (ptype->type) {
    case Partio::INT:    return "int";
    case Partio::FLOAT:  return "float";
    case Partio::VECTOR: return "vector";
    default:             return "none";
    }
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
#if USE_PARTIO
    if (! filename)
        return 0;
    PointCloud *pc = PointCloud::get(filename);
    if (pc == NULL) { // The file failed to load
        sg->context->error ("pointcloud_search: could not open \"%s\"", filename.c_str());
        return 0;
    }

    const Partio::ParticlesData *cloud = pc->read_access();
    if (cloud == NULL) { // The file failed to load
        sg->context->error ("pointcloud_search: could not open \"%s\"", filename.c_str());
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

    ASSERT (sizeof(size_t) == sizeof(Partio::ParticleIndex) &&
            "Only will work if Partio ParticleIndex is the size of a size_t");
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
#if USE_PARTIO
    if (! count)
        return 1;  // always succeed if not asking for any data

    PointCloud *pc = PointCloud::get(filename);
    if (pc == NULL) { // The file failed to load
        sg->context->error ("pointcloud_get: could not open \"%s\"", filename.c_str());
        return 0;
    }

    const Partio::ParticlesData *cloud = pc->read_access();
    if (cloud == NULL) { // The file failed to load
        sg->context->error ("pointcloud_get: could not open \"%s\"", filename.c_str());
        return 0;
    }

    // lookup the ParticleAttribute pointer needed for a query
    Partio::ParticleAttribute *attr = pc->m_attributes[attr_name].get();
    if (! attr) {
        sg->context->error ("Accessing unexisting attribute %s in pointcloud \"%s\"", attr_name.c_str(), filename.c_str());
        return 0;
    }

    // Now make sure that types are compatible
    TypeDesc element_type = attr_type.elementtype ();
    int attr_partio_type = 0;

    // Convert the OSL (OIIO) type to the equivalent Partio type
    if (element_type == TypeDesc::TypeFloat)
        attr_partio_type = Partio::FLOAT;
    else if (element_type == TypeDesc::TypeInt)
        attr_partio_type = Partio::INT;
    else if (element_type == TypeDesc::TypeColor  || element_type == TypeDesc::TypePoint ||
             element_type == TypeDesc::TypeVector || element_type == TypeDesc::TypeNormal)
        attr_partio_type = Partio::VECTOR;
    else {
        // error ("Unsupported attribute type %s for pointcloud query in attribute %s",
        //       element_type.c_str(), attr_name.c_str());
        return 0;
    }

    // Finally check for some equivalent types like float3 and vector
    if (!compatiblePartioType(attr, attr_partio_type)) {
        sg->context->error ("Type of attribute \"%s\" : %s[%d] not compatible with OSL's %s in \"%s\" pointcloud",
                    attr_name.c_str(), partioTypeString(attr), attr->count,
                    element_type.c_str(), filename.c_str());
        return 0;
    }

    ASSERT (sizeof(size_t) == sizeof(Partio::ParticleIndex) &&
            "Only will work if Partio ParticleIndex is the size of a size_t");
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
#if USE_PARTIO
    if (! filename)
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
        indices = (size_t *)alloca (sizeof(size_t) * max_points);

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

    size_t *indices = (size_t *)alloca (sizeof(size_t) * count);
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



