// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <cstdarg>
#include <sstream>

#include "pointcloud.h"

#include "oslexec_pvt.h"

OSL_NAMESPACE_ENTER
namespace pvt {

#ifdef USE_PARTIO
using PointCloudMap
    = std::unordered_map<ustringhash, std::unique_ptr<PointCloud>>;
static PointCloudMap pointclouds;
static OIIO::spin_mutex pointcloudmap_mutex;

PointCloud*
PointCloud::get(ustringhash filename, bool write)
{
    if (filename.empty())
        return nullptr;
    spin_lock lock(pointcloudmap_mutex);
    PointCloudMap::const_iterator found = pointclouds.find(filename);
    if (found != pointclouds.end())
        return found->second.get();
    // Not found. Create a new one.
    Partio::ParticlesDataMutable* partio_cloud = nullptr;
    if (!write) {
        // Mute Partio error prints: by default Partio::read sends errors directly
        // to std::err, but in most cases we want errors to go via errorfmt so the
        // renderer can recognize the message as an error, as we do in
        // pointcloud_search and pointcloud_get.
        std::stringstream m_errorStream;

        partio_cloud = Partio::read(filename.c_str(), false, m_errorStream);
        if (!partio_cloud)
            return nullptr;
    } else {
        partio_cloud = Partio::create();
    }
    PointCloud* pc = new PointCloud(filename, partio_cloud, write);
    pointclouds[filename].reset(pc);
    return pc;
}


PointCloud::PointCloud(ustringhash filename,
                       Partio::ParticlesDataMutable* partio_cloud, bool write)
    : m_filename(filename), m_partio_cloud(partio_cloud), m_write(write)
{
    if (!m_partio_cloud)
        return;  // empty cloud

    if (!m_write) {
        // partio requires this for accelerated lookups
        m_partio_cloud->sort();

        // Create & stash a ParticleAttribute record for each attribute.
        // These will be automatically freed by ~PointCloud when the map
        // destructs.
        for (int i = 0, e = m_partio_cloud->numAttributes(); i < e; ++i) {
            Partio::ParticleAttribute* a = new Partio::ParticleAttribute();
            m_partio_cloud->attributeInfo(i, *a);
            m_attributes[ustring(a->name)].reset(a);
        }
    }
}



PointCloud::~PointCloud()
{
    // Save the file if we wrote to it
    if (m_write && !m_filename.empty())
        Partio::write(m_filename.c_str(), *m_partio_cloud);
    if (m_partio_cloud)
        m_partio_cloud->release();
}
#endif

}  // namespace pvt



int
RendererServices::pointcloud_search(ShaderGlobals* sg, ustringhash filename,
                                    const Vec3& center, float radius,
                                    int max_points, bool sort,
                                    size_t* out_indices, float* out_distances,
                                    int derivs_offset)
{
#ifdef USE_PARTIO
    if (filename.empty())
        return 0;
    PointCloud* pc = PointCloud::get(ustring_from(filename));
    if (pc == NULL) {  // The file failed to load
        sg->context->errorfmt("pointcloud_search: could not open \"{}\"",
                              ustring_from(filename));
        return 0;
    }

    const Partio::ParticlesData* cloud = pc->read_access();
    if (cloud == NULL) {  // The file failed to load
        sg->context->errorfmt("pointcloud_search: could not open \"{}\"",
                              ustring_from(filename));
        return 0;
    }

    // Early exit if the pointcloud contains no particles.
    if (cloud->numParticles() == 0)
        return 0;

    // If we need derivs of the distances, we'll need access to the
    // found point's positions.
    Partio::ParticleAttribute* pos_attr = NULL;
    if (derivs_offset) {
        pos_attr = pc->m_attributes[u_position].get();
        if (!pos_attr)
            return 0;  // No "position" attribute -- fail
    }

    static_assert(sizeof(size_t) == sizeof(Partio::ParticleIndex),
                  "Partio ParticleIndex should be the size of a size_t");
    // FIXME -- if anybody cares about an architecture in which that is not
    // the case, we can easily allocate local space to retrieve the indices,
    // then copy them back to the caller's indices.

    Partio::ParticleIndex* indices = (Partio::ParticleIndex*)out_indices;
    float* dist2                   = out_distances;
    if (!dist2)  // If not supplied, allocate our own
        dist2 = (float*)sg->context->alloc_scratch(max_points * sizeof(float),
                                                   sizeof(float));

    float finalRadius;
    int count = cloud->findNPoints(&center[0], max_points, radius, indices,
                                   dist2, &finalRadius);

    // If sorting, allocate some temp space and sort the distances and
    // indices at the same time.
    if (sort && count > 1) {
        SortedPointRecord* sorted
            = (SortedPointRecord*)sg->context->alloc_scratch(
                count * sizeof(SortedPointRecord), sizeof(SortedPointRecord));
        for (int i = 0; i < count; ++i)
            sorted[i] = SortedPointRecord(dist2[i], indices[i]);
        std::sort(sorted, sorted + count, SortedPointCompare());
        for (int i = 0; i < count; ++i) {
            dist2[i]   = sorted[i].first;
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
            Vec3* positions = (Vec3*)sg->context->alloc_scratch(sizeof(Vec3)
                                                                    * count,
                                                                sizeof(float));
            // FIXME(Partio): this function really should be marked as const because it is just a wrapper of a private const method
            const_cast<Partio::ParticlesData*>(cloud)->data(*pos_attr, count,
                                                            indices, true,
                                                            (void*)positions);
            const Vec3& dCdx     = (&center)[1];
            const Vec3& dCdy     = (&center)[2];
            float* d_distance_dx = out_distances + derivs_offset;
            float* d_distance_dy = out_distances + derivs_offset * 2;
            for (int i = 0; i < count; ++i) {
                if (out_distances[i] > 0) {
                    d_distance_dx[i] = 1.0f / out_distances[i]
                                       * ((center.x - positions[i].x) * dCdx.x
                                          + (center.y - positions[i].y) * dCdx.y
                                          + (center.z - positions[i].z)
                                                * dCdx.z);
                    d_distance_dy[i] = 1.0f / out_distances[i]
                                       * ((center.x - positions[i].x) * dCdy.x
                                          + (center.y - positions[i].y) * dCdy.y
                                          + (center.z - positions[i].z)
                                                * dCdy.z);
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
RendererServices::pointcloud_get(ShaderGlobals* sg, ustringhash filename,
                                 size_t* indices, int count,
                                 ustringhash attr_name, TypeDesc attr_type,
                                 void* out_data)
{
#ifdef USE_PARTIO
    if (!count)
        return 1;  // always succeed if not asking for any data

    PointCloud* pc = PointCloud::get(ustring_from(filename));
    if (pc == NULL) {  // The file failed to load
        sg->context->errorfmt("pointcloud_get: could not open \"{}\"",
                              ustring_from(filename));
        return 0;
    }

    const Partio::ParticlesData* cloud = pc->read_access();
    if (cloud == NULL) {  // The file failed to load
        sg->context->errorfmt("pointcloud_get: could not open \"{}\"",
                              ustring_from(filename));
        return 0;
    }

    // lookup the ParticleAttribute pointer needed for a query
    Partio::ParticleAttribute* attr = pc->m_attributes[attr_name].get();
    if (!attr) {
        sg->context->errorfmt(
            "Accessing unexisting attribute {} in pointcloud \"{}\"", attr_name,
            ustring_from(filename));
        return 0;
    }

    // Type the partio file contains:
    TypeDesc partio_type = TypeDescOfPartioType(attr);
    // Type the OSL shader has provided in destination array:
    TypeDesc element_type = attr_type.elementtype();

    // Finally check for some equivalent types like float3 and vector
    if (!compatiblePartioType(partio_type, element_type)) {
        sg->context->errorfmt(
            "Type of attribute \"{}\" : {} not compatible with OSL's {} in \"{}\" pointcloud",
            attr_name, partio_type, element_type, ustring_from(filename));
        return 0;
    }

    // For safety, clamp the count to the most that will fit in the output
    int maxn = basevals(attr_type) / basevals(partio_type);
    if (maxn < count) {
        sg->context->errorfmt(
            "Point cloud attribute \"{}\" : {} with retrieval count {} will not fit in %s",
            attr_name, partio_type, count, attr_type);
        count = maxn;
    }

    static_assert(sizeof(size_t) == sizeof(Partio::ParticleIndex),
                  "Partio ParticleIndex should be the size of a size_t");
    // FIXME -- if anybody cares about an architecture in which that is not
    // the case, we can easily allocate local space to retrieve the indices,
    // then copy them back to the caller's indices.

    // Actual data query
    if (partio_type == TypeString) {
        // strings are special cases because they are stored as int index
        int* strindices = OSL_ALLOCA(int, count);
        const_cast<Partio::ParticlesData*>(cloud)->data(
            *attr, count, (const Partio::ParticleIndex*)indices,
            /*sorted=*/false, (void*)strindices);
        const auto& strings = cloud->indexedStrs(*attr);
        int sicount         = int(strings.size());
        for (int i = 0; i < count; ++i) {
            int ind = strindices[i];
            if (ind >= 0 && ind < sicount)
                ((ustringrep*)out_data)[i] = ustringrep(strings[ind]);
            else
                ((ustringrep*)out_data)[i] = ustringrep();
        }
    } else {
        // All cases aside from strings are simple.
        const_cast<Partio::ParticlesData*>(cloud)->data(
            *attr, count, (const Partio::ParticleIndex*)indices,
            /*sorted=*/false, out_data);
        // FIXME: it is regrettable that we need this const_cast (and the
        // one a few lines above). It's to work around a bug in partio where
        // they fail to declare this method as const, even though it could
        // be. We should submit a patch to partio to fix this.
    }
    return 1;
#else
    return 0;
#endif
}



bool
RendererServices::pointcloud_write(ShaderGlobals* /*sg*/, ustringhash filename,
                                   const Vec3& pos, int nattribs,
                                   const ustringrep* names,
                                   const TypeDesc* types, const void** data)
{
#ifdef USE_PARTIO
    if (filename.empty())
        return false;
    PointCloud* pc = PointCloud::get(ustring_from(filename),
                                     true /* create file to write */);
    spin_lock lock(pc->m_mutex);
    Partio::ParticlesDataMutable* cloud = pc->write_access();
    if (cloud == NULL)  // The file failed to load
        return false;

    // Mark the pointcloud as written, so we will save it later
    pc->m_write = true;

    // first time only -- add "position" attribute
    if (cloud->numParticles() == 0)
        pc->m_position_attribute = cloud->addAttribute("position",
                                                       Partio::VECTOR, 3);

    // Make sure all the attributes mentioned have been added properly
    bool ok = true;
    std::vector<Partio::ParticleAttribute*> partattrs;
    partattrs.reserve(nattribs);
    for (int i = 0; i < nattribs; ++i) {
        Partio::ParticleAttribute* a = pc->m_attributes[names[i]].get();
        if (!a) {  // attribute needs to be added
            Partio::ParticleAttributeType pt = PartioType(types[i]);
            if (pt == Partio::NONE) {
                ok = false;
            } else {
                a  = new Partio::ParticleAttribute();
                *a = cloud->addAttribute(names[i].c_str(), pt,
                                         pt == Partio::VECTOR ? 3
                                                              : 1 /*count*/);
                pc->m_attributes[names[i]].reset(a);
            }
        }
        partattrs.push_back(a);
    }

    // Make a new particle
    Partio::ParticleIndex p = cloud->addParticle();
    *(Vec3*)cloud->dataWrite<float>(pc->m_position_attribute, p) = pos;
    for (int i = 0; i < nattribs; ++i) {
        Partio::ParticleAttribute* a = partattrs[i];
        if (a && PartioType(types[i]) == a->type) {
            switch (a->type) {
            case Partio::FLOAT:
                *(float*)cloud->dataWrite<float>(*a, p) = *(float*)(data[i]);
                break;
            case Partio::VECTOR:
                *(Vec3*)cloud->dataWrite<float>(*a, p) = *(Vec3*)(data[i]);
                break;
            case Partio::INT:
                *(int*)cloud->dataWrite<int>(*a, p) = *(int*)(data[i]);
                break;
            case Partio::INDEXEDSTR: {
                ustringrep s     = *(ustringrep*)(data[i]);
                const char* sstr = s.c_str();
                int index        = cloud->lookupIndexedStr(*a, sstr);
                if (index == -1)
                    index = cloud->registerIndexedStr(*a, sstr);
                *(int*)cloud->dataWrite<int>(*a, p) = index;
            } break;
            case Partio::NONE: break;
            }
        }
    }

    return ok;
#else
    return false;
#endif
}

namespace pvt {

OSL_SHADEOP int
osl_pointcloud_search(ShaderGlobals* sg, ustring_pod filename, void* center,
                      float radius, int max_points, int sort, void* out_indices,
                      void* out_distances, int derivs_offset, int nattrs, ...)
{
    ShadingSystemImpl& shadingsys(sg->context->shadingsys());
    if (shadingsys.no_pointcloud())  // Debug mode to skip pointcloud expense
        return 0;

    // RS::pointcloud_search takes size_t index array (because of the
    // presumed use of Partio underneath), but OSL only has int, so we
    // have to allocate and copy out.  But, on architectures where int
    // and size_t are the same, we can take a shortcut and let
    // pointcloud_search fill in the array in place (assuming it's
    // passed in the first place).
    size_t* indices;
    if (sizeof(int) == sizeof(size_t) && out_indices)
        indices = (size_t*)out_indices;
    else
        indices = OSL_ALLOCA(size_t, max_points);

    int count
        = sg->renderer->pointcloud_search(sg, USTR(filename), *((Vec3*)center),
                                          radius, max_points, sort, indices,
                                          (float*)out_distances, derivs_offset);
    va_list args;
    va_start(args, nattrs);
    for (int i = 0; i < nattrs; i++) {
        ustring_pod attr_name_rep = (ustring_pod)va_arg(args, ustring_pod);
        ustringrep attr_name      = USTR(attr_name_rep);
        long long lltype          = va_arg(args, long long);
        TypeDesc attr_type        = TYPEDESC(lltype);
        void* out_data            = va_arg(args, void*);
        sg->renderer->pointcloud_get(sg, USTR(filename), indices, count,
                                     attr_name, attr_type, out_data);
    }
    va_end(args);

    // Only copy out if we need to
    if (out_indices && sizeof(int) != sizeof(size_t))
        for (int i = 0; i < count; ++i)
            ((int*)out_indices)[i] = indices[i];

    shadingsys.pointcloud_stats(1, 0, count);

    return count;
}



OSL_SHADEOP int
osl_pointcloud_get(ShaderGlobals* sg, ustring_pod filename, void* in_indices,
                   int count, ustring_pod attr_name, long long attr_type,
                   void* out_data)
{
    ShadingSystemImpl& shadingsys(sg->context->shadingsys());
    if (shadingsys.no_pointcloud())  // Debug mode to skip pointcloud expense
        return 0;

    size_t* indices = OSL_ALLOCA(size_t, count);
    for (int i = 0; i < count; ++i)
        indices[i] = ((int*)in_indices)[i];

    shadingsys.pointcloud_stats(0, 1, 0);

    return sg->renderer->pointcloud_get(sg, USTR(filename), (size_t*)indices,
                                        count, USTR(attr_name),
                                        TYPEDESC(attr_type), out_data);
}



OSL_SHADEOP void
osl_pointcloud_write_helper(ustringrep* names, TypeDesc* types, void** values,
                            int index, ustring_pod name, long long type,
                            void* val)
{
    names[index]  = USTR(name);
    types[index]  = TYPEDESC(type);
    values[index] = val;
}



OSL_SHADEOP int
osl_pointcloud_write(ShaderGlobals* sg, ustring_pod filename, const Vec3* pos,
                     int nattribs, const ustringrep* names,
                     const TypeDesc* types, const void** values)
{
    ShadingSystemImpl& shadingsys(sg->context->shadingsys());
    if (shadingsys.no_pointcloud())  // Debug mode to skip pointcloud expense
        return 0;

    shadingsys.pointcloud_stats(0, 0, 0, 1);
    return sg->renderer->pointcloud_write(sg, USTR(filename), *pos, nattribs,
                                          names, types, values);
}

}  // namespace pvt
OSL_NAMESPACE_EXIT
