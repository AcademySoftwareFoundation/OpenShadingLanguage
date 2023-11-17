// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of hash operations.
///
/////////////////////////////////////////////////////////////////////////

#include <cstdarg>

#include "pointcloud.h"

#include <OSL/oslconfig.h>

#include <OSL/batched_rendererservices.h>
#include <OSL/batched_shaderglobals.h>
#include <OSL/wide.h>

#include "oslexec_pvt.h"

OSL_NAMESPACE_ENTER
namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)

using BatchedRendererServices = OSL::BatchedRendererServices<__OSL_WIDTH>;
using WidthTag                = OSL::WidthOf<__OSL_WIDTH>;
using PointCloudSearchResults = BatchedRendererServices::PointCloudSearchResults;

#include "define_opname_macros.h"

namespace {

OSL_FORCEINLINE void
default_pointcloud_search(BatchedShaderGlobals* bsg, ustringhash filename,
                          const void* wcenter_, Wide<const float> wradius,
                          int max_points, bool sort,
                          PointCloudSearchResults& results)
{
#ifdef USE_PARTIO
    ShadingContext* ctx = bsg->uniform.context;

    if (filename.empty()) {
        assign_all(results.wnum_points(), 0);
        return;
    }
    PointCloud* pc = PointCloud::get(filename);
    if (pc == NULL) {  // The file failed to load
        ctx->batched<__OSL_WIDTH>().errorfmt(
            results.mask(), "pointcloud_search: could not open \"{}\"",
            filename);
        assign_all(results.wnum_points(), 0);
        return;
    }

    const Partio::ParticlesData* cloud = pc->read_access();
    if (cloud == NULL) {  // The file failed to load
        ctx->batched<__OSL_WIDTH>().errorfmt(
            results.mask(), "pointcloud_search: could not open \"{}\"",
            filename);
        assign_all(results.wnum_points(), 0);
        return;
    }

    // Early exit if the pointcloud contains no particles.
    if (cloud->numParticles() == 0) {
        assign_all(results.wnum_points(), 0);
        return;
    }

    // If we need derivs of the distances, we'll need access to the
    // found point's positions.
    Partio::ParticleAttribute* pos_attr = NULL;
    if (results.distances_have_derivs()) {
        pos_attr = pc->m_attributes[u_position].get();
        if (!pos_attr) {
            // No "position" attribute -- fail
            assign_all(results.wnum_points(), 0);
            return;
        }
    }

    static_assert(sizeof(size_t) == sizeof(Partio::ParticleIndex),
                  "Partio ParticleIndex should be the size of a size_t");
    // FIXME -- if anybody cares about an architecture in which that is not
    // the case, we can easily allocate local space to retrieve the indices,
    // then copy them back to the caller's indices.

    // Partio needs size_t indices, and our batched representation is
    // structure of arrays (wide) so we need a scalar temporary
    // indices array
    // TODO: evaluate if sg->context->alloc_scratch should be used instead?
    Partio::ParticleIndex* indices
        = (Partio::ParticleIndex*)OSL_ALLOCA(size_t, max_points);

    Wide<const OSL::Vec3> wcenter(wcenter_);

    // Partio uses array of structure data layout for distances,
    // and our batched representation is
    // structure of arrays (wide) so we need a scalar temporary
    // distances array
    float* dist2              = OSL_ALLOCA(float, max_points);
    SortedPointRecord* sorted = OSL_ALLOCA(SortedPointRecord, max_points);
    auto windices             = results.windices();
    auto wnum_points          = results.wnum_points();
    results.mask().foreach ([=](ActiveLane lane) -> void {
        const OSL::Vec3 center = wcenter[lane];

        const float radius = wradius[lane];
        float finalRadius;
        int count = cloud->findNPoints(&center[0], max_points, radius, indices,
                                       dist2, &finalRadius);

        // If sorting, allocate some temp space and sort the distances and
        // indices at the same time.
        if (sort && count > 1) {
            //SortedPointRecord *sorted = (SortedPointRecord *) sg->context->alloc_scratch (count * sizeof(SortedPointRecord), sizeof(SortedPointRecord));
            //SortedPointRecord *sorted = OSL_ALLOCA(SortedPointRecord, count);
            for (int i = 0; i < count; ++i)
                sorted[i] = SortedPointRecord(dist2[i], indices[i]);
            std::sort(sorted, sorted + count, SortedPointCompare());
            for (int i = 0; i < count; ++i) {
                dist2[i]   = sorted[i].first;
                indices[i] = sorted[i].second;
            }
        }

        // copy scalar indices out to wide results
        auto out_indices = windices[lane];
        for (int i = 0; i < count; ++i) {
            int indice     = static_cast<int>(indices[i]);
            out_indices[i] = indice;
        }

        if (results.has_distances()) {
            auto wdistances    = results.wdistances();
            auto out_distances = wdistances[lane];
            // The 'out_distances' lane proxy object holds a reference to
            // the Masked<float[]> returned from results.wdistances().
            // So make sure that the Masked<float[]> is not a temporary object.
            // IE: DO NOT DO THIS!
            // auto out_distances = results.wdistances()[lane];

            // Convert the squared distances to straight distances
            for (int i = 0; i < count; ++i) {
                float dist       = sqrtf(dist2[i]);
                out_distances[i] = dist;
            }

            if (results.distances_have_derivs()) {
                // We are going to need the positions if we need to compute
                // distance derivs
                //OSL::Vec3 *positions = (OSL::Vec3 *) sg->context->alloc_scratch (sizeof(OSL::Vec3) * count, sizeof(float));
                OSL::Vec3* positions = OSL_ALLOCA(OSL::Vec3, count);
                // FIXME(Partio): this function really should be marked as const because it is just a wrapper of a private const method
                const_cast<Partio::ParticlesData*>(cloud)->data(
                    *pos_attr, count, indices, true, (void*)positions);

                Wide<const Dual2<OSL::Vec3>> wdcenter(wcenter_);
                const Dual2<OSL::Vec3> dcenter = wdcenter[lane];

                const OSL::Vec3& dCval = dcenter.val();
                const OSL::Vec3& dCdx  = dcenter.dx();
                const OSL::Vec3& dCdy  = dcenter.dy();
                auto wdistances_dx     = results.wdistancesDx();
                auto wdistances_dy     = results.wdistancesDy();
                auto d_distance_dx     = wdistances_dx[lane];
                auto d_distance_dy     = wdistances_dy[lane];
                for (int i = 0; i < count; ++i) {
                    if (out_distances[i] > 0.0f) {
                        d_distance_dx[i]
                            = 1.0f / out_distances[i]
                              * ((dCval.x - positions[i].x) * dCdx.x
                                 + (dCval.y - positions[i].y) * dCdx.y
                                 + (dCval.z - positions[i].z) * dCdx.z);
                        d_distance_dy[i]
                            = 1.0f / out_distances[i]
                              * ((dCval.x - positions[i].x) * dCdy.x
                                 + (dCval.y - positions[i].y) * dCdy.y
                                 + (dCval.z - positions[i].z) * dCdy.z);
                    } else {
                        // distance is 0, derivs would be infinite which could cause trouble downstream
                        d_distance_dx[i] = 0.0f;
                        d_distance_dy[i] = 0.0f;
                    }
                }
            }
        }
        wnum_points[lane] = count;
    });
#else
    assign_all(results.wnum_points(), 0);
#endif
}



OSL_FORCEINLINE void
dispatch_pointcloud_search(BatchedShaderGlobals* bsg, ustringhash filename,
                           const void* wcenter, Wide<const float> wradius,
                           int max_points, bool sort,
                           PointCloudSearchResults& results)
{
    auto* bsr = bsg->uniform.renderer->batched(WidthTag());
    if (bsr->is_overridden_pointcloud_search()) {
        return bsr->pointcloud_search(bsg, filename, wcenter, wradius,
                                      max_points, sort, results);
    } else {
        return default_pointcloud_search(bsg, filename, wcenter, wradius,
                                         max_points, sort, results);
    }
}



OSL_FORCEINLINE Mask
default_pointcloud_get(BatchedShaderGlobals* bsg, ustringhash filename,
                       Wide<const int[]> windices, Wide<const int> wnum_points,
                       ustringhash attr_name, MaskedData wout_data)
{
#ifdef USE_PARTIO
    Mask success { false };
    ShadingContext* ctx = bsg->uniform.context;

    PointCloud* pc = PointCloud::get(filename);
    // defer reporting errors as only lanes with non zero num_points
    // should report errors
    const Partio::ParticlesData* cloud = nullptr;
    if (pc != nullptr) {
        cloud = pc->read_access();
    }
    Partio::ParticleAttribute* attr = nullptr;
    if (cloud != nullptr) {
        attr = pc->m_attributes[attr_name].get();
    }

    TypeDesc attr_type = wout_data.type();
    // Type the OSL shader has provided in destination array:
    TypeDesc element_type = attr_type.elementtype();

    // Type the partio file contains:
    TypeDesc partio_type;
    int* strindices                = nullptr;
    void* aos_buffer               = nullptr;
    bool is_compatible_with_partio = false;
    int maxn                       = 0;
    if (attr != nullptr) {
        partio_type               = TypeDescOfPartioType(attr);
        is_compatible_with_partio = compatiblePartioType(partio_type,
                                                         element_type);
        maxn                      = basevals(attr_type) / basevals(partio_type);
        if (partio_type == OIIO::TypeString) {
            // strings are special cases because they are stored as int index
            // Ensure alloca's happen outside loops
            strindices = OSL_ALLOCA(int, attr_type.numelements());
        } else {
            aos_buffer = OSL_ALLOCA(char, attr_type.size());
        }
    }

    Partio::ParticleIndex* indices
        = (Partio::ParticleIndex*)OSL_ALLOCA(size_t, windices.length());

    wout_data.mask().foreach ([=, &success](ActiveLane lane) -> void {
        int count = wnum_points[lane];
        if (!count) {
            success.set_on(lane);  // always succeed if not asking for any data
            return;
        }

        if (pc == nullptr) {  // The file failed to load
            ctx->batched<__OSL_WIDTH>().errorfmt(
                Mask { lane }, "pointcloud_get: could not open \"{}\"",
                filename);
            return;
        }

        if (cloud == nullptr) {  // The file failed to load
            ctx->batched<__OSL_WIDTH>().errorfmt(
                Mask { lane }, "pointcloud_get: could not open \"{}\"",
                filename);
            return;
        }

        // lookup the ParticleAttribute pointer needed for a query
        if (attr == nullptr) {
            ctx->batched<__OSL_WIDTH>().errorfmt(
                Mask { lane },
                "Accessing unexisting attribute {} in pointcloud \"{}\"",
                attr_name, filename);
            return;
        }


        // Finally check for some equivalent types like float3 and vector
        if (!is_compatible_with_partio) {
            ctx->batched<__OSL_WIDTH>().errorfmt(
                Mask { lane },
                "Type of attribute \"{}\" : {} not compatible with OSL's {} in \"{}\" pointcloud",
                attr_name, partio_type, element_type, filename);
            return;
        }

        // For safety, clamp the count to the most that will fit in the output
        if (maxn < count) {
            ctx->batched<__OSL_WIDTH>().errorfmt(
                Mask { lane },
                "Point cloud attribute \"{}\" : {} with retrieval count {} will not fit in {}",
                attr_name, partio_type, count, attr_type);
            count = maxn;
        }
        // Copy int indices out of SOA wide format into local AOS size_t
        auto int_indices = windices[lane];
        for (int i = 0; i < count; ++i) {
            indices[i] = int_indices[i];
        }

        static_assert(sizeof(size_t) == sizeof(Partio::ParticleIndex),
                      "Partio ParticleIndex should be the size of a size_t");
        // FIXME -- if anybody cares about an architecture in which that is not
        // the case, we can easily allocate local space to retrieve the indices,
        // then copy them back to the caller's indices.

        // Actual data query
        if (partio_type == OIIO::TypeString) {
            // strings are special cases because they are stored as int index
            const_cast<Partio::ParticlesData*>(cloud)->data(
                *attr, count, (const Partio::ParticleIndex*)indices,
                /*sorted*/ /*true*/ false, (void*)strindices);
            const auto& strings = cloud->indexedStrs(*attr);
            int sicount         = int(strings.size());
            OSL_DASSERT(Masked<ustring[]>::is(wout_data));
            Masked<ustring[]> wout_strings(wout_data);
            auto out_strings = wout_strings[lane];
            for (int i = 0; i < count; ++i) {
                int ind = strindices[i];
                if (ind >= 0 && ind < sicount)
                    out_strings[i] = ustring(strings[ind]);
                else
                    out_strings[i] = ustring();
            }
        } else {
            // All cases aside from strings are simple.
            const_cast<Partio::ParticlesData*>(cloud)->data(
                *attr, count, (const Partio::ParticleIndex*)indices,
                /*sorted*/ /*true*/ false, aos_buffer);
            // FIXME: it is regrettable that we need this const_cast (and the
            // one a few lines above). It's to work around a bug in partio where
            // they fail to declare this method as const, even though it could
            // be. We should submit a patch to partio to fix this.
            wout_data.assign_val_lane_from_scalar(lane, aos_buffer);
        }
        success.set_on(lane);
    });
    return success;
#else
    return Mask { false };
#endif
}



OSL_FORCEINLINE Mask
dispatch_pointcloud_get(BatchedShaderGlobals* bsg, ustringhash filename,
                        Wide<const int[]> windices, Wide<const int> wnum_points,
                        ustringhash attr_name, MaskedData wout_data)
{
    auto* bsr = bsg->uniform.renderer->batched(WidthTag());
    if (bsr->is_overridden_pointcloud_get()) {
        return bsr->pointcloud_get(bsg, filename, windices, wnum_points,
                                   attr_name, wout_data);
    } else {
        return default_pointcloud_get(bsg, filename, windices, wnum_points,
                                      attr_name, wout_data);
    }
}



Mask
default_pointcloud_write(BatchedShaderGlobals* bsg, ustringhash filename,
                         Wide<const OSL::Vec3> wpos, int nattribs,
                         const ustring* attr_names, const TypeDesc* attr_types,
                         const void** ptrs_to_wide_attr_value, Mask mask)
{
#ifdef USE_PARTIO
    if (filename.empty())
        return Mask { false };

    PointCloud* pc = PointCloud::get(filename, true /* create file to write */);
    spin_lock lock(pc->m_mutex);
    Partio::ParticlesDataMutable* cloud = pc->write_access();
    if (cloud == NULL)  // The file failed to load
        return Mask { false };

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
        const ustring attr_name      = attr_names[i];
        Partio::ParticleAttribute* a = pc->m_attributes[attr_name].get();
        if (!a) {  // attribute needs to be added
            Partio::ParticleAttributeType pt = PartioType(attr_types[i]);
            if (pt == Partio::NONE) {
                ok = false;
            } else {
                a  = new Partio::ParticleAttribute();
                *a = cloud->addAttribute(attr_name.c_str(), pt,
                                         pt == Partio::VECTOR ? 3
                                                              : 1 /*count*/);
                pc->m_attributes[attr_name].reset(a);
            }
        }
        partattrs.push_back(a);
    }

    mask.foreach ([=](ActiveLane lane) -> void {
        // Make a new particle
        Partio::ParticleIndex p = cloud->addParticle();
        const Vec3 pos          = wpos[lane];
        *(Vec3*)cloud->dataWrite<float>(pc->m_position_attribute, p) = pos;
        for (int i = 0; i < nattribs; ++i) {
            Partio::ParticleAttribute* a       = partattrs[i];
            const void* ptr_to_wide_attr_value = ptrs_to_wide_attr_value[i];
            if (a && PartioType(attr_types[i]) == a->type) {
                switch (a->type) {
                case Partio::FLOAT: {
                    Wide<const float> wdata(ptr_to_wide_attr_value);
                    *(float*)cloud->dataWrite<float>(*a, p) = wdata[lane];
                } break;
                case Partio::VECTOR: {
                    Wide<const Vec3> wdata(ptr_to_wide_attr_value);
                    *(Vec3*)cloud->dataWrite<float>(*a, p) = wdata[lane];
                } break;
                case Partio::INT: {
                    Wide<const int> wdata(ptr_to_wide_attr_value);
                    *(int*)cloud->dataWrite<int>(*a, p) = wdata[lane];
                } break;
                case Partio::INDEXEDSTR: {
                    Wide<const ustring> wdata(ptr_to_wide_attr_value);
                    ustring ustr  = wdata[lane];
                    const char* s = ustr.c_str();
                    int index     = cloud->lookupIndexedStr(*a, s);
                    if (index == -1)
                        index = cloud->registerIndexedStr(*a, s);
                    *(int*)cloud->dataWrite<int>(*a, p) = index;
                } break;
                case Partio::NONE: break;
                }
            }
        }
    });

    return ok ? mask : Mask { false };
#else
    return Mask { false };
#endif
}



OSL_FORCEINLINE Mask
dispatch_pointcloud_write(BatchedShaderGlobals* bsg, ustringhash filename,
                          Wide<const OSL::Vec3> wpos, int nattribs,
                          const ustring* attr_names, const TypeDesc* attr_types,
                          const void** ptrs_to_wide_attr_value, Mask mask)
{
    auto* bsr = bsg->uniform.renderer->batched(WidthTag());
    if (bsr->is_overridden_pointcloud_write()) {
        return bsr->pointcloud_write(bsg, filename, wpos, nattribs, attr_names,
                                     attr_types, ptrs_to_wide_attr_value, mask);
    } else {
        return default_pointcloud_write(bsg, filename, wpos, nattribs,
                                        attr_names, attr_types,
                                        ptrs_to_wide_attr_value, mask);
    }
}

}  // namespace



OSL_BATCHOP void
__OSL_MASKED_OP(pointcloud_search)(
    BatchedShaderGlobals* bsg, void* wout_num_points_, ustring_pod filename,
    const void* wcenter_, void* wradius_, int max_points, int sort,
    void* wout_indices_, int indices_array_length, void* wout_distances_,
    int distances_array_length, int distances_has_derivs, int mask_value,
    int nattrs, ...)
{
    if (wout_indices_ == nullptr) {
        wout_indices_        = OSL_ALLOCA(Block<int>, max_points);
        indices_array_length = max_points;
    }
    PointCloudSearchResults pcsr { wout_num_points_,
                                   wout_indices_,
                                   indices_array_length,
                                   wout_distances_,
                                   distances_array_length,
                                   static_cast<bool>(distances_has_derivs),
                                   mask_value };

    ShadingContext* ctx = bsg->uniform.context;
    if (ctx->shadingsys().no_pointcloud()) {
        // Debug mode to skip pointcloud expense
        assign_all(pcsr.wnum_points(), 0);
        return;  // mask_value;
    }

    Wide<const float> wradius(wradius_);

    dispatch_pointcloud_search(bsg, USTR(filename).uhash(), wcenter_, wradius,
                               max_points, sort, pcsr);

    if (nattrs > 0) {
        Wide<const int[]> windices { wout_indices_, indices_array_length };
        Wide<const int> wnum_points { wout_num_points_ };

        va_list args;
        va_start(args, nattrs);
        for (int i = 0; i < nattrs; i++) {
            ustringrep attr_name = USTREP(va_arg(args, ustring_pod));
            TypeDesc attr_type   = TYPEDESC(va_arg(args, long long));
            void* out_data       = va_arg(args, void*);
            dispatch_pointcloud_get(
                bsg, USTR(filename), windices, wnum_points, attr_name,
                MaskedData { attr_type, false, Mask { mask_value }, out_data });
        }
        va_end(args);
    }

    // Ignore stats for now, batched stats would need per lane counts
    //shadingsys.pointcloud_stats (1, 0, count);
}



OSL_BATCHOP int
__OSL_MASKED_OP(pointcloud_get)(BatchedShaderGlobals* bsg, ustring_pod filename,
                                void* windices_, int indices_array_length,
                                void* wnum_points_, ustring_pod attr_name,
                                long long attr_type_, void* wout_data_,
                                int mask_value)
{
    ShadingContext* ctx = bsg->uniform.context;
    if (ctx->shadingsys()
            .no_pointcloud())  // Debug mode to skip pointcloud expense
        return 0;              // mask_value;

    //shadingsys.pointcloud_stats (0, 1, 0);
    Wide<const int[]> windices { windices_, indices_array_length };
    Wide<const int> wnum_points { wnum_points_ };
    TypeDesc attr_type = TYPEDESC(attr_type_);

    Mask success = dispatch_pointcloud_get(
        bsg, USTR(filename).uhash(), windices, wnum_points,
        USTR(attr_name).uhash(),
        MaskedData { attr_type, false, Mask { mask_value }, wout_data_ });
    return success.value();
}



OSL_BATCHOP int
__OSL_MASKED_OP(pointcloud_write)(BatchedShaderGlobals* bsg,
                                  ustring_pod filename, const void* wpos_,
                                  int nattribs, const ustringrep* attr_names,
                                  const TypeDesc* attr_types,
                                  const void** ptrs_to_wide_attr_value,
                                  int mask_value)
{
    ShadingContext* ctx = bsg->uniform.context;

    // Debug mode to skip pointcloud expense
    if (ctx->shadingsys().no_pointcloud())
        return 0;  // mask_value;

    Wide<const OSL::Vec3> wpos(wpos_);

    Mask success = dispatch_pointcloud_write(bsg, USTR(filename).uhash(), wpos,
                                             nattribs, attr_names, attr_types,
                                             ptrs_to_wide_attr_value,
                                             Mask(mask_value));
    return success.value();
}

}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_EXIT
