// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#ifdef USE_PARTIO
#    include <Partio.h>
#    include <memory>
#    include <unordered_map>
#endif

#include <OSL/oslconfig.h>

OSL_NAMESPACE_ENTER
namespace pvt {

#ifdef USE_PARTIO

class OSLEXECPUBLIC PointCloud {
public:
    PointCloud(ustringhash filename, Partio::ParticlesDataMutable* partio_cloud,
               bool write);
    ~PointCloud();

    PointCloud(const PointCloud&)             = delete;
    PointCloud(const PointCloud&&)            = delete;
    PointCloud& operator=(const PointCloud&)  = delete;
    PointCloud& operator=(const PointCloud&&) = delete;

    static PointCloud* get(ustringhash filename, bool write = false);

    typedef std::unordered_map<ustringhash,
                               std::unique_ptr<Partio::ParticleAttribute>>
        AttributeMap;

    const Partio::ParticlesData* read_access() const
    {
        OSL_DASSERT(!m_write);
        return m_partio_cloud;
    }
    Partio::ParticlesDataMutable* write_access() const
    {
        OSL_DASSERT(m_write);
        return m_partio_cloud;
    }

    ustringhash m_filename;

private:
    // hide just this field, because we want to control how it is accessed
    Partio::ParticlesDataMutable* m_partio_cloud;

public:
    AttributeMap m_attributes;
    bool m_write;
    Partio::ParticleAttribute m_position_attribute;
    OIIO::spin_mutex m_mutex;
};

namespace {  // anon

static ustring u_position("position");

// some helper classes to make the sort easy
typedef std::pair<float, Partio::ParticleIndex> SortedPointRecord;  // dist,index
struct SortedPointCompare {
    bool operator()(const SortedPointRecord& a, const SortedPointRecord& b)
    {
        return a.first < b.first;
    }
};

inline Partio::ParticleAttributeType
PartioType(TypeDesc t)
{
    if (t == TypeFloat)
        return Partio::FLOAT;
    if (t.basetype == TypeDesc::FLOAT && t.aggregate == TypeDesc::VEC3)
        return Partio::VECTOR;
    if (t == TypeInt)
        return Partio::INT;
    if (t == TypeString)
        return Partio::INDEXEDSTR;
    return Partio::NONE;
}



// Helper: number of base values
inline int
basevals(TypeDesc t)
{
    return t.numelements() * int(t.aggregate);
}



bool
compatiblePartioType(TypeDesc partio_type, TypeDesc osl_element_type)
{
    // Matching types (treating all VEC3 aggregates as equivalent)...
    if (equivalent(partio_type, osl_element_type))
        return true;

    // Consider arrays and aggregates as interchangeable, as long as the
    // totals are the same.
    if (partio_type.basetype == osl_element_type.basetype
        && basevals(partio_type) == basevals(osl_element_type))
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
TypeDescOfPartioType(const Partio::ParticleAttribute* ptype)
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
        type = TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3, TypeDesc::NOSEMANTICS);
        if (ptype->count != 3)
            type = TypeDesc::UNKNOWN;  // Must be 3: punt
        break;
    case Partio::INDEXEDSTR: type = TypeDesc::STRING; break;
    default: break;  // Any other future types -- return UNKNOWN
    }
    return type;
}

}  // namespace

#endif

}  // namespace pvt
OSL_NAMESPACE_EXIT
