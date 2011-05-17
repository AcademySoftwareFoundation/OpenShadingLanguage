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


#include "oslexec.h"
#include "simplerend.h"
using namespace OSL;

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {


bool
SimpleRenderer::get_matrix (Matrix44 &result, TransformationPtr xform,
                            float time)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    result = *(OSL::Matrix44 *)xform;
    return true;
}



bool
SimpleRenderer::get_matrix (Matrix44 &result, ustring from, float time)
{
    TransformMap::const_iterator found = m_named_xforms.find (from);
    if (found != m_named_xforms.end()) {
        result = *(found->second);
        return true;
    } else {
        return false;
    }
}


void
SimpleRenderer::name_transform (const char *name, const OSL::Matrix44 &xform)
{
    shared_ptr<Transformation> M (new OSL::Matrix44 (xform));
    m_named_xforms[ustring(name)] = M;
}

bool
SimpleRenderer::get_array_attribute (void *renderstate, bool derivatives, ustring object,
                                     TypeDesc type, ustring name,
                                     int index, void *val)
{
    return false;
}

bool
SimpleRenderer::get_attribute (void *renderstate, bool derivatives, ustring object,
                               TypeDesc type, ustring name, void *val)
{
    return false;
}

bool
SimpleRenderer::get_userdata (bool derivatives, ustring name, TypeDesc type, void *renderstate, void *val)
{
    return false;
}

bool
SimpleRenderer::has_userdata (ustring name, TypeDesc type, void *renderstate)
{
    return false;
}

void *
SimpleRenderer::get_pointcloud_attr_query (ustring *attr_names, TypeDesc *attr_types,
                                           bool derivatives, int nattrs)
{
#if 0
    // Example code of how to cache some useful query info to
    // save time quering Partio for pointclouds

    m_attr_queries.push_back(AttrQuery());
    AttrQuery &query = m_attr_queries.back();
    // Make space for what we need. The only reason to use
    // std::vector is to skip the delete
    query.attr_names.resize(nattrs);
    query.attr_partio_types.resize(nattrs);
    // capacity will keep the length of the smallest array passed
    // to the query. Just to prevent buffer overruns
    query.capacity = -1;

    for (int i = 0; i < nattrs; ++i)
    {
        query.attr_names[i] = attr_names[i];
        TypeDesc element_type = attr_types[i].elementtype ();
        if (query.capacity < 0)
           query.capacity = attr_types[i].numelements();
        else
           query.capacity = MIN(query.capacity, (int)attr_types[i].numelements());

        // Convert the OSL (OIIO) type to the equivalent Partio type so
        // we can do a fast check at query time.
        if (element_type == TypeDesc::TypeFloat)
           query.attr_partio_types[i] = Partio::FLOAT;
        else if (element_type == TypeDesc::TypeInt)
           query.attr_partio_types[i] = Partio::INT;
        else if (element_type == TypeDesc::TypeColor  || element_type == TypeDesc::TypePoint ||
                 element_type == TypeDesc::TypeVector || element_type == TypeDesc::TypeNormal)
           query.attr_partio_types[i] = Partio::VECTOR;
        else
        {
            // Report some error of unknown type
            return NULL;
        }
    }
    // This is valid until the end of RenderServices
    return &query;
#else
    return NULL;
#endif
}

int
SimpleRenderer::pointcloud (ustring filename, const OSL::Vec3 &center, float radius,
                            int max_points, void *_attr_query, void **attr_outdata)
{
#if 0
    // Example code of how to query Partio for this pointcloud lookup
    // using some cached that in attr_query

    if (!_attr_query)
        return 0;
    AttrQuery *attr_query = (AttrQuery *)_attr_query;
    if (attr_query->capacity < max_points)
        return 0;

    // Get the pointcloud entry for the given filename
    Partio::ParticleData * cloud = get_pointcloud(filename);

    // Now we have to look up all the attributes in the file. We can't do this
    // before hand cause we never know what we are going to load.
    int nattrs = attr_query->attr_names.size();
    Partio::ParticleAttribute **attr = (Partio::ParticleAttribute **)alloca (sizeof(Partio::ParticleAttribute *) * nattrs );
    for (int i = 0; i < nattrs; ++i)
    {
        // Special case attributes
        if (attr_query->attr_names[i] == u_distance || attr_query->attr_names[i] == u_index)
            continue;
        // lookup the ParticleAttribute pointer, left unimplemented ...
        attr[i] = partio_attr_by_name(cloud, attr_query->attr_names[i]);
        if (attr[i]->type != attr_query->attr_partio_types[i])
        {
            // Issue an error here and return, types don't match
        }
    }

    std::vector<Partio::ParticleIndex *> indices;
    std::vector<float>                   dist2;

    // Finally, do the lookup
    entry->cloud->findNPoints((const float *)&center, max_points, radius, indices, dist2);
    int count = indices.size();

    // Retrieve the attributes directly to user space.
    for (int j = 0; j < nattrs; ++j)
    {
        // special cases
        if (attr_query->attr_names[j] == u_distance)
        {
           for (int i = 0; i < count; ++i)
              ((float *)attr_outdata[j])[i] = sqrtf(dist2[i]);
        }
        else if (attr_query->attr_names[j] == u_index)
        {
           for (int i = 0; i < count; ++i)
              ((int *)attr_outdata[j])[i] = indices[i];
        }
        else if (attr[j])
        {
           // Note we make a single call per attribute, we don't loop over the
           // points. Partio does it, so it is there that we have to care about
           // performance

           entry->cloud->data (*attr[j], count, &indices[0], true /* What is this sorted flag? */, attr_outdata[j]);
        }
    }
    return count;
#else
    return 0;
#endif
}

};  // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
