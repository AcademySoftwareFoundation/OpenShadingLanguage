// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "rapidobj/rapidobj.hpp"
#include "raytracer.h"

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/timer.h>

OSL_NAMESPACE_BEGIN

void
Scene::add_model(const std::string& filename, const ShaderMap& shadermap,
                 int shaderID, OIIO::ErrorHandler& errhandler)
{
    OIIO::Timer timer;
    rapidobj::MaterialLibrary materials = rapidobj::MaterialLibrary::Default(
        rapidobj::Load::Optional);
    rapidobj::Result obj_file = rapidobj::ParseFile(filename, materials);
    if (obj_file.error) {
        // we were unable to parse the scene
        errhandler.errorfmt("Error while reading {} - {}", filename,
                            obj_file.error.code.message());
        return;
    }
    if (!Triangulate(obj_file)) {
        errhandler.errorfmt("Unable to triangulate model from {}", filename);
        return;
    }
    std::vector<int> material_ids;
    bool use_material_ids = false;
    if (!shadermap.empty()) {
        // try to match up obj materials with existing shader names in our shadermap
        for (auto&& mat : obj_file.materials) {
            auto it = shadermap.find(mat.name);
            if (it != shadermap.end()) {
                errhandler.infofmt("Found material {}", mat.name);
                material_ids.emplace_back(it->second);
                use_material_ids = true;
            } else {
                material_ids.emplace_back(shaderID);
                errhandler.infofmt("Material {} did not match any shader names",
                                   mat.name);
            }
        }
    }
    int nverts   = obj_file.attributes.positions.size() / 3;
    int base_idx = verts.size();
    verts.reserve(verts.size() + nverts);
    for (int i = 0, i3 = 0; i < nverts; i++, i3 += 3) {
        float x = obj_file.attributes.positions[i3 + 0];
        float y = obj_file.attributes.positions[i3 + 1];
        float z = obj_file.attributes.positions[i3 + 2];
        verts.emplace_back(x, y, z);
    }
    int n_base_idx = normals.size();
    int nnorms     = obj_file.attributes.normals.size() / 3;
    normals.reserve(normals.size() + nnorms);
    for (int i = 0, i3 = 0; i < nnorms; i++, i3 += 3) {
        float x = obj_file.attributes.normals[i3 + 0];
        float y = obj_file.attributes.normals[i3 + 1];
        float z = obj_file.attributes.normals[i3 + 2];
        normals.emplace_back(x, y, z);
    }
    int uv_base_idx = uvs.size();
    int nuvs        = obj_file.attributes.texcoords.size() / 2;
    uvs.reserve(uvs.size() + nuvs);
    for (int i = 0, i2 = 0; i < nuvs; i++, i2 += 2) {
        float u = obj_file.attributes.texcoords[i2 + 0];
        float v = obj_file.attributes.texcoords[i2 + 1];
        uvs.emplace_back(u, v);
    }
    int ntris = 0;
    for (auto&& shape : obj_file.shapes) {
        int shapeShaderID = shaderID;
        if (!use_material_ids) {
            // if we couldn't find material names to use, try to lookup matching shaders by the mesh name instead
            auto it = shadermap.find(shape.name);
            if (it != shadermap.end()) {
                errhandler.infofmt("Found mesh {}", shape.name);
                shapeShaderID = it->second;
            } else {
                errhandler.infofmt("Mesh {} did not match any shader names",
                                   shape.name);
            }
        }
        OSL_ASSERT(shape.mesh.material_ids.size() == 0
                   || shape.mesh.material_ids.size()
                          == shape.mesh.indices.size() / 3);
        for (int i = 0, n = shape.mesh.indices.size(), f = 0; i < n;
             i += 3, f++) {
            int a = shape.mesh.indices[i + 0].position_index;
            int b = shape.mesh.indices[i + 1].position_index;
            int c = shape.mesh.indices[i + 2].position_index;
            triangles.emplace_back(
                TriangleIndices { base_idx + a, base_idx + b, base_idx + c });
            int na = shape.mesh.indices[i + 0].normal_index;
            int nb = shape.mesh.indices[i + 1].normal_index;
            int nc = shape.mesh.indices[i + 2].normal_index;
            // either all are valid, or none are valid
            OSL_DASSERT((na < 0 && nb < 0 && nc < 0)
                        || (na >= 0 && nb >= 0 && nc >= 0));
            n_triangles.emplace_back(TriangleIndices {
                na < 0 ? -1 : n_base_idx + na,
                na < 0 ? -1 : n_base_idx + nb,
                na < 0 ? -1 : n_base_idx + nc,
            });
            int ta = shape.mesh.indices[i + 0].texcoord_index;
            int tb = shape.mesh.indices[i + 1].texcoord_index;
            int tc = shape.mesh.indices[i + 2].texcoord_index;
            OSL_DASSERT((ta < 0 && tb < 0 && tc < 0)
                        || (ta >= 0 && tb >= 0 && tc >= 0));
            uv_triangles.emplace_back(TriangleIndices {
                ta < 0 ? -1 : uv_base_idx + ta, ta < 0 ? -1 : uv_base_idx + tb,
                ta < 0 ? -1 : uv_base_idx + tc });

            if (use_material_ids && !shape.mesh.material_ids.empty()) {
                // remap the material ID to our indexing
                int obj_mat_id = shape.mesh.material_ids[f];
                OSL_ASSERT(obj_mat_id >= 0);
                OSL_ASSERT(obj_mat_id < int(material_ids.size()));
                shaderids.emplace_back(material_ids[obj_mat_id]);
            } else
                shaderids.emplace_back(shapeShaderID);
            ntris++;
        }
        last_index.emplace_back(triangles.size());
    }
    double loadtime = timer();
    errhandler.infofmt("Parsed {} vertices and {} triangles from {} in {}",
                       nverts, ntris, filename,
                       OIIO::Strutil::timeintervalformat(loadtime, 2));
}

void
Scene::add_sphere(const Vec3& c, float r, int shaderID, int resolution)
{
    const int W    = 2 * resolution;
    const int H    = resolution;
    const int NV   = 2 + W * H;  // poles + grid = total vertex count
    int base_idx   = verts.size();
    int n_base_idx = normals.size();
    int t_base_idx = uvs.size();
    // vertices
    verts.emplace_back(c + Vec3(0, r, 0));  // pole +z
    normals.emplace_back(0, 1, 0);
    // W * H grid of points
    for (int y = 0; y < H; y++) {
        float t = float(y + 0.5f) / float(H);
        float z = cosf(t * float(M_PI));
        float q = sqrtf(std::max(0.0f, 1.0f - z * z));
        for (int x = 0; x < W; x++) {
            // match the previous parameterization
            const float a = float(2 * M_PI) * float(x) / float(W);
            const Vec3 n(q * -sinf(a), z, q * -cosf(a));
            verts.emplace_back(c + r * n);
            normals.emplace_back(n);
        }
    }
    verts.emplace_back(c - Vec3(0, r, 0));  // pole -z
    normals.emplace_back(0, -1, 0);
    // create rows for the poles (we use triangles instead of quads near the poles, so the top vertex should be evenly spaced)
    for (int y = 0; y < 2; y++)
        for (int x = 0; x < W; x++) {
            float s = float(x + 0.5f) / float(W);
            uvs.emplace_back(s, y);
        }
    // now create the rest of the plane with a regular spacing
    for (int y = 0; y < H; y++)
        for (int x = 0; x <= W; x++) {
            float s = float(x) / float(W);
            float t = float(y + 0.5f) / float(H);
            uvs.emplace_back(s, t);
        }

    for (int x0 = 0, x1 = W - 1; x0 < W; x1 = x0, x0++) {
        // tri to pole
        triangles.emplace_back(
            TriangleIndices { base_idx, base_idx + 1 + x1, base_idx + 1 + x0 });
        n_triangles.emplace_back(TriangleIndices {
            n_base_idx, n_base_idx + 1 + x1, n_base_idx + 1 + x0 });
        uv_triangles.emplace_back(
            TriangleIndices { t_base_idx + x1, t_base_idx + 2 * W + x1,
                              t_base_idx + 2 * W + x1 + 1 });

        shaderids.emplace_back(shaderID);
        for (int y = 0; y < H - 1; y++) {
            // quads
            int i00 = 1 + (x0 + W * (y + 0));
            int i10 = 1 + (x1 + W * (y + 0));
            int i11 = 1 + (x1 + W * (y + 1));
            int i01 = 1 + (x0 + W * (y + 1));

            triangles.emplace_back(TriangleIndices {
                base_idx + i00, base_idx + i10, base_idx + i11 });
            triangles.emplace_back(TriangleIndices {
                base_idx + i00, base_idx + i11, base_idx + i01 });

            n_triangles.emplace_back(TriangleIndices {
                n_base_idx + i00, n_base_idx + i10, n_base_idx + i11 });
            n_triangles.emplace_back(TriangleIndices {
                n_base_idx + i00, n_base_idx + i11, n_base_idx + i01 });

            int t00 = 2 * W + x1 + 1 + (W + 1) * (y + 0);
            int t10 = 2 * W + x1 + (W + 1) * (y + 0);
            int t11 = 2 * W + x1 + (W + 1) * (y + 1);
            int t01 = 2 * W + x1 + 1 + (W + 1) * (y + 1);
            uv_triangles.emplace_back(TriangleIndices {
                t_base_idx + t00, t_base_idx + t10, t_base_idx + t11 });
            uv_triangles.emplace_back(TriangleIndices {
                t_base_idx + t00, t_base_idx + t11, t_base_idx + t01 });

            shaderids.emplace_back(shaderID);
            shaderids.emplace_back(shaderID);
        }
        triangles.emplace_back(TriangleIndices { base_idx + NV - 1,
                                                 base_idx + NV - 1 - W + x0,
                                                 base_idx + NV - 1 - W + x1 });
        n_triangles.emplace_back(
            TriangleIndices { n_base_idx + NV - 1, n_base_idx + NV - 1 - W + x0,
                              n_base_idx + NV - 1 - W + x1 });
        uv_triangles.emplace_back(
            TriangleIndices { t_base_idx + W + x1,
                              t_base_idx + 2 * W + x1 + 1 + (W + 1) * (H - 1),
                              t_base_idx + 2 * W + x1 + (W + 1) * (H - 1) });
        shaderids.emplace_back(shaderID);
    }
    last_index.emplace_back(triangles.size());
}

void
Scene::add_quad(const Vec3& p, const Vec3& ex, const Vec3& ey, int shaderID,
                int resolution)
{
    // add vertices
    int base_idx   = verts.size();
    int t_base_idx = uvs.size();
    for (int v = 0; v <= resolution; v++)
        for (int u = 0; u <= resolution; u++) {
            float s = float(u) / float(resolution);
            float t = float(v) / float(resolution);
            verts.emplace_back(p + s * ex + t * ey);
            uvs.emplace_back(s, t);
        }
    for (int v = 0; v < resolution; v++)
        for (int u = 0; u < resolution; u++) {
            int i00 = (u + 0) + (v + 0) * (resolution + 1);
            int i10 = (u + 1) + (v + 0) * (resolution + 1);
            int i11 = (u + 1) + (v + 1) * (resolution + 1);
            int i01 = (u + 0) + (v + 1) * (resolution + 1);
            triangles.emplace_back(TriangleIndices {
                base_idx + i00, base_idx + i10, base_idx + i11 });
            triangles.emplace_back(TriangleIndices {
                base_idx + i00, base_idx + i11, base_idx + i01 });
            n_triangles.emplace_back(TriangleIndices { -1, -1, -1 });
            n_triangles.emplace_back(TriangleIndices { -1, -1, -1 });
            uv_triangles.emplace_back(TriangleIndices {
                t_base_idx + i00, t_base_idx + i10, t_base_idx + i11 });
            uv_triangles.emplace_back(TriangleIndices {
                t_base_idx + i00, t_base_idx + i11, t_base_idx + i01 });
            shaderids.emplace_back(shaderID);
            shaderids.emplace_back(shaderID);
        }
    last_index.emplace_back(triangles.size());
}


OSL_NAMESPACE_END
